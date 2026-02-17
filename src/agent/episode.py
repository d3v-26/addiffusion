"""Episode loop: agent interacts with the diffusion pipeline.

Implements the full episode structure from research.md §3.2:
- 3 mandatory warmup continues (D-16)
- Agent decides continue/stop/refine at each step after warmup
- Tracks action sequence, NFE, rewards, state features

References:
    - research.md §3.2 (episode structure)
    - discovery.md D-16 (warmup for unreliable early predictions)
    - discovery.md D-18 (episode/scheduler interaction)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch

from src.agent.networks import (
    ACTION_CONTINUE,
    ACTION_NAMES,
    ACTION_REFINE,
    ACTION_STOP,
    PolicyNetwork,
    ValueNetwork,
)
from src.agent.state import StateExtractor
from src.diffusion.attention import AttentionExtractor
from src.diffusion.pipeline import AdaptiveDiffusionPipeline, PipelineState, StepOutput
from src.diffusion.refine import apply_refine_action


@dataclass
class Transition:
    """A single (s, a, r, s', done) transition."""

    state_features: torch.Tensor  # phi(s_i): (1, d)
    action: int  # 0=continue, 1=stop, 2=refine
    log_prob: float  # log pi(a|s)
    value: float  # V(s)
    reward: float  # r(s, a, s')
    next_state_features: Optional[torch.Tensor] = None  # phi(s_{i+1})
    done: bool = False
    nfe: int = 0  # NFE consumed by this action
    timestep: int = 0  # Diffusion timestep
    z0_pred: Optional[torch.Tensor] = None  # One-step prediction (for reward computation)


@dataclass
class EpisodeResult:
    """Result of a complete episode."""

    transitions: list[Transition] = field(default_factory=list)
    final_image: Optional[torch.Tensor] = None  # Decoded pixel-space image (1, 3, H, W)
    final_z0: Optional[torch.Tensor] = None  # Final latent prediction
    total_nfe: int = 0
    total_steps: int = 0  # Number of agent decisions (excluding warmup)
    prompt: str = ""
    action_sequence: list[str] = field(default_factory=list)


class EpisodeRunner:
    """Runs a single episode: agent controls diffusion inference.

    Episode flow:
        1. Prepare diffusion pipeline (sample noise, set up scheduler)
        2. Run warmup_steps mandatory continues (D-16)
        3. At each step, extract state features, query agent, execute action
        4. Collect transitions for PPO training
        5. Return final image and trajectory data
    """

    def __init__(
        self,
        pipeline: AdaptiveDiffusionPipeline,
        state_extractor: StateExtractor,
        attention_extractor: AttentionExtractor,
        warmup_steps: int = 3,
        guidance_scale: float = 7.5,
        refine_k: int = 2,
        refine_r_noise: float = 0.5,
        mask_threshold: float = 0.5,
        blur_sigma: float = 3.0,
    ):
        self.pipeline = pipeline
        self.state_extractor = state_extractor
        self.attention_extractor = attention_extractor
        self.warmup_steps = warmup_steps
        self.guidance_scale = guidance_scale
        self.refine_k = refine_k
        self.refine_r_noise = refine_r_noise
        self.mask_threshold = mask_threshold
        self.blur_sigma = blur_sigma

    @torch.no_grad()
    def run_episode(
        self,
        prompt: str,
        policy: PolicyNetwork,
        value_net: ValueNetwork,
        num_steps: int = 50,
        seed: Optional[int] = None,
        deterministic: bool = False,
        reward_fn=None,
    ) -> EpisodeResult:
        """Run a complete episode.

        Args:
            prompt: Text prompt.
            policy: Policy network for action selection.
            value_net: Value network for baseline estimation.
            num_steps: Maximum denoising steps (N_max).
            seed: Random seed for reproducibility.
            deterministic: If True, use argmax actions (for evaluation).
            reward_fn: Optional callable(prev_z0, curr_z0, action, step_info) → float.
                       If None, rewards are set to 0 (filled in later).

        Returns:
            EpisodeResult with transitions, final image, and metadata.
        """
        result = EpisodeResult(prompt=prompt)
        self.state_extractor.reset_cache()

        # Prepare diffusion state
        pipe_state = self.pipeline.prepare(
            prompt, num_steps=num_steps, seed=seed, guidance_scale=self.guidance_scale,
        )
        n_max = len(pipe_state.timesteps)

        prev_z0_pred = None
        prev_decoded = None

        while not pipe_state.is_done:
            step_idx = pipe_state.step_index
            is_warmup = step_idx < self.warmup_steps

            # Debug: verify dtype consistency before UNet call
            assert pipe_state.z_t.dtype == self.pipeline.dtype, (
                f"[BUG] z_t dtype={pipe_state.z_t.dtype} != pipeline dtype={self.pipeline.dtype} "
                f"at step {step_idx}. Likely a float32 promotion from alphas_cumprod math."
            )

            # Run denoising step to get current prediction + attention maps
            self.attention_extractor.clear()
            step_out = self.pipeline.denoise_step(pipe_state, guidance_scale=self.guidance_scale)

            # Decode one-step prediction for state features
            decoded_image = self.pipeline.decode(step_out.z0_pred)

            # Extract state features
            phi = self.state_extractor.extract(
                decoded_image=decoded_image,
                prompt=prompt,
                timestep=step_out.timestep,
                step_ratio=step_idx / n_max,
                nfe_ratio=pipe_state.total_nfe / n_max,
            )

            if is_warmup:
                # Mandatory continue during warmup (D-16)
                action = ACTION_CONTINUE
                log_prob = 0.0
                value = 0.0
            else:
                # Agent decides
                action_tensor, log_prob_tensor, _ = policy.get_action(phi, deterministic=deterministic)
                value_tensor = value_net(phi)
                action = action_tensor.item()
                log_prob = log_prob_tensor.item()
                value = value_tensor.item()

            # Execute action
            nfe = 0
            z0_for_reward = step_out.z0_pred

            if action == ACTION_CONTINUE:
                self.pipeline.advance_state(pipe_state, step_out)
                nfe = 1
                result.action_sequence.append("continue")

            elif action == ACTION_STOP:
                # Early termination — decode and return
                result.final_z0 = step_out.z0_pred
                result.final_image = decoded_image
                pipe_state.is_done = True
                nfe = 0
                result.action_sequence.append("stop")

            elif action == ACTION_REFINE:
                z0_refined, refine_nfe = apply_refine_action(
                    self.pipeline, pipe_state, self.attention_extractor,
                    k=self.refine_k, r_noise=self.refine_r_noise,
                    mask_threshold=self.mask_threshold, blur_sigma=self.blur_sigma,
                    guidance_scale=self.guidance_scale,
                )
                nfe = refine_nfe
                z0_for_reward = z0_refined
                result.action_sequence.append("refine")

            # Compute reward if reward function provided
            reward = 0.0
            if reward_fn is not None:
                reward = reward_fn(
                    prev_z0=prev_z0_pred,
                    curr_z0=z0_for_reward,
                    action=action,
                    step_index=step_idx,
                    n_max=n_max,
                    is_terminal=pipe_state.is_done,
                    prompt=prompt,
                    decoded_image=decoded_image,
                )

            # Record transition
            transition = Transition(
                state_features=phi.cpu(),
                action=action,
                log_prob=log_prob,
                value=value,
                reward=reward,
                done=pipe_state.is_done,
                nfe=nfe,
                timestep=step_out.timestep,
                z0_pred=z0_for_reward.cpu() if z0_for_reward is not None else None,
            )
            result.transitions.append(transition)

            prev_z0_pred = z0_for_reward
            prev_decoded = decoded_image

        # If we didn't stop early, decode the final prediction
        if result.final_image is None and pipe_state.history:
            result.final_z0 = pipe_state.history[-1].z0_pred
            result.final_image = self.pipeline.decode(result.final_z0)

        result.total_nfe = pipe_state.total_nfe
        result.total_steps = max(0, len(result.transitions) - self.warmup_steps)

        return result
