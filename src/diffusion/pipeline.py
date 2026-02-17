"""Step-by-step diffusion inference wrapper.

Wraps a pretrained latent diffusion model to allow per-step agent control.
All operations are in latent space (z_t at h x w resolution). Pixel-space
images are obtained only via explicit VAE decode.

References:
    - research.md §3.1, §3.2 (episode structure)
    - discovery.md D-14 (latent space), D-15 (CFG), D-18 (episode structure)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F
from diffusers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler,
    PNDMScheduler,
    StableDiffusionPipeline,
    UniPCMultistepScheduler,
)


SCHEDULER_REGISTRY = {
    "ddim": DDIMScheduler,
    "dpm_solver": DPMSolverMultistepScheduler,
    "euler": EulerDiscreteScheduler,
    "unipc": UniPCMultistepScheduler,
    "pndm": PNDMScheduler,
}


@dataclass
class StepOutput:
    """Result of a single denoising step."""

    z_t: torch.Tensor  # Current noisy latent
    z0_pred: torch.Tensor  # One-step clean prediction in latent space
    timestep: int  # Current diffusion timestep value
    step_index: int  # Index in the scheduler's timestep sequence
    nfe: int  # NFE consumed by this step


@dataclass
class PipelineState:
    """Mutable state of a running denoising episode."""

    z_t: torch.Tensor  # Current noisy latent (h x w x 4)
    timesteps: torch.Tensor  # Full scheduler timestep sequence
    step_index: int = 0  # Current position in timestep sequence
    total_nfe: int = 0  # Cumulative NFE for this episode
    prompt_embeds: Optional[torch.Tensor] = None
    negative_prompt_embeds: Optional[torch.Tensor] = None
    text_input_ids: Optional[torch.Tensor] = None  # For attention token mapping
    is_done: bool = False
    history: list[StepOutput] = field(default_factory=list)


class AdaptiveDiffusionPipeline:
    """Wraps a pretrained SD pipeline for step-by-step agent-controlled inference.

    Usage:
        pipe = AdaptiveDiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5")
        state = pipe.prepare("a photo of a cat", num_steps=50, seed=42)
        while not state.is_done:
            step_out = pipe.denoise_step(state)
            # Agent decides action based on step_out
            # If stop: image = pipe.decode(step_out.z0_pred)
            # If refine: pipe.refine_regions(state, mask, ...)
    """

    def __init__(
        self,
        sd_pipeline: StableDiffusionPipeline,
        scheduler_name: str = "ddim",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        self.pipe = sd_pipeline
        self.unet = sd_pipeline.unet
        self.vae = sd_pipeline.vae
        self.tokenizer = sd_pipeline.tokenizer
        self.text_encoder = sd_pipeline.text_encoder
        self.device = device
        self.dtype = dtype

        # Set scheduler
        if scheduler_name in SCHEDULER_REGISTRY:
            self.pipe.scheduler = SCHEDULER_REGISTRY[scheduler_name].from_config(
                self.pipe.scheduler.config
            )
        self.scheduler = self.pipe.scheduler

        # Latent scaling factor (0.18215 for SD 1.5)
        self.vae_scale_factor = self.vae.config.scaling_factor

        # Spatial dimensions in latent space
        self.latent_channels = self.unet.config.in_channels  # 4
        self.downsample_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)  # 8

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        scheduler_name: str = "ddim",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ) -> "AdaptiveDiffusionPipeline":
        sd_pipe = StableDiffusionPipeline.from_pretrained(
            model_id, torch_dtype=dtype, safety_checker=None
        ).to(device)
        return cls(sd_pipe, scheduler_name=scheduler_name, device=device, dtype=dtype)

    def get_latent_dims(self, height: int = 512, width: int = 512) -> tuple[int, int]:
        """Return latent spatial dimensions (h, w) for a given pixel resolution."""
        return height // self.downsample_factor, width // self.downsample_factor

    @torch.no_grad()
    def encode_prompt(
        self,
        prompt: str,
        negative_prompt: str = "",
        guidance_scale: float = 7.5,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode text prompt to embeddings.

        Returns:
            prompt_embeds: (1, seq_len, hidden_dim)
            negative_prompt_embeds: (1, seq_len, hidden_dim) — for CFG
            text_input_ids: (1, seq_len) — token IDs for attention mapping
        """
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.device)
        prompt_embeds = self.text_encoder(text_input_ids)[0]

        # Negative prompt for CFG (D-15: CFG doubles network evals per step)
        uncond_inputs = self.tokenizer(
            negative_prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        negative_prompt_embeds = self.text_encoder(
            uncond_inputs.input_ids.to(self.device)
        )[0]

        return prompt_embeds, negative_prompt_embeds, text_input_ids

    @torch.no_grad()
    def prepare(
        self,
        prompt: str,
        num_steps: int = 50,
        guidance_scale: float = 7.5,
        height: int = 512,
        width: int = 512,
        seed: Optional[int] = None,
        negative_prompt: str = "",
    ) -> PipelineState:
        """Initialize an episode: encode prompt, sample noise, set up scheduler.

        Returns a PipelineState that tracks the evolving latent.
        """
        # Encode prompt
        prompt_embeds, negative_prompt_embeds, text_input_ids = self.encode_prompt(
            prompt, negative_prompt, guidance_scale
        )

        # Set up scheduler timesteps
        self.scheduler.set_timesteps(num_steps, device=self.device)
        timesteps = self.scheduler.timesteps

        # Sample initial noise in latent space (D-14)
        h, w = self.get_latent_dims(height, width)
        generator = torch.Generator(device=self.device)
        if seed is not None:
            generator.manual_seed(seed)
        z_t = torch.randn(
            (1, self.latent_channels, h, w),
            generator=generator,
            device=self.device,
            dtype=self.dtype,
        )

        # Scale initial noise by scheduler's init_noise_sigma
        z_t = z_t * self.scheduler.init_noise_sigma

        return PipelineState(
            z_t=z_t,
            timesteps=timesteps,
            step_index=0,
            total_nfe=0,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            text_input_ids=text_input_ids,
        )

    @torch.no_grad()
    def denoise_step(
        self,
        state: PipelineState,
        guidance_scale: float = 7.5,
    ) -> StepOutput:
        """Execute one denoising step. Returns the step output without advancing state.

        The caller is responsible for calling `advance_state()` to commit the step,
        or taking a different action (stop, refine).
        """
        if state.is_done:
            raise RuntimeError("Episode is already done")
        if state.step_index >= len(state.timesteps):
            raise RuntimeError("No more timesteps in schedule")

        t = state.timesteps[state.step_index]

        # Expand latent for CFG (conditional + unconditional)
        latent_model_input = torch.cat([state.z_t, state.z_t])
        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

        # Combined prompt embeddings for CFG
        combined_embeds = torch.cat(
            [state.negative_prompt_embeds, state.prompt_embeds]
        )

        # UNet forward pass (2 network evals due to CFG — D-15)
        noise_pred = self.unet(
            latent_model_input,
            t,
            encoder_hidden_states=combined_embeds,
        ).sample

        # CFG combination
        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_pred_cond - noise_pred_uncond
        )

        # Scheduler step — produces denoised latent and predicted clean latent
        scheduler_output = self.scheduler.step(noise_pred, t, state.z_t)
        z_next = scheduler_output.prev_sample

        # One-step clean prediction (z0_pred) — used for state features
        # Most schedulers provide this as pred_original_sample
        if hasattr(scheduler_output, "pred_original_sample") and scheduler_output.pred_original_sample is not None:
            z0_pred = scheduler_output.pred_original_sample
        else:
            # Fallback: compute manually from noise prediction
            alpha_prod_t = self.scheduler.alphas_cumprod[t.cpu().long()].to(state.z_t.device)
            z0_pred = (state.z_t - (1 - alpha_prod_t).sqrt() * noise_pred) / alpha_prod_t.sqrt()

        return StepOutput(
            z_t=z_next,
            z0_pred=z0_pred,
            timestep=t.item(),
            step_index=state.step_index,
            nfe=1,  # 1 scheduler step (D-15: 2 network evals, but NFE = scheduler steps)
        )

    def advance_state(self, state: PipelineState, step_output: StepOutput) -> None:
        """Commit a continue action: advance the state to the next step."""
        state.z_t = step_output.z_t
        state.step_index += 1
        state.total_nfe += step_output.nfe
        state.history.append(step_output)

        if state.step_index >= len(state.timesteps):
            state.is_done = True

    @torch.no_grad()
    def decode(self, z0: torch.Tensor) -> torch.Tensor:
        """Decode latent to pixel-space image. Returns tensor in [0, 1]."""
        z0 = z0 / self.vae_scale_factor
        image = self.vae.decode(z0).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        return image

    @torch.no_grad()
    def add_noise(
        self,
        z0: torch.Tensor,
        timestep: int,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Re-noise a clean latent to a given timestep level.

        Used for post-refinement state transition (D-19):
        z_{t_{i+1}} = sqrt(alpha_bar) * z0_refined + sqrt(1 - alpha_bar) * eps
        """
        noise = torch.randn(z0.shape, dtype=z0.dtype, device=z0.device, generator=generator)
        t_tensor = torch.tensor([timestep], device=self.device, dtype=torch.long)
        z_noisy = self.scheduler.add_noise(z0, noise, t_tensor)
        return z_noisy

    @torch.no_grad()
    def unet_forward(
        self,
        z_t: torch.Tensor,
        t: int | torch.Tensor,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: torch.Tensor,
        guidance_scale: float = 7.5,
    ) -> torch.Tensor:
        """Single UNet forward pass with CFG. Returns predicted noise.

        Utility for RegionRefine which needs raw denoising without scheduler stepping.
        """
        if isinstance(t, int):
            t = torch.tensor([t], device=self.device, dtype=torch.long)

        latent_model_input = torch.cat([z_t, z_t])
        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
        combined_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        noise_pred = self.unet(
            latent_model_input, t, encoder_hidden_states=combined_embeds
        ).sample

        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_pred_cond - noise_pred_uncond
        )
        return noise_pred

    def predict_clean(
        self,
        z_t: torch.Tensor,
        noise_pred: torch.Tensor,
        t: int | torch.Tensor,
    ) -> torch.Tensor:
        """Compute one-step clean prediction from noisy latent and predicted noise."""
        if isinstance(t, int):
            t = torch.tensor([t], device=self.device, dtype=torch.long)
        alpha_prod_t = self.scheduler.alphas_cumprod[t.cpu().long()].to(z_t.device)
        # Reshape for broadcasting
        while alpha_prod_t.dim() < z_t.dim():
            alpha_prod_t = alpha_prod_t.unsqueeze(-1)
        z0_pred = (z_t - (1 - alpha_prod_t).sqrt() * noise_pred) / alpha_prod_t.sqrt()
        return z0_pred

    @property
    def n_max(self) -> int:
        """Maximum number of steps for the current scheduler configuration."""
        return len(self.scheduler.timesteps)
