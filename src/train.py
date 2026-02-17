"""Main training script for AdDiffusion agent.

Usage:
    uv run python src/train.py --config configs/default.yaml
    uv run python src/train.py --config configs/default.yaml training.total_iterations=100

References:
    - research.md §3.6 (training procedure)
    - plan.md Phase 1, Week 4 (training setup)
    - experiment.md EXP-T01 (training protocol)
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
import yaml

from src.agent.episode import EpisodeRunner
from src.agent.networks import PolicyNetwork, ValueNetwork
from src.agent.ppo import PPOConfig, PPOTrainer, episodes_to_batch
from src.agent.state import StateExtractor
from src.diffusion.attention import UNetAttentionExtractor
from src.diffusion.pipeline import AdaptiveDiffusionPipeline
from src.rewards.reward import RewardComputer, RewardConfig


def load_config(config_path: str, overrides: list[str] = None) -> dict:
    """Load YAML config with optional dotted-key overrides."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    if overrides:
        for override in overrides:
            if "=" not in override:
                continue
            key, value = override.split("=", 1)
            parts = key.split(".")
            d = cfg
            for p in parts[:-1]:
                d = d.setdefault(p, {})
            # Parse value type
            if value.lower() in ("true", "false"):
                value = value.lower() == "true"
            else:
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        pass
            d[parts[-1]] = value

    return cfg


def load_prompts(path: str, max_prompts: int = None) -> list[str]:
    """Load prompts from a JSON file (COCO captions format or list)."""
    with open(path) as f:
        data = json.load(f)

    if isinstance(data, list):
        # Simple list of strings or dicts with 'prompt' key
        prompts = []
        for item in data:
            if isinstance(item, str):
                prompts.append(item)
            elif isinstance(item, dict):
                prompts.append(item.get("prompt", item.get("caption", item.get("text", ""))))
        return prompts[:max_prompts] if max_prompts else prompts

    if "annotations" in data:
        # COCO captions format
        prompts = [ann["caption"] for ann in data["annotations"]]
        return prompts[:max_prompts] if max_prompts else prompts

    raise ValueError(f"Unknown prompt format in {path}")


def select_prompt(
    prompts: list[str],
    iteration: int,
    curriculum: dict,
) -> str:
    """Select a prompt based on curriculum schedule (D-32).

    Simple heuristic: shorter prompts = simpler.
    """
    simple_until = curriculum.get("simple_until", 500)
    mixed_until = curriculum.get("mixed_until", 1200)

    # Sort by length as complexity proxy
    sorted_prompts = sorted(prompts, key=len)
    n = len(sorted_prompts)

    if iteration < simple_until:
        # Simple: bottom third by length
        pool = sorted_prompts[: n // 3]
    elif iteration < mixed_until:
        # Mixed: bottom two-thirds
        pool = sorted_prompts[: 2 * n // 3]
    else:
        # Full distribution
        pool = sorted_prompts

    return random.choice(pool)


def train(cfg: dict):
    """Main training loop."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if cfg["model"]["dtype"] == "float16" else torch.float32

    print("=" * 60)
    print("AdDiffusion Training")
    print("=" * 60)

    # Initialize components
    print("Loading diffusion pipeline...")
    pipeline = AdaptiveDiffusionPipeline.from_pretrained(
        cfg["model"]["model_id"],
        scheduler_name=cfg["model"]["scheduler"],
        device=device,
        dtype=dtype,
    )

    print("Setting up attention extractor...")
    attn_ext = UNetAttentionExtractor(latent_h=64, latent_w=64)
    attn_ext.hook_with_processor(pipeline.unet)

    print("Setting up state extractor...")
    state_ext = StateExtractor(device=device, dtype=dtype)

    print("Setting up reward computer...")
    reward_cfg = RewardConfig(
        alpha_1=cfg["reward"]["alpha_1"],
        alpha_2=cfg["reward"]["alpha_2"],
        alpha_3=cfg["reward"]["alpha_3"],
        alpha_4=cfg["reward"]["alpha_4"],
        c_nfe=cfg["reward"]["c_nfe"],
        beta_1=cfg["reward"]["beta_1"],
        beta_2=cfg["reward"]["beta_2"],
        beta_3=cfg["reward"]["beta_3"],
        clip_norm=cfg["reward"]["clip_norm"],
        aesthetic_norm=cfg["reward"]["aesthetic_norm"],
        image_reward_norm=cfg["reward"]["image_reward_norm"],
        normalize=cfg["reward"]["normalize"],
        refine_k=cfg["refinement"]["k"],
    )
    reward_computer = RewardComputer(config=reward_cfg, device=device)

    print("Setting up agent networks...")
    policy = PolicyNetwork(
        input_dim=cfg["agent"]["state_dim"],
        hidden_dim=cfg["agent"]["hidden_dim"],
        num_actions=cfg["agent"]["num_actions"],
    ).to(device).float()

    value_net = ValueNetwork(
        input_dim=cfg["agent"]["state_dim"],
        hidden_dim=cfg["agent"]["hidden_dim"],
    ).to(device).float()

    ppo_cfg = PPOConfig(
        lr=cfg["ppo"]["lr"],
        gamma_d=cfg["ppo"]["gamma_d"],
        gae_lambda=cfg["ppo"]["gae_lambda"],
        clip_epsilon=cfg["ppo"]["clip_epsilon"],
        entropy_coeff=cfg["ppo"]["entropy_coeff"],
        value_coeff=cfg["ppo"]["value_coeff"],
        max_grad_norm=cfg["ppo"]["max_grad_norm"],
        ppo_epochs=cfg["ppo"]["ppo_epochs"],
        mini_batch_size=cfg["ppo"]["mini_batch_size"],
    )
    trainer = PPOTrainer(policy, value_net, config=ppo_cfg, device=device)

    # Episode runner
    runner = EpisodeRunner(
        pipeline=pipeline,
        state_extractor=state_ext,
        attention_extractor=attn_ext,
        warmup_steps=cfg["agent"]["warmup_steps"],
        guidance_scale=cfg["model"]["guidance_scale"],
        refine_k=cfg["refinement"]["k"],
        refine_r_noise=cfg["refinement"]["r_noise"],
        mask_threshold=cfg["refinement"]["mask_threshold"],
        blur_sigma=cfg["refinement"]["blur_sigma"],
    )

    # Load prompts
    print(f"Loading prompts from {cfg['training']['prompt_dataset']}...")
    prompts = load_prompts(cfg["training"]["prompt_dataset"])
    print(f"  Loaded {len(prompts)} prompts")

    # Set seed
    seed = cfg["training"]["seed"]
    random.seed(seed)
    torch.manual_seed(seed)

    # Training loop
    total_iters = cfg["training"]["total_iterations"]
    batch_size = cfg["ppo"]["batch_size"]
    num_steps = cfg["training"]["num_steps"]
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)

    print(f"\nStarting training: {total_iters} iterations, batch={batch_size}")
    print(f"  N_max={num_steps}, warmup={cfg['agent']['warmup_steps']}")
    print()

    for iteration in range(1, total_iters + 1):
        iter_start = time.time()

        # Collect batch of episodes
        episodes = []
        batch_nfe = 0
        batch_rewards = 0

        for b in range(batch_size):
            prompt = select_prompt(prompts, iteration, cfg["training"]["curriculum"])
            ep_seed = seed + iteration * batch_size + b

            # Create reward function closure
            prev_metrics = {}

            def reward_fn(prev_z0, curr_z0, action, step_index, n_max, is_terminal, prompt, decoded_image, _prev=prev_metrics):
                prev_img = _prev.get("prev_decoded")
                reward, metrics = reward_computer.compute_reward(
                    prev_image=prev_img,
                    curr_image=decoded_image,
                    prompt=prompt,
                    action=action,
                    is_terminal=is_terminal,
                    prev_clip=_prev.get("clip_score"),
                    prev_ir=_prev.get("image_reward"),
                )
                _prev["prev_decoded"] = decoded_image
                _prev["clip_score"] = metrics.get("clip_score")
                _prev["image_reward"] = metrics.get("image_reward")
                return reward

            ep = runner.run_episode(
                prompt=prompt,
                policy=policy,
                value_net=value_net,
                num_steps=num_steps,
                seed=ep_seed,
                reward_fn=reward_fn,
            )
            episodes.append(ep)
            batch_nfe += ep.total_nfe
            batch_rewards += sum(t.reward for t in ep.transitions)

        # Convert to PPO batch
        batch = episodes_to_batch(episodes, gamma=ppo_cfg.gamma_d, lam=ppo_cfg.gae_lambda)

        # PPO update
        ppo_metrics = trainer.update(batch)

        iter_time = time.time() - iter_start

        # Logging
        if iteration % cfg["training"]["log_every"] == 0:
            avg_nfe = batch_nfe / batch_size
            avg_reward = batch_rewards / batch_size

            # Action distribution
            all_actions = []
            for ep in episodes:
                all_actions.extend(ep.action_sequence)
            n_actions = len(all_actions)
            action_dist = {
                "continue": all_actions.count("continue") / max(n_actions, 1),
                "stop": all_actions.count("stop") / max(n_actions, 1),
                "refine": all_actions.count("refine") / max(n_actions, 1),
            }

            print(
                f"Iter {iteration:4d}/{total_iters} | "
                f"reward={avg_reward:+.3f} | "
                f"NFE={avg_nfe:.1f} | "
                f"actions: C={action_dist['continue']:.2f} S={action_dist['stop']:.2f} R={action_dist['refine']:.2f} | "
                f"loss={ppo_metrics['total_loss']:.4f} | "
                f"entropy={ppo_metrics['entropy']:.3f} | "
                f"clip_frac={ppo_metrics['clip_fraction']:.3f} | "
                f"time={iter_time:.1f}s"
            )

        # Checkpoint
        if iteration % cfg["training"]["checkpoint_every"] == 0:
            ckpt_path = checkpoint_dir / f"agent_v1_iter{iteration}.pt"
            torch.save({
                "iteration": iteration,
                "policy_state_dict": policy.state_dict(),
                "value_state_dict": value_net.state_dict(),
                "optimizer_state_dict": trainer.optimizer.state_dict(),
                "config": cfg,
            }, ckpt_path)
            print(f"  Checkpoint saved: {ckpt_path}")

    # Final checkpoint
    final_path = checkpoint_dir / "agent_v1_final.pt"
    torch.save({
        "iteration": total_iters,
        "policy_state_dict": policy.state_dict(),
        "value_state_dict": value_net.state_dict(),
        "optimizer_state_dict": trainer.optimizer.state_dict(),
        "config": cfg,
    }, final_path)
    print(f"\nTraining complete. Final checkpoint: {final_path}")


def main():
    parser = argparse.ArgumentParser(description="AdDiffusion Agent Training")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args, overrides = parser.parse_known_args()

    cfg = load_config(args.config, overrides)
    train(cfg)


if __name__ == "__main__":
    main()
