"""Learning rate sweep for PPO training (Phase 2, Week 7).

Runs 500 PPO training iterations for each candidate learning rate, using the
default reward config. Reports mean reward and NFE at final 50 iterations.

Usage:
    uv run python scripts/tune_lr.py \
        --lrs 1e-4 3e-4 1e-3 \
        --n_iters 500 \
        --prompts_file data/coco/prompts_1k.json \
        --output_dir outputs/tuning/lr_sweep/
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
import traceback

import matplotlib.pyplot as plt
import torch

from src.agent.episode import EpisodeRunner
from src.agent.networks import PolicyNetwork, ValueNetwork
from src.agent.ppo import PPOConfig, PPOTrainer, episodes_to_batch
from src.agent.state import StateExtractor
from src.diffusion.attention import UNetAttentionExtractor
from src.diffusion.pipeline import AdaptiveDiffusionPipeline
from src.rewards.reward import RewardComputer, RewardConfig


MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
BATCH_SIZE = 4  # prompts per iteration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Learning rate sweep for PPO training.")
    parser.add_argument(
        "--lrs",
        nargs="+",
        type=float,
        default=[1e-4, 3e-4, 1e-3],
        help="Learning rates to sweep (space-separated floats).",
    )
    parser.add_argument("--n_iters", type=int, default=500, help="Training iterations per LR.")
    parser.add_argument(
        "--prompts_file",
        type=str,
        default="data/coco/prompts_1k.json",
        help="Path to JSON file with list of prompt strings.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/tuning/lr_sweep/",
        help="Directory for results and plots.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device string.")
    parser.add_argument(
        "--max_prompts",
        type=int,
        default=None,
        help="Cap the number of prompts loaded. Shuffles then slices. Default: use all.",
    )
    return parser.parse_args()


def load_prompts(path: str) -> list[str]:
    """Load prompts from a JSON file.

    Accepts either:
    - A plain list of strings: ["prompt1", "prompt2", ...]
    - COCO captions format: {"annotations": [{"caption": "..."}, ...]}
    """
    with open(path, "r") as fh:
        data = json.load(fh)
    if isinstance(data, list):
        return [str(p) for p in data]
    if isinstance(data, dict) and "annotations" in data:
        return [str(ann["caption"]) for ann in data["annotations"] if "caption" in ann]
    raise ValueError(f"Unrecognised prompts format in {path}.")


def make_reward_fn(reward_computer: RewardComputer, pipeline: AdaptiveDiffusionPipeline):
    """Return a reward callable compatible with EpisodeRunner.run_episode."""
    return reward_computer.make_episode_reward_fn()


def run_sweep_for_lr(
    lr: float,
    prompts: list[str],
    n_iters: int,
    device: str,
    seed: int,
) -> tuple[list[float], list[float]]:
    """Run PPO training for `n_iters` iterations at a given learning rate.

    Returns:
        rewards_per_iter: Mean episode total reward for each iteration.
        nfe_per_iter: Mean episode NFE for each iteration.
    """
    torch.manual_seed(seed)
    random.seed(seed)

    print(f"\n[tune_lr] Loading pipeline for LR={lr:.2e} ...")
    pipeline = AdaptiveDiffusionPipeline.from_pretrained(MODEL_ID, device=device)

    state_extractor = StateExtractor(device=device)
    attention_extractor = UNetAttentionExtractor(latent_h=64, latent_w=64)
    attention_extractor.hook(pipeline.unet)

    policy = PolicyNetwork(input_dim=StateExtractor.TOTAL_DIM, hidden_dim=512).to(device)
    value_net = ValueNetwork(input_dim=StateExtractor.TOTAL_DIM, hidden_dim=512).to(device)

    ppo_config = PPOConfig(lr=lr)
    trainer = PPOTrainer(policy=policy, value_net=value_net, config=ppo_config, device=device)

    reward_config = RewardConfig()
    reward_computer = RewardComputer(config=reward_config, device=device)
    reward_fn = make_reward_fn(reward_computer, pipeline)

    episode_runner = EpisodeRunner(
        pipeline=pipeline,
        state_extractor=state_extractor,
        attention_extractor=attention_extractor,
    )

    rewards_per_iter: list[float] = []
    nfe_per_iter: list[float] = []

    for iteration in range(n_iters):
        batch_prompts = random.sample(prompts, min(BATCH_SIZE, len(prompts)))
        episodes = []

        for i, prompt in enumerate(batch_prompts):
            try:
                ep = episode_runner.run_episode(
                    prompt=prompt,
                    policy=policy,
                    value_net=value_net,
                    num_steps=50,
                    seed=seed + iteration * BATCH_SIZE + i,
                    reward_fn=reward_fn,
                )
                episodes.append(ep)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"  [OOM] iter={iteration}, prompt_idx={i} — skipping.")
                    torch.cuda.empty_cache()
                else:
                    print(f"  [ERROR] iter={iteration}: {e}")
                    traceback.print_exc()

        if not episodes:
            rewards_per_iter.append(float("nan"))
            nfe_per_iter.append(float("nan"))
            continue

        # PPO update
        try:
            traj_batch = episodes_to_batch(
                episodes, gamma=ppo_config.gamma_d, lam=ppo_config.gae_lambda
            )
            trainer.update(traj_batch)
        except (ValueError, RuntimeError) as e:
            print(f"  [TRAIN ERROR] iter={iteration}: {e}")

        ep_rewards = [
            sum(t.reward for t in ep.transitions)
            for ep in episodes
            if ep.transitions
        ]
        ep_nfes = [ep.total_nfe for ep in episodes]

        mean_reward = sum(ep_rewards) / len(ep_rewards) if ep_rewards else float("nan")
        mean_nfe = sum(ep_nfes) / len(ep_nfes) if ep_nfes else float("nan")

        rewards_per_iter.append(mean_reward)
        nfe_per_iter.append(mean_nfe)

        if (iteration + 1) % 50 == 0:
            print(
                f"  iter {iteration + 1}/{n_iters}  "
                f"mean_reward={mean_reward:.4f}  mean_nfe={mean_nfe:.1f}"
            )

    # Clean up hooks
    attention_extractor.remove_hooks()

    return rewards_per_iter, nfe_per_iter


def final_50_stats(values: list[float]) -> tuple[float, float]:
    """Return mean and std of the last 50 non-nan values."""
    tail = [v for v in values[-50:] if v == v]  # filter nan
    if not tail:
        return float("nan"), float("nan")
    n = len(tail)
    mean = sum(tail) / n
    variance = sum((v - mean) ** 2 for v in tail) / n
    std = variance ** 0.5
    return mean, std


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[tune_lr] Loading prompts from {args.prompts_file} ...")
    prompts = load_prompts(args.prompts_file)
    print(f"[tune_lr] Loaded {len(prompts)} prompts.")
    if args.max_prompts is not None and len(prompts) > args.max_prompts:
        rng = random.Random(args.seed)
        rng.shuffle(prompts)
        prompts = prompts[: args.max_prompts]
        print(f"[tune_lr] Capped prompts to {len(prompts)} (--max_prompts {args.max_prompts})")

    all_rewards: dict[str, list[float]] = {}
    all_nfes: dict[str, list[float]] = {}
    summary: dict[str, dict] = {}

    for lr in args.lrs:
        lr_key = f"{lr:.2e}"
        rewards, nfes = run_sweep_for_lr(
            lr=lr,
            prompts=prompts,
            n_iters=args.n_iters,
            device=args.device,
            seed=args.seed,
        )
        all_rewards[lr_key] = rewards
        all_nfes[lr_key] = nfes

        mean_r, std_r = final_50_stats(rewards)
        mean_nfe, _ = final_50_stats(nfes)

        summary[lr_key] = {
            "final_mean_reward": round(mean_r, 6),
            "final_std_reward": round(std_r, 6),
            "final_mean_nfe": round(mean_nfe, 2),
        }
        print(
            f"\n[tune_lr] LR={lr_key}  "
            f"final_mean_reward={mean_r:.4f}  "
            f"final_mean_nfe={mean_nfe:.1f}"
        )

    # Pick best LR by highest mean reward
    best_lr_key = max(summary, key=lambda k: summary[k]["final_mean_reward"])
    best_lr_float = float(best_lr_key)

    print("\n" + "=" * 60)
    print("Learning Rate Sweep Results (final 50 iterations):")
    print(f"{'LR':<12} {'Mean Reward':>14} {'Std Reward':>12} {'Mean NFE':>10}")
    print("-" * 60)
    for lr_key, stats in summary.items():
        marker = " <-- best" if lr_key == best_lr_key else ""
        print(
            f"{lr_key:<12} "
            f"{stats['final_mean_reward']:>14.4f} "
            f"{stats['final_std_reward']:>12.4f} "
            f"{stats['final_mean_nfe']:>10.1f}"
            f"{marker}"
        )
    print("=" * 60)

    # Save results.json
    results_path = os.path.join(args.output_dir, "results.json")
    output = {"best_lr": best_lr_float, "results": summary}
    with open(results_path, "w") as fh:
        json.dump(output, fh, indent=2)
    print(f"\n[tune_lr] Saved results to {results_path}")

    # Plot learning curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    iters = list(range(1, args.n_iters + 1))

    for lr_key in all_rewards:
        axes[0].plot(iters, all_rewards[lr_key], label=f"LR={lr_key}", alpha=0.85)
        axes[1].plot(iters, all_nfes[lr_key], label=f"LR={lr_key}", alpha=0.85)

    axes[0].set_title("Mean Episode Reward")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Mean Reward")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Mean Episode NFE")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Mean NFE")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Learning Rate Sweep — Phase 2 Week 7", fontsize=13)
    plt.tight_layout()

    curves_path = os.path.join(args.output_dir, "curves.png")
    plt.savefig(curves_path, dpi=150)
    plt.close(fig)
    print(f"[tune_lr] Saved learning curves to {curves_path}")
    print(f"[tune_lr] Best LR: {best_lr_key}")


if __name__ == "__main__":
    main()
