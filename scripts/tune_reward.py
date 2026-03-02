"""Reward weight profile sweep (Phase 2, Week 7).

Tests three reward configurations to select the best quality/efficiency tradeoff
before the full hyperparameter sweep in optuna_sweep.py.

Usage:
    uv run python scripts/tune_reward.py \
        --n_iters 500 \
        --prompts_file data/coco/prompts_1k.json \
        --output_dir outputs/tuning/reward_sweep/
"""

from __future__ import annotations

import argparse
import json
import os
import random
import traceback

import torch

from src.agent.episode import EpisodeRunner
from src.agent.networks import PolicyNetwork, ValueNetwork
from src.agent.ppo import PPOConfig, PPOTrainer, episodes_to_batch
from src.agent.state import StateExtractor
from src.diffusion.attention import UNetAttentionExtractor
from src.diffusion.pipeline import AdaptiveDiffusionPipeline
from src.rewards.reward import RewardComputer, RewardConfig


MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
DEFAULT_LR = 3e-4
BATCH_SIZE = 4  # prompts per iteration

# Three reward profiles to compare (research.md §3.6 defaults are in "balanced")
REWARD_PROFILES: dict[str, dict] = {
    "quality_heavy": {
        "alpha_1": 2.0,
        "alpha_2": 1.0,
        "alpha_3": 0.2,
        "alpha_4": 1.6,
        "c_nfe": 0.005,
    },
    "balanced": {
        "alpha_1": 1.0,
        "alpha_2": 0.5,
        "alpha_3": 0.2,
        "alpha_4": 0.8,
        "c_nfe": 0.01,
    },
    "efficiency_heavy": {
        "alpha_1": 0.5,
        "alpha_2": 0.2,
        "alpha_3": 0.2,
        "alpha_4": 0.4,
        "c_nfe": 0.02,
    },
}

# Fixed terminal weights and normalization (same across all profiles)
FIXED_REWARD_KWARGS: dict = {
    "beta_1": 2.0,
    "beta_2": 1.0,
    "beta_3": 1.5,
    "normalize": True,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reward weight profile sweep.")
    parser.add_argument("--n_iters", type=int, default=500, help="Training iterations per profile.")
    parser.add_argument(
        "--prompts_file",
        type=str,
        default="data/coco/prompts_1k.json",
        help="Path to JSON file with list of prompt strings.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/tuning/reward_sweep/",
        help="Directory for results and YAML.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device string.")
    return parser.parse_args()


def load_prompts(path: str) -> list[str]:
    with open(path, "r") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}, got {type(data)}.")
    return [str(p) for p in data]


def make_reward_fn(reward_computer: RewardComputer, pipeline: AdaptiveDiffusionPipeline):
    """Return a reward callable compatible with EpisodeRunner.run_episode."""

    def reward_fn(
        prev_z0,
        curr_z0,
        action: int,
        step_index: int,
        n_max: int,
        is_terminal: bool,
        prompt: str,
        decoded_image,
    ) -> float:
        try:
            prev_image = pipeline.decode(prev_z0) if prev_z0 is not None else None
            total, _ = reward_computer.compute_reward(
                prev_image=prev_image,
                curr_image=decoded_image,
                prompt=prompt,
                action=action,
                is_terminal=is_terminal,
            )
            return total
        except Exception:
            return 0.0

    return reward_fn


def run_sweep_for_profile(
    profile_name: str,
    profile_weights: dict,
    prompts: list[str],
    n_iters: int,
    device: str,
    seed: int,
) -> tuple[list[float], list[float]]:
    """Run PPO training for `n_iters` iterations with a given reward profile.

    Returns:
        rewards_per_iter: Mean episode total reward per iteration.
        nfe_per_iter: Mean episode NFE per iteration.
    """
    torch.manual_seed(seed)
    random.seed(seed)

    print(f"\n[tune_reward] Loading pipeline for profile='{profile_name}' ...")
    pipeline = AdaptiveDiffusionPipeline.from_pretrained(MODEL_ID, device=device)

    state_extractor = StateExtractor(device=device)
    attention_extractor = UNetAttentionExtractor(latent_h=64, latent_w=64)
    attention_extractor.hook(pipeline.unet)

    policy = PolicyNetwork(input_dim=StateExtractor.TOTAL_DIM, hidden_dim=512).to(device)
    value_net = ValueNetwork(input_dim=StateExtractor.TOTAL_DIM, hidden_dim=512).to(device)

    ppo_config = PPOConfig(lr=DEFAULT_LR)
    trainer = PPOTrainer(policy=policy, value_net=value_net, config=ppo_config, device=device)

    reward_config = RewardConfig(**{**profile_weights, **FIXED_REWARD_KWARGS})
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
                f"  [{profile_name}] iter {iteration + 1}/{n_iters}  "
                f"mean_reward={mean_reward:.4f}  mean_nfe={mean_nfe:.1f}"
            )

    attention_extractor.remove_hooks()
    return rewards_per_iter, nfe_per_iter


def final_50_stats(values: list[float]) -> tuple[float, float]:
    """Return mean and std of the last 50 non-nan values."""
    tail = [v for v in values[-50:] if v == v]
    if not tail:
        return float("nan"), float("nan")
    n = len(tail)
    mean = sum(tail) / n
    variance = sum((v - mean) ** 2 for v in tail) / n
    return mean, variance ** 0.5


def write_best_yaml(best_profile: str, profile_weights: dict, output_path: str) -> None:
    """Write the best reward profile as a Hydra-compatible YAML file."""
    full_params = {**profile_weights, **FIXED_REWARD_KWARGS}
    lines = [
        "# Best reward profile from Phase 2 Week 7 tuning",
        f"# Profile: {best_profile}",
        "# Selected by: highest mean_reward in final 50 iters",
        "",
        "reward:",
    ]
    key_order = [
        "alpha_1", "alpha_2", "alpha_3", "alpha_4",
        "c_nfe", "beta_1", "beta_2", "beta_3", "normalize",
    ]
    for key in key_order:
        val = full_params.get(key)
        if val is None:
            continue
        if isinstance(val, bool):
            lines.append(f"  {key}: {'true' if val else 'false'}")
        elif isinstance(val, float):
            # Preserve concise float representation
            lines.append(f"  {key}: {val}")
        else:
            lines.append(f"  {key}: {val}")

    with open(output_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[tune_reward] Loading prompts from {args.prompts_file} ...")
    prompts = load_prompts(args.prompts_file)
    print(f"[tune_reward] Loaded {len(prompts)} prompts.")

    summary: dict[str, dict] = {}

    for profile_name, profile_weights in REWARD_PROFILES.items():
        rewards, nfes = run_sweep_for_profile(
            profile_name=profile_name,
            profile_weights=profile_weights,
            prompts=prompts,
            n_iters=args.n_iters,
            device=args.device,
            seed=args.seed,
        )
        mean_r, std_r = final_50_stats(rewards)
        mean_nfe, _ = final_50_stats(nfes)

        summary[profile_name] = {
            "final_mean_reward": round(mean_r, 6),
            "final_std_reward": round(std_r, 6),
            "final_mean_nfe": round(mean_nfe, 2),
            "profile_weights": profile_weights,
        }
        print(
            f"\n[tune_reward] Profile='{profile_name}'  "
            f"final_mean_reward={mean_r:.4f}  "
            f"final_mean_nfe={mean_nfe:.1f}"
        )

    # Select best profile
    best_profile = max(summary, key=lambda k: summary[k]["final_mean_reward"])

    print("\n" + "=" * 70)
    print("Reward Profile Sweep Results (final 50 iterations):")
    print(f"{'Profile':<20} {'Mean Reward':>14} {'Std Reward':>12} {'Mean NFE':>10}")
    print("-" * 70)
    for profile_name, stats in summary.items():
        marker = " <-- best" if profile_name == best_profile else ""
        print(
            f"{profile_name:<20} "
            f"{stats['final_mean_reward']:>14.4f} "
            f"{stats['final_std_reward']:>12.4f} "
            f"{stats['final_mean_nfe']:>10.1f}"
            f"{marker}"
        )
    print("=" * 70)

    # Save results.json
    results_path = os.path.join(args.output_dir, "results.json")
    serialisable_summary = {
        k: {sk: sv for sk, sv in v.items() if sk != "profile_weights"}
        for k, v in summary.items()
    }
    output = {"best_profile": best_profile, "results": serialisable_summary}
    with open(results_path, "w") as fh:
        json.dump(output, fh, indent=2)
    print(f"\n[tune_reward] Saved results to {results_path}")

    # Write best profile YAML to configs/
    configs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")
    os.makedirs(configs_dir, exist_ok=True)
    yaml_path = os.path.join(configs_dir, "tuned_reward.yaml")
    write_best_yaml(
        best_profile=best_profile,
        profile_weights=REWARD_PROFILES[best_profile],
        output_path=yaml_path,
    )
    print(f"[tune_reward] Wrote best profile YAML to {yaml_path}")
    print(f"[tune_reward] Best profile: {best_profile}")


if __name__ == "__main__":
    main()
