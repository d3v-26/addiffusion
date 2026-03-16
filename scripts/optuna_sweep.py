"""Optuna reward hyperparameter sweep (Phase 2, Week 7).

Runs a 20-trial Optuna study over reward weights, using JournalStorage
(NFS-safe — no SQLite) so the sweep can be resumed or parallelised across
SLURM jobs on shared filesystems.

Objective: mean_reward (final 50 iters) - 0.5 * std_reward (final 50 iters).

Usage:
    uv run python scripts/optuna_sweep.py \
        --n_trials 20 \
        --n_iters 200 \
        --prompts_file data/coco/prompts_1k.json \
        --output_dir outputs/tuning/optuna/ \
        --study_name reward_sweep
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

import torch

import optuna
from optuna.storages import JournalStorage, JournalFileBackend

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optuna reward hyperparameter sweep.")
    parser.add_argument("--n_trials", type=int, default=20, help="Number of Optuna trials.")
    parser.add_argument(
        "--n_iters",
        type=int,
        default=200,
        help="PPO training iterations per trial (shorter than full sweep for speed).",
    )
    parser.add_argument(
        "--prompts_file",
        type=str,
        default="data/coco/prompts_1k.json",
        help="Path to JSON file with list of prompt strings.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/tuning/optuna/",
        help="Directory for journal, summary JSON, and plots.",
    )
    parser.add_argument(
        "--study_name",
        type=str,
        default="reward_sweep",
        help="Optuna study name (used for resuming).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Global random seed.")
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


def final_50_stats(values: list[float]) -> tuple[float, float]:
    """Return mean and std of the last 50 non-nan values."""
    tail = [v for v in values[-50:] if v == v]
    if not tail:
        return float("nan"), float("nan")
    n = len(tail)
    mean = sum(tail) / n
    variance = sum((v - mean) ** 2 for v in tail) / n
    return mean, variance ** 0.5


def run_trial(
    reward_config: RewardConfig,
    prompts: list[str],
    n_iters: int,
    device: str,
    seed: int,
    pipeline: AdaptiveDiffusionPipeline,
) -> tuple[list[float], list[float]]:
    """Run PPO training for one Optuna trial and return per-iteration stats.

    The pipeline is passed in and reused across trials to avoid repeated
    model loading.

    Returns:
        rewards_per_iter, nfe_per_iter
    """
    torch.manual_seed(seed)
    random.seed(seed)

    state_extractor = StateExtractor(device=device)
    attention_extractor = UNetAttentionExtractor(latent_h=64, latent_w=64)
    attention_extractor.hook(pipeline.unet)

    policy = PolicyNetwork(input_dim=StateExtractor.TOTAL_DIM, hidden_dim=512).to(device)
    value_net = ValueNetwork(input_dim=StateExtractor.TOTAL_DIM, hidden_dim=512).to(device)

    ppo_config = PPOConfig(lr=DEFAULT_LR)
    trainer = PPOTrainer(policy=policy, value_net=value_net, config=ppo_config, device=device)

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
                f"    iter {iteration + 1}/{n_iters}  "
                f"mean_reward={mean_reward:.4f}  mean_nfe={mean_nfe:.1f}"
            )

    attention_extractor.remove_hooks()
    return rewards_per_iter, nfe_per_iter


def build_objective(
    prompts: list[str],
    n_iters: int,
    device: str,
    seed: int,
    pipeline: AdaptiveDiffusionPipeline,
):
    """Return a closure that Optuna calls for each trial."""

    def objective(trial: optuna.Trial) -> float:
        # Search space (exact values from spec)
        alpha_1 = trial.suggest_float("alpha_1", 0.5, 3.0)
        alpha_2 = trial.suggest_float("alpha_2", 0.1, 1.5)
        alpha_3 = trial.suggest_float("alpha_3", 0.05, 0.5)
        alpha_4 = trial.suggest_float("alpha_4", 0.3, 2.0)
        c_nfe   = trial.suggest_float("c_nfe", 0.005, 0.05, log=True)
        beta_1  = trial.suggest_float("beta_1", 1.0, 4.0)
        beta_2  = trial.suggest_float("beta_2", 0.5, 2.0)
        beta_3  = trial.suggest_float("beta_3", 0.5, 3.0)

        reward_config = RewardConfig(
            alpha_1=alpha_1,
            alpha_2=alpha_2,
            alpha_3=alpha_3,
            alpha_4=alpha_4,
            c_nfe=c_nfe,
            beta_1=beta_1,
            beta_2=beta_2,
            beta_3=beta_3,
            normalize=True,
        )

        trial_seed = seed + trial.number * 1000
        print(
            f"\n[optuna] Trial {trial.number}  "
            f"alpha_1={alpha_1:.3f} alpha_2={alpha_2:.3f} "
            f"alpha_3={alpha_3:.3f} alpha_4={alpha_4:.3f} "
            f"c_nfe={c_nfe:.4f}"
        )

        rewards, _ = run_trial(
            reward_config=reward_config,
            prompts=prompts,
            n_iters=n_iters,
            device=device,
            seed=trial_seed,
            pipeline=pipeline,
        )

        mean_r, std_r = final_50_stats(rewards)
        if mean_r != mean_r:  # nan check
            return float("-inf")

        obj = mean_r - 0.5 * std_r
        print(
            f"[optuna] Trial {trial.number}  "
            f"mean_reward={mean_r:.4f}  std_reward={std_r:.4f}  objective={obj:.4f}"
        )
        return obj

    return objective


def save_top3_yaml(study: optuna.Study, output_path: str) -> None:
    """Write top-3 trial configs to a YAML file."""
    completed = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
    ]
    completed.sort(key=lambda t: t.value, reverse=True)
    top3 = completed[:3]

    lines = [
        "# Top-3 Optuna trials from Phase 2 Week 7 reward sweep",
        "# Generated by scripts/optuna_sweep.py",
        "",
        "top_trials:",
    ]

    for rank, trial in enumerate(top3, start=1):
        lines.append(f"  - rank: {rank}")
        lines.append(f"    trial_number: {trial.number}")
        obj_str = f"{trial.value:.4f}" if trial.value is not None else "null"
        lines.append(f"    objective: {obj_str}")
        lines.append("    params:")
        for param_name, param_val in trial.params.items():
            lines.append(f"      {param_name}: {param_val}")

    with open(output_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def save_study_summary(study: optuna.Study, output_path: str) -> None:
    """Write full study summary as JSON."""
    trials_data = []
    for t in study.trials:
        trials_data.append(
            {
                "number": t.number,
                "state": t.state.name,
                "value": t.value,
                "params": t.params,
            }
        )

    best = study.best_trial
    summary = {
        "study_name": study.study_name,
        "direction": "maximize",
        "n_trials": len(study.trials),
        "best_trial": {
            "number": best.number,
            "objective": best.value,
            "params": best.params,
        },
        "trials": trials_data,
    }
    with open(output_path, "w") as fh:
        json.dump(summary, fh, indent=2)


def save_optuna_plots(study: optuna.Study, output_dir: str) -> None:
    """Save Optuna visualisation plots as PNG files (soft dependency)."""
    try:
        import optuna.visualization as vis
        import matplotlib
        matplotlib.use("Agg")

        history_fig = vis.plot_optimization_history(study)
        history_fig.write_image(os.path.join(output_dir, "optimization_history.png"))

        importance_fig = vis.plot_param_importances(study)
        importance_fig.write_image(os.path.join(output_dir, "param_importances.png"))

        print(
            f"[optuna] Saved Optuna visualizations to {output_dir}"
        )
    except Exception as e:
        print(f"[optuna] Could not save Optuna plots (requires plotly + kaleido): {e}")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    configs_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs"
    )
    os.makedirs(configs_dir, exist_ok=True)

    print(f"[optuna] Loading prompts from {args.prompts_file} ...")
    prompts = load_prompts(args.prompts_file)
    print(f"[optuna] Loaded {len(prompts)} prompts.")

    # Load pipeline once and reuse across trials
    print(f"[optuna] Loading diffusion pipeline ...")
    pipeline = AdaptiveDiffusionPipeline.from_pretrained(MODEL_ID, device=args.device)

    # JournalStorage: NFS-safe, no SQLite required (critical for HPC clusters)
    journal_path = os.path.join(args.output_dir, "journal.log")
    storage = JournalStorage(JournalFileBackend(journal_path))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        storage=storage,
        direction="maximize",
        study_name=args.study_name,
        load_if_exists=True,  # Resume interrupted runs
    )

    objective = build_objective(
        prompts=prompts,
        n_iters=args.n_iters,
        device=args.device,
        seed=args.seed,
        pipeline=pipeline,
    )

    print(f"[optuna] Starting study '{args.study_name}' — {args.n_trials} trials.")
    study.optimize(objective, n_trials=args.n_trials)

    print(f"\n[optuna] Best trial: #{study.best_trial.number}")
    print(f"  Objective: {study.best_trial.value:.4f}")
    print(f"  Params: {study.best_trial.params}")

    # Save top-3 YAML
    top3_path = os.path.join(configs_dir, "optuna_top3.yaml")
    save_top3_yaml(study, top3_path)
    print(f"[optuna] Saved top-3 configs to {top3_path}")

    # Save full study summary JSON
    summary_path = os.path.join(args.output_dir, "study_summary.json")
    save_study_summary(study, summary_path)
    print(f"[optuna] Saved study summary to {summary_path}")

    # Save visualisation plots (requires plotly + kaleido)
    save_optuna_plots(study, args.output_dir)


if __name__ == "__main__":
    main()
