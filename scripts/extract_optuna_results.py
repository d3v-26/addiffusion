"""Extract and save Optuna study results from an existing journal log.

Run this on the HPC login node (no GPU required) after a job timeout to
recover configs/optuna_top3.yaml and outputs/tuning/optuna/study_summary.json
from the journal written by optuna_sweep.py.

Usage:
    uv run python scripts/extract_optuna_results.py \\
        --output_dir /blue/ruogu.fang/pateld3/addiffusion/outputs/tuning/optuna/ \\
        --study_name reward_sweep
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

from scripts.optuna_sweep import save_top3_yaml, save_study_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract Optuna results from journal.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/tuning/optuna/",
        help="Directory containing journal.log (same as used in optuna_sweep.py).",
    )
    parser.add_argument(
        "--study_name",
        type=str,
        default="reward_sweep",
        help="Optuna study name.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    journal_path = os.path.join(args.output_dir, "journal.log")
    if not os.path.exists(journal_path):
        print(f"[extract] ERROR: journal not found at {journal_path}")
        sys.exit(1)

    storage = JournalStorage(JournalFileBackend(journal_path))
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.load_study(study_name=args.study_name, storage=storage)

    all_trials = study.trials
    complete = [t for t in all_trials if t.state == optuna.trial.TrialState.COMPLETE]
    failed = [t for t in all_trials if t.state == optuna.trial.TrialState.FAIL]
    running = [t for t in all_trials if t.state == optuna.trial.TrialState.RUNNING]

    print(f"[extract] Study '{args.study_name}' — {len(all_trials)} total trials")
    print(f"  COMPLETE : {len(complete)}")
    print(f"  FAILED   : {len(failed)}")
    print(f"  RUNNING  : {len(running)} (incomplete at timeout)")

    if not complete:
        print("[extract] No complete trials found — nothing to save.")
        sys.exit(0)

    best = max(complete, key=lambda t: t.value if t.value is not None else float("-inf"))
    print(f"\n[extract] Best completed trial: #{best.number}")
    print(f"  Objective : {best.value:.4f}")
    print(f"  Params    :")
    for k, v in best.params.items():
        print(f"    {k}: {v}")

    configs_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs"
    )
    os.makedirs(configs_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    top3_path = os.path.join(configs_dir, "optuna_top3.yaml")
    save_top3_yaml(study, top3_path)
    print(f"\n[extract] Saved top-3 configs to {top3_path}")

    summary_path = os.path.join(args.output_dir, "study_summary.json")
    save_study_summary(study, summary_path)
    print(f"[extract] Saved study summary to {summary_path}")

    remaining = max(0, 20 - len(complete))
    if remaining > 0:
        print(
            f"\n[extract] {remaining} more COMPLETE trial(s) needed to reach 20. "
            f"Resume with:\n"
            f"  sbatch scripts/optuna_sweep.slurm {remaining}"
        )
    else:
        print(f"\n[extract] Sweep complete — {len(complete)} complete trials available.")


if __name__ == "__main__":
    main()
