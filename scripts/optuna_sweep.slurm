#!/bin/bash
#SBATCH --job-name=optuna_sweep
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --output=logs/optuna_%j.out
#SBATCH --error=logs/optuna_%j.err

# Usage: sbatch scripts/optuna_sweep.slurm [--n_trials 20] [--n_jobs 1]
# For parallel execution use a job array: sbatch --array=0-3 scripts/optuna_sweep.slurm

set -e
cd "$(dirname "$0")/.."
mkdir -p logs outputs/tuning/optuna configs

N_TRIALS=${1:-20}

uv run python scripts/optuna_sweep.py \
    --n_trials "$N_TRIALS" \
    --n_iters 200 \
    --prompts_file data/coco/prompts_1k.json \
    --output_dir outputs/tuning/optuna/ \
    --study_name reward_sweep
