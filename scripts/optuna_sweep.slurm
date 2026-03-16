#!/bin/bash
#SBATCH --job-name=optuna_sweep
#SBATCH --partition=hpg-turin
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --time=12:00:00
#SBATCH --output=/blue/ruogu.fang/pateld3/addiffusion/logs/optuna_%j.out
#SBATCH --error=/blue/ruogu.fang/pateld3/addiffusion/logs/optuna_%j.err

# Usage: sbatch scripts/optuna_sweep.slurm [--n_trials 20] [--n_jobs 1]
# For parallel execution use a job array: sbatch --array=0-3 scripts/optuna_sweep.slurm

set -e
cd "/blue/ruogu.fang/pateld3/addiffusion/"
mkdir -p /blue/ruogu.fang/pateld3/addiffusion/logs /blue/ruogu.fang/pateld3/addiffusion/outputs/tuning/optuna /blue/ruogu.fang/pateld3/addiffusion/configs

N_TRIALS=${1:-20}

uv run python /blue/ruogu.fang/pateld3/addiffusion/scripts/optuna_sweep.py \
    --n_trials "$N_TRIALS" \
    --n_iters 200 \
    --prompts_file /blue/ruogu.fang/pateld3/addiffusion/data/coco/prompts_1k.json \
    --output_dir /blue/ruogu.fang/pateld3/addiffusion/outputs/tuning/optuna/ \
    --study_name reward_sweep
