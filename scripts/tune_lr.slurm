#!/bin/bash
#SBATCH --job-name=tune_lr
#SBATCH --partition=hpg-turin
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --time=06:00:00
#SBATCH --output=/blue/ruogu.fang/pateld3/addiffusion/logs/tune_lr_%j.out
#SBATCH --error=/blue/ruogu.fang/pateld3/addiffusion/logs/tune_lr_%j.err

set -e
cd "/blue/ruogu.fang/pateld3/addiffusion/"
mkdir -p logs outputs/tuning/lr_sweep

uv run python ./scripts/tune_lr.py \
    --lrs 1e-4 3e-4 1e-3 \
    --n_iters 500 \
    --prompts_file data/coco/prompts_1k.json \
    --output_dir outputs/tuning/lr_sweep/
