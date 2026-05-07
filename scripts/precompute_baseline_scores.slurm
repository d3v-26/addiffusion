#!/bin/bash
#SBATCH --job-name=precompute_baselines
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=/blue/ruogu.fang/pateld3/addiffusion/logs/precompute_baselines_%j.out
#SBATCH --error=/blue/ruogu.fang/pateld3/addiffusion/logs/precompute_baselines_%j.err

# Precompute DDIM-20 and DDIM-50 baseline scores for all training prompts.
# Must be run before training. Output: data/baseline_scores.json
#
# Usage: sbatch scripts/precompute_baseline_scores.slurm

set -e

module load cuda/12.8.1
module load gcc/11.4.0

cd "/blue/ruogu.fang/pateld3/addiffusion/"

export PYTHONUNBUFFERED=1
export HF_HOME=/blue/ruogu.fang/pateld3/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export TORCH_HOME=/blue/ruogu.fang/pateld3/.cache/torch

mkdir -p /blue/ruogu.fang/pateld3/addiffusion/logs
mkdir -p /blue/ruogu.fang/pateld3/addiffusion/data

echo "=== Precompute Baseline Scores ==="
echo "Job ID  : ${SLURM_JOB_ID}"
echo "Node    : ${SLURM_NODELIST}"
echo "GPU     : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Started : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "==================================="

uv run python -u /blue/ruogu.fang/pateld3/addiffusion/scripts/precompute_baseline_scores.py \
    --config /blue/ruogu.fang/pateld3/addiffusion/configs/default.yaml \
    --prompts_file /blue/ruogu.fang/pateld3/addiffusion/data/coco/annotations/captions_val2014.json \
    --output /blue/ruogu.fang/pateld3/addiffusion/data/baseline_scores.json \
    --seed 42

echo "=== Done: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
