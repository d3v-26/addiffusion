#!/bin/bash
#SBATCH --job-name=agent_inference
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/blue/ruogu.fang/pateld3/addiffusion/logs/agent_inference_%j.out
#SBATCH --error=/blue/ruogu.fang/pateld3/addiffusion/logs/agent_inference_%j.err

# Run trained agent on DrawBench for G2 evaluation.
#
# Usage:
#   sbatch scripts/run_agent_inference.slurm
#
# Outputs go to outputs/baselines/agent/ in the same format as run_baselines.py,
# so eval_metrics.slurm can be run with "agent" as the method name afterward.
#
# References:
#   - plan.md Phase 3 Week 8 (G2 evaluation)
#   - phase2.md Decision Gate G2 criteria

set -e

module load cuda/12.8.1
module load gcc/11.4.0

cd "/blue/ruogu.fang/pateld3/addiffusion/"

export PYTHONUNBUFFERED=1
export HF_HOME=/blue/ruogu.fang/pateld3/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export TORCH_HOME=/blue/ruogu.fang/pateld3/.cache/torch

mkdir -p /blue/ruogu.fang/pateld3/addiffusion/logs
mkdir -p /blue/ruogu.fang/pateld3/addiffusion/outputs/baselines/agent

echo "=== AdDiffusion Agent Inference ==="
echo "Job ID  : ${SLURM_JOB_ID}"
echo "Node    : ${SLURM_NODELIST}"
echo "GPU     : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Started : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "===================================="

uv run python -u /blue/ruogu.fang/pateld3/addiffusion/scripts/run_agent_inference.py \
    --checkpoint /blue/ruogu.fang/pateld3/addiffusion/checkpoints/agent_v1_final.pt \
    --config /blue/ruogu.fang/pateld3/addiffusion/configs/default.yaml \
    --prompts_file /blue/ruogu.fang/pateld3/addiffusion/data/drawbench/prompts.json \
    --output_dir /blue/ruogu.fang/pateld3/addiffusion/outputs/baselines/agent/ \
    --seed 42 \
    --deterministic

echo "=== Done: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
