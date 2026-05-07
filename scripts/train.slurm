#!/bin/bash
#SBATCH --job-name=addiffusion-train
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/blue/ruogu.fang/pateld3/addiffusion/logs/train_%j.out
#SBATCH --error=/blue/ruogu.fang/pateld3/addiffusion/logs/train_%j.err

# Load modules
module load cuda/12.8.1
module load gcc/11.4.0

# Activate environment
cd "/blue/ruogu.fang/pateld3/addiffusion/"

# Unbuffered Python output (flush print() immediately, not just at buffer fill)
export PYTHONUNBUFFERED=1

# Set HF cache
export HF_HOME=/blue/ruogu.fang/pateld3/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export TORCH_HOME=/blue/ruogu.fang/pateld3/.cache/torch

# Create log directory
mkdir -p /blue/ruogu.fang/pateld3/addiffusion/logs

# Run training
echo "Starting training at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Memory: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader)"

uv run python -u /blue/ruogu.fang/pateld3/addiffusion/src/train.py \
    --config /blue/ruogu.fang/pateld3/addiffusion/configs/default.yaml \
    training.total_iterations=2000 \
    training.seed=42 \
    ppo.batch_size=8

echo "Training finished at $(date)"
