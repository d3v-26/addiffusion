#!/bin/bash
#SBATCH --job-name=addiffusion-pilot-reward-v2
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/blue/ruogu.fang/pateld3/addiffusion/logs/pilot_reward_v2_%j.out
#SBATCH --error=/blue/ruogu.fang/pateld3/addiffusion/logs/pilot_reward_v2_%j.err

# Load modules
module load cuda/12.8.0
module load gcc/11.4.0

# Activate environment
cd "/blue/ruogu.fang/pateld3/addiffusion/"

# Unbuffered Python output
export PYTHONUNBUFFERED=1

# Set HF cache
export HF_HOME=/blue/ruogu.fang/pateld3/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export TORCH_HOME=/blue/ruogu.fang/pateld3/.cache/torch

# Create log directory
mkdir -p /blue/ruogu.fang/pateld3/addiffusion/logs

# Info
echo "Starting pilot reward v2 at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Memory: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader)"

uv run python -u /blue/ruogu.fang/pateld3/addiffusion/src/train.py \
    --config /blue/ruogu.fang/pateld3/addiffusion/configs/default.yaml \
    training.total_iterations=200 \
    training.checkpoint_every=50 \
    training.log_every=5 \
    training.curriculum.simple_until=200 \
    training.curriculum.mixed_until=200 \
    logging.use_wandb=true \
    logging.project=addiffusion-pilots

echo "Pilot reward v2 finished at $(date)"
