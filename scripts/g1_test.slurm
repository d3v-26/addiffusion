#!/bin/bash
#SBATCH --job-name=g1-test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/g1_test_%j.out
#SBATCH --error=logs/g1_test_%j.err

# Load modules
module load cuda/12.4.1
module load gcc/11.4.0

# Activate environment
cd $SLURM_SUBMIT_DIR

# Set caches
export HF_HOME=/blue/ruogu.fang/pateld3/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export TORCH_HOME=/blue/ruogu.fang/pateld3/.cache/torch

echo "========================================"
echo "G1 Decision Gate Test: 100 iterations"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Memory: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader)"
echo ""

uv run python -u tests/test_g1.py 2>&1

echo ""
echo "Finished: $(date)"
