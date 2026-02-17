#!/bin/bash
#SBATCH --job-name=addiffusion-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# Load modules
module load cuda/12.4.1
module load gcc/11.4.0

# Activate environment
cd $SLURM_SUBMIT_DIR
source .venv/bin/activate

# Set HF cache
export HF_HOME=/scratch/$USER/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export TORCH_HOME=/scratch/$USER/torch_cache

# Create log directory
mkdir -p logs

# Run training
echo "Starting training at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Memory: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader)"

uv run python src/train.py \
    --config configs/default.yaml \
    training.total_iterations=2000 \
    training.seed=42 \
    logging.use_wandb=true

echo "Training finished at $(date)"
