#!/bin/bash
#SBATCH --job-name=baselines
#SBATCH --partition=hpg-turin
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --time=04:00:00
#SBATCH --output=/blue/ruogu.fang/pateld3/addiffusion/logs/baselines_%j.out
#SBATCH --error=/blue/ruogu.fang/pateld3/addiffusion/logs/baselines_%j.err

# Run one or more baselines on DrawBench prompts (Phase 2, Week 6).
#
# Usage:
#   sbatch scripts/run_baselines.slurm ddim20
#   sbatch scripts/run_baselines.slurm ddim20,ddim50,dpm20
#
# The first positional argument is a comma-separated list of baseline names.
# Supported names: ddim20, ddim50, dpm20, dpm50, euler20, unipc20, pndm20,
#                  sag, attend_excite, lcm, sdxl_turbo, random_search, oracle
#
# References:
#   - plan.md Phase 2 Week 6 (baseline generation)
#   - experiment.md §3 (commands and validation)

set -e
cd "/blue/ruogu.fang/pateld3/addiffusion/"

mkdir -p /blue/ruogu.fang/pateld3/addiffusion/logs /blue/ruogu.fang/pateld3/addiffusion/outputs/baselines

BASELINES=${1:-"ddim20"}  # Comma-separated list; default to ddim20 if not provided

echo "=== AdDiffusion Baseline Run ==="
echo "Baselines : ${BASELINES}"
echo "Job ID    : ${SLURM_JOB_ID}"
echo "Node      : ${SLURM_NODELIST}"
echo "Started   : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "================================"

uv run python /blue/ruogu.fang/pateld3/addiffusion/scripts/run_baselines.py \
    --baselines "${BASELINES}" \
    --prompts_file /blue/ruogu.fang/pateld3/addiffusion/data/drawbench/prompts.json \
    --output_dir /blue/ruogu.fang/pateld3/addiffusion/outputs/baselines/ \
    --seeds 42

echo "=== Done: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
