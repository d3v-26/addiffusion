#!/bin/bash
#SBATCH --job-name=eval_metrics
#SBATCH --partition=hpg-turin
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --time=02:00:00
#SBATCH --output=/blue/ruogu.fang/pateld3/addiffusion/logs/eval_metrics_%j.out
#SBATCH --error=/blue/ruogu.fang/pateld3/addiffusion/logs/eval_metrics_%j.err

# Compute all evaluation metrics for a single baseline method (Phase 2, Week 6).
#
# Usage:
#   sbatch scripts/eval_metrics.slurm ddim20
#   sbatch scripts/eval_metrics.slurm agent
#
# The first positional argument is the method name.  The script expects:
#   - outputs/baselines/{METHOD}/*.png  — generated images
#   - data/drawbench/prompts.json       — prompts used during generation
#
# Results are written to outputs/evaluation/metrics/{METHOD}.json.
#
# To compute FID against a reference directory, set REFERENCE_DIR:
#   sbatch --export=ALL,REFERENCE_DIR=data/coco_reference scripts/eval_metrics.slurm ddim20
#
# References:
#   - plan.md Phase 2 Week 6 (evaluation pipeline)
#   - src/evaluation/compute_metrics.py (metric implementations)
#   - experiment.md §4 (output layout)

set -e
cd "/blue/ruogu.fang/pateld3/addiffusion/"

mkdir -p /blue/ruogu.fang/pateld3/addiffusion/logs /blue/ruogu.fang/pateld3/addiffusion/outputs/evaluation/metrics

METHOD=${1:-"ddim20"}

echo "=== AdDiffusion Metric Evaluation ==="
echo "Method    : ${METHOD}"
echo "Job ID    : ${SLURM_JOB_ID}"
echo "Node      : ${SLURM_NODELIST}"
echo "Started   : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "======================================"

# Build optional --reference_dir argument
REFERENCE_ARG=""
if [ -n "${REFERENCE_DIR}" ]; then
    echo "Reference : ${REFERENCE_DIR}"
    REFERENCE_ARG="--reference_dir ${REFERENCE_DIR}"
fi

uv run python /blue/ruogu.fang/pateld3/addiffusion/src/evaluation/compute_metrics.py \
    --images_dir "/blue/ruogu.fang/pateld3/addiffusion/outputs/baselines/${METHOD}/" \
    --prompts_file /blue/ruogu.fang/pateld3/addiffusion/data/drawbench/prompts.json \
    --output_file "/blue/ruogu.fang/pateld3/addiffusion/outputs/evaluation/metrics/${METHOD}.json" \
    --metrics fid clip_score cmmd imagereward hpsv2 is \
    ${REFERENCE_ARG}

echo "=== Done: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "Results written to /blue/ruogu.fang/pateld3/addiffusion/outputs/evaluation/metrics/${METHOD}.json"
