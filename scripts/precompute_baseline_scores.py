"""Precompute DDIM-20 and DDIM-50 baseline scores for training prompts.

Runs both baselines on all (or a capped subset of) training prompts and writes
data/baseline_scores.json, which is used by RewardComputer for per-prompt
terminal reward normalization.

Usage:
    uv run python scripts/precompute_baseline_scores.py \\
        --config /blue/ruogu.fang/pateld3/addiffusion/configs/default.yaml \\
        --prompts_file /blue/ruogu.fang/pateld3/addiffusion/data/coco/annotations/captions_val2014.json \\
        --output /blue/ruogu.fang/pateld3/addiffusion/data/baseline_scores.json \\
        --seed 42

References:
    - docs/superpowers/specs/2026-04-08-reward-redesign-design.md
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import yaml

from src.diffusion.pipeline import AdaptiveDiffusionPipeline
from src.rewards.reward import RewardComputer, RewardConfig


def load_prompts(path: str, max_prompts: int = None) -> list[str]:
    """Load prompts sorted by length (matches curriculum simple_until heuristic)."""
    with open(path) as f:
        data = json.load(f)

    if isinstance(data, list):
        prompts = [p if isinstance(p, str) else p.get("caption", p.get("prompt", "")) for p in data]
    elif "annotations" in data:
        prompts = [ann["caption"] for ann in data["annotations"]]
    else:
        raise ValueError(f"Unknown prompt format in {path}")

    # Deduplicate preserving insertion order
    seen = set()
    unique = []
    for p in prompts:
        if p and p not in seen:
            seen.add(p)
            unique.append(p)

    # Sort by length (matches curriculum: shorter = simpler = trained earlier)
    unique.sort(key=len)

    if max_prompts:
        unique = unique[:max_prompts]

    return unique


def run_ddim(
    pipeline: AdaptiveDiffusionPipeline,
    prompt: str,
    num_steps: int,
    seed: int,
    guidance_scale: float,
) -> torch.Tensor:
    """Run DDIM with `num_steps` steps and return the decoded image (1, 3, H, W)."""
    state = pipeline.prepare(prompt, num_steps=num_steps, seed=seed, guidance_scale=guidance_scale)
    while not state.is_done:
        step_out = pipeline.denoise_step(state, guidance_scale=guidance_scale)
        pipeline.advance_state(state, step_out)
    return pipeline.decode(state.history[-1].z0_pred)


def main():
    parser = argparse.ArgumentParser(description="Precompute DDIM-20 and DDIM-50 baseline scores.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--prompts_file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_prompts", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[precompute] Device: {device}")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    dtype = torch.float16 if cfg["model"]["dtype"] == "float16" else torch.float32
    guidance_scale = cfg["model"]["guidance_scale"]

    print("[precompute] Loading pipeline...")
    pipeline = AdaptiveDiffusionPipeline.from_pretrained(
        cfg["model"]["model_id"],
        scheduler_name="ddim",
        device=device,
        dtype=dtype,
    )

    print("[precompute] Loading reward models...")
    rc = RewardComputer(config=RewardConfig(), device=device)

    prompts = load_prompts(args.prompts_file, max_prompts=args.max_prompts)
    print(f"[precompute] {len(prompts)} prompts to process")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume from existing output if present
    if output_path.exists():
        with open(output_path) as f:
            scores = json.load(f)
        print(f"[precompute] Resuming — {len(scores)} prompts already done")
    else:
        scores = {}

    t_start = time.time()
    for i, prompt in enumerate(prompts):
        if prompt in scores:
            continue

        try:
            img20 = run_ddim(pipeline, prompt, num_steps=20, seed=args.seed, guidance_scale=guidance_scale)
            img50 = run_ddim(pipeline, prompt, num_steps=50, seed=args.seed, guidance_scale=guidance_scale)

            scores[prompt] = {
                "ddim20": {
                    "clip": rc.clip_score(img20, prompt),
                    "aesthetic": rc.aesthetic_score(img20),
                    "image_reward": rc.image_reward_score(img20, prompt),
                },
                "ddim50": {
                    "clip": rc.clip_score(img50, prompt),
                    "aesthetic": rc.aesthetic_score(img50),
                    "image_reward": rc.image_reward_score(img50, prompt),
                },
            }
        except Exception as e:
            print(f"[precompute] WARNING: skipped prompt {i} ({repr(prompt[:40])}) due to {e}")
            continue

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            remaining = (len(prompts) - i - 1) / max(rate, 1e-6)
            print(f"  [{i+1}/{len(prompts)}] {rate:.1f} prompts/s | ETA {remaining/3600:.1f}h")
            # Save checkpoint every 100 prompts
            with open(output_path, "w") as f:
                json.dump(scores, f)

    # Final save
    with open(output_path, "w") as f:
        json.dump(scores, f, indent=2)

    total_time = time.time() - t_start
    print(f"\n[precompute] Done. {len(scores)} prompts in {total_time/3600:.2f}h")
    print(f"[precompute] Output: {output_path}")


if __name__ == "__main__":
    main()
