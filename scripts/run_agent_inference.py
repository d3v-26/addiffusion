"""Run trained AdDiffusion agent on a prompt set for evaluation.

Generates images using the trained agent checkpoint and writes outputs to
outputs/baselines/agent/ in the same format as run_baselines.py:
    - PNG files: {seed}_{idx:04d}.png
    - results.jsonl: one JSON record per image (nfe, action_sequence, time_s, ...)
    - metadata.json: checkpoint path, config, timestamp

Usage:
    uv run python scripts/run_agent_inference.py \\
        --checkpoint checkpoints/agent_v1_final.pt \\
        --config configs/default.yaml \\
        --prompts_file data/drawbench/prompts.json \\
        --output_dir outputs/baselines/agent/ \\
        --seed 42 \\
        --deterministic

References:
    - plan.md Phase 3 Week 8 (G2 evaluation)
    - phase2.md Decision Gate G2 criteria
    - src/agent/episode.py (EpisodeRunner)
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
from torchvision.transforms.functional import to_pil_image

from src.agent.episode import EpisodeRunner
from src.agent.networks import PolicyNetwork, ValueNetwork
from src.agent.state import StateExtractor
from src.diffusion.attention import UNetAttentionExtractor
from src.diffusion.pipeline import AdaptiveDiffusionPipeline


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_prompts(path: str) -> list[str]:
    """Load prompts from JSON array, JSONL, or COCO captions file."""
    with open(path, "r", encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)

        if first_char == "[":
            data = json.load(f)
            return [str(p) for p in data]

        if first_char == "{":
            try:
                data = json.load(f)
                if "annotations" in data:
                    return [ann["caption"] for ann in data["annotations"]]
                for key in ("Prompts", "prompt", "caption", "text"):
                    if key in data:
                        return [str(data[key])]
            except json.JSONDecodeError:
                f.seek(0)

        # JSONL
        f.seek(0)
        prompts = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            for key in ("Prompts", "prompt", "caption", "text"):
                if key in record:
                    prompts.append(str(record[key]))
                    break
        return prompts


def main():
    parser = argparse.ArgumentParser(description="Run trained agent for G2 evaluation.")
    parser.add_argument("--checkpoint", required=True, help="Path to agent checkpoint (.pt)")
    parser.add_argument("--config", default="configs/default.yaml", help="Config YAML")
    parser.add_argument("--prompts_file", required=True, help="JSON prompts file")
    parser.add_argument("--output_dir", required=True, help="Output directory for images + JSONL")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use argmax actions (greedy) instead of sampling",
    )
    parser.add_argument("--device", default=None, help="Torch device (default: cuda if available)")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[run_agent_inference] Device: {device}")

    cfg = load_config(args.config)
    dtype = torch.float16 if cfg["model"]["dtype"] == "float16" else torch.float32

    # Load pipeline
    print("[run_agent_inference] Loading diffusion pipeline...")
    pipeline = AdaptiveDiffusionPipeline.from_pretrained(
        cfg["model"]["model_id"],
        scheduler_name=cfg["model"]["scheduler"],
        device=device,
        dtype=dtype,
    )

    attn_ext = UNetAttentionExtractor(latent_h=64, latent_w=64)
    attn_ext.hook_with_processor(pipeline.unet)

    state_ext = StateExtractor(device=device, dtype=dtype)

    # Load checkpoint
    print(f"[run_agent_inference] Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)

    policy = PolicyNetwork(
        input_dim=cfg["agent"]["state_dim"],
        hidden_dim=cfg["agent"]["hidden_dim"],
        num_actions=cfg["agent"]["num_actions"],
    ).to(device).float()
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.eval()

    value_net = ValueNetwork(
        input_dim=cfg["agent"]["state_dim"],
        hidden_dim=cfg["agent"]["hidden_dim"],
    ).to(device).float()
    value_net.load_state_dict(ckpt["value_state_dict"])
    value_net.eval()

    print(f"[run_agent_inference] Loaded checkpoint from iteration {ckpt.get('iteration', 'unknown')}")

    runner = EpisodeRunner(
        pipeline=pipeline,
        state_extractor=state_ext,
        attention_extractor=attn_ext,
        warmup_steps=cfg["agent"]["warmup_steps"],
        guidance_scale=cfg["model"]["guidance_scale"],
        refine_k=cfg["refinement"]["k"],
        refine_r_noise=cfg["refinement"]["r_noise"],
        mask_threshold=cfg["refinement"]["mask_threshold"],
        blur_sigma=cfg["refinement"]["blur_sigma"],
    )

    prompts = load_prompts(args.prompts_file)
    print(f"[run_agent_inference] {len(prompts)} prompts loaded")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "method": "agent",
        "checkpoint": str(args.checkpoint),
        "trained_iteration": ckpt.get("iteration", "unknown"),
        "config": args.config,
        "prompts_file": args.prompts_file,
        "seed": args.seed,
        "deterministic": args.deterministic,
        "num_prompts": len(prompts),
        "num_steps": cfg["training"]["num_steps"],
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    results_path = output_dir / "results.jsonl"
    total_nfe = 0
    t_total = time.time()

    with open(results_path, "w") as results_f:
        for idx, prompt in enumerate(prompts):
            seed = args.seed + idx
            t0 = time.time()

            ep = runner.run_episode(
                prompt=prompt,
                policy=policy,
                value_net=value_net,
                num_steps=cfg["training"]["num_steps"],
                seed=seed,
                deterministic=args.deterministic,
                reward_fn=None,
            )

            elapsed = time.time() - t0

            img_filename = f"{seed}_{idx:04d}.png"
            if ep.final_image is not None:
                pil_img = to_pil_image(ep.final_image[0].cpu().float().clamp(0, 1))
                pil_img.save(output_dir / img_filename)
            else:
                print(f"[WARNING] No final image at idx={idx}")

            record = {
                "idx": idx,
                "prompt": prompt,
                "seed": seed,
                "nfe": ep.total_nfe,
                "action_sequence": ep.action_sequence,
                "total_steps": ep.total_steps,
                "time_s": elapsed,
                "image_file": img_filename,
                "method_name": "agent",
            }
            results_f.write(json.dumps(record) + "\n")
            total_nfe += ep.total_nfe

            if (idx + 1) % 10 == 0 or (idx + 1) == len(prompts):
                print(
                    f"  [{idx+1:4d}/{len(prompts)}] nfe={ep.total_nfe} "
                    f"actions={ep.action_sequence} time={elapsed:.1f}s | "
                    f"avg_nfe={total_nfe/(idx+1):.1f}"
                )

    avg_nfe = total_nfe / len(prompts)
    print(f"\n[run_agent_inference] Done in {time.time()-t_total:.1f}s")
    print(f"  Mean NFE : {avg_nfe:.2f}  (DDIM-50 budget = {cfg['training']['num_steps']})")
    print(f"  Results  : {results_path}")


if __name__ == "__main__":
    main()
