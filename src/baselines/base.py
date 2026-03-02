"""Base class and shared utilities for all Phase 2 baselines.

All baseline implementations inherit from BaseBaseline and return
BaselineResult objects, providing a uniform interface for Week 6
metric computation (compute_metrics.py consumes results.jsonl).

Output layout per baseline:
    outputs/baselines/{method_name}/{seed}_{idx:04d}.png
    outputs/baselines/{method_name}/results.jsonl   ← one JSON line per image
    outputs/baselines/{method_name}/metadata.json   ← method config + timestamp
"""

from __future__ import annotations

import datetime
import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import torch
from PIL import Image


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class BaselineResult:
    """Uniform result container returned by every baseline's generate()."""

    image: Image.Image
    prompt: str
    seed: int
    nfe: int          # Scheduler steps consumed (D-15: counts steps, not network evals)
    time_s: float     # Wall-clock seconds for this image (includes agent overhead)
    method_name: str  # e.g. "ddim20"
    metadata: dict = field(default_factory=dict)  # Method-specific extras


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class BaseBaseline(ABC):
    """Abstract base for all baseline image generators.

    Subclasses must set class attributes `name` and `nfe`, and implement
    `generate()`.  For baselines with dynamic NFE (e.g. oracle), set `nfe = -1`
    and populate `BaselineResult.nfe` per image.
    """

    name: str  # Short identifier, e.g. "ddim20"
    nfe: int   # Fixed NFE (-1 if dynamic)

    @abstractmethod
    def generate(
        self,
        prompts: list[str],
        seeds: list[int],
        output_dir: str,
        **kwargs,
    ) -> list[BaselineResult]:
        """Generate one image per (prompt, seed) pair.

        Args:
            prompts: Text prompts.  len(prompts) == len(seeds).
            seeds:   Per-image random seeds.
            output_dir: Directory where PNGs and results.jsonl are written.
            **kwargs: Method-specific overrides (e.g. sag_scale).

        Returns:
            List of BaselineResult, length == len(prompts).
        """
        ...


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def load_sd15_pipeline(
    model_id: str = "stable-diffusion-v1-5/stable-diffusion-v1-5",
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
):
    """Load SD 1.5 as a StableDiffusionPipeline with safety checker disabled.

    Uses the canonical HF model ID consistent with configs/default.yaml.
    """
    from diffusers import StableDiffusionPipeline

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)
    return pipe


def save_results(results: list[BaselineResult], output_dir: str) -> list[BaselineResult]:
    """Persist images and metadata for a batch of BaselineResults.

    Appends to results.jsonl (so repeated calls accumulate safely).
    Writes metadata.json once on first call.

    Args:
        results:    Results from a single generate() call.
        output_dir: Target directory (created if absent).

    Returns:
        The same results list (for chaining).
    """
    os.makedirs(output_dir, exist_ok=True)
    jsonl_path = os.path.join(output_dir, "results.jsonl")

    with open(jsonl_path, "a", encoding="utf-8") as f:
        for idx, r in enumerate(results):
            img_name = f"{r.seed}_{idx:04d}.png"
            img_path = os.path.join(output_dir, img_name)
            r.image.save(img_path)

            record = {
                "filename": img_name,
                "prompt": r.prompt,
                "seed": r.seed,
                "nfe": r.nfe,
                "time_s": round(r.time_s, 4),
                "method_name": r.method_name,
                "metadata": r.metadata,
            }
            f.write(json.dumps(record) + "\n")

    # Write method-level metadata once
    meta_path = os.path.join(output_dir, "metadata.json")
    if not os.path.exists(meta_path) and results:
        meta = {
            "method_name": results[0].method_name,
            "n_images": len(results),
            "timestamp": datetime.datetime.utcnow().isoformat(),
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    return results
