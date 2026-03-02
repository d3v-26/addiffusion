"""Tests for LCMBaseline and SDXLTurboBaseline.

Validates that both distillation-based fast-generation baselines produce a
single PIL image with a positive NFE and wall-clock time, and write a
results.jsonl file to the output directory.

LCM runs SD 1.5 with the LCM-LoRA adapter in 4 steps (CFG disabled).
SDXL-Turbo runs adversarially-distilled SDXL in 4 steps (CFG disabled).

Run:
    uv run python tests/test_baselines_distill.py
Requires: GPU.  LCM needs ~10 GB VRAM; SDXL-Turbo needs ~24 GB VRAM.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.baselines.lcm import LCMBaseline
from src.baselines.sdxl_turbo import SDXLTurboBaseline

# ---------------------------------------------------------------------------
# Shared test parameters
# ---------------------------------------------------------------------------

PROMPT = "a red bicycle"
SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _assert_jsonl_exists(output_dir: str) -> None:
    """Raise AssertionError if results.jsonl is missing or empty."""
    jsonl_path = os.path.join(output_dir, "results.jsonl")
    assert os.path.exists(jsonl_path), f"results.jsonl not found in {output_dir}"
    with open(jsonl_path, encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    assert len(lines) >= 1, "results.jsonl is empty"
    for line in lines:
        json.loads(line)


# ---------------------------------------------------------------------------
# LCM test
# ---------------------------------------------------------------------------

def test_lcm_baseline() -> None:
    """LCMBaseline: 4-step generation, correct NFE, valid image, jsonl written."""
    print("=" * 60)
    print("TEST: LCMBaseline (B5 — LCM-LoRA SD 1.5, 4 steps)")
    print("=" * 60)

    try:
        baseline = LCMBaseline(device=DEVICE)
        output_dir = tempfile.mkdtemp(prefix="test_lcm_")

        results = baseline.generate(
            prompts=[PROMPT],
            seeds=[SEED],
            output_dir=output_dir,
        )

        assert len(results) == 1, f"Expected 1 result, got {len(results)}"

        r = results[0]
        assert r.nfe == 4, f"Expected nfe == 4, got {r.nfe}"
        assert r.image.size == (512, 512), (
            f"Expected image size (512, 512), got {r.image.size}"
        )
        assert r.time_s > 0, f"Expected time_s > 0, got {r.time_s}"
        assert r.method_name == "lcm_4", (
            f"Expected method_name 'lcm_4', got {r.method_name!r}"
        )
        assert r.prompt == PROMPT, f"Prompt mismatch: {r.prompt!r}"
        assert r.seed == SEED, f"Seed mismatch: {r.seed}"

        _assert_jsonl_exists(output_dir)

        print(f"  Output dir: {output_dir}")
        print(f"  Image size: {r.image.size}")
        print(f"  NFE: {r.nfe}  |  time_s: {r.time_s:.2f}s")
        print("[PASS] LCMBaseline")

    except Exception as exc:
        print(f"[FAIL] LCMBaseline: {exc}")
        raise


# ---------------------------------------------------------------------------
# SDXL-Turbo test
# ---------------------------------------------------------------------------

def test_sdxl_turbo_baseline() -> None:
    """SDXLTurboBaseline: 4-step generation, 512x512 image, jsonl written.

    Note: Requires ~24 GB VRAM.  Skip this test on A100-40GB nodes and run
    on an A100-80GB node instead.
    """
    print("\n" + "=" * 60)
    print("TEST: SDXLTurboBaseline (B5 — SDXL-Turbo, 4 steps)")
    print("=" * 60)

    # Soft VRAM guard: warn if < 20 GB but do not skip — let the OOM surface
    # naturally so it is visible in CI logs.
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        if vram_gb < 20:
            print(
                f"  WARNING: Only {vram_gb:.1f} GB VRAM detected. "
                "SDXL-Turbo requires ~24 GB; test may OOM."
            )

    try:
        baseline = SDXLTurboBaseline(device=DEVICE)
        output_dir = tempfile.mkdtemp(prefix="test_sdxl_turbo_")

        results = baseline.generate(
            prompts=[PROMPT],
            seeds=[SEED],
            output_dir=output_dir,
        )

        assert len(results) == 1, f"Expected 1 result, got {len(results)}"

        r = results[0]
        assert r.nfe == 4, f"Expected nfe == 4, got {r.nfe}"
        assert r.image.size == (512, 512), (
            f"Expected image size (512, 512), got {r.image.size}"
        )
        assert r.time_s > 0, f"Expected time_s > 0, got {r.time_s}"
        assert r.method_name == "sdxl_turbo_4", (
            f"Expected method_name 'sdxl_turbo_4', got {r.method_name!r}"
        )
        assert r.prompt == PROMPT, f"Prompt mismatch: {r.prompt!r}"
        assert r.seed == SEED, f"Seed mismatch: {r.seed}"

        _assert_jsonl_exists(output_dir)

        print(f"  Output dir: {output_dir}")
        print(f"  Image size: {r.image.size}")
        print(f"  NFE: {r.nfe}  |  time_s: {r.time_s:.2f}s")
        print("[PASS] SDXLTurboBaseline")

    except Exception as exc:
        print(f"[FAIL] SDXLTurboBaseline: {exc}")
        raise


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_lcm_baseline()
    test_sdxl_turbo_baseline()

    print("\n" + "=" * 60)
    print("ALL DISTILLATION BASELINE TESTS PASSED")
    print("=" * 60)
