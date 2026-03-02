"""Tests for SAGBaseline and AttendExciteBaseline.

Validates that both attention-guided baselines produce a single 512x512 PIL
image, record a positive NFE and wall-clock time, and write a results.jsonl
file to the output directory.

Run:
    uv run python tests/test_baselines_attention.py
Requires: GPU with ~10 GB VRAM and HuggingFace weights for SD 1.5.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.baselines.attend_excite import AttendExciteBaseline
from src.baselines.sag import SAGBaseline

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
    # Each line must be valid JSON
    for line in lines:
        json.loads(line)


# ---------------------------------------------------------------------------
# SAG test
# ---------------------------------------------------------------------------

def test_sag_baseline() -> None:
    """SAGBaseline: single image, correct shape, positive NFE and time."""
    print("=" * 60)
    print("TEST: SAGBaseline (B6 — Self-Attention Guidance, 20 steps)")
    print("=" * 60)

    try:
        baseline = SAGBaseline(device=DEVICE)
        output_dir = tempfile.mkdtemp(prefix="test_sag_")

        results = baseline.generate(
            prompts=[PROMPT],
            seeds=[SEED],
            output_dir=output_dir,
        )

        assert len(results) == 1, f"Expected 1 result, got {len(results)}"

        r = results[0]
        assert r.image.size == (512, 512), (
            f"Expected image size (512, 512), got {r.image.size}"
        )
        assert r.nfe > 0, f"Expected nfe > 0, got {r.nfe}"
        assert r.time_s > 0, f"Expected time_s > 0, got {r.time_s}"
        assert r.method_name == "sag_20", (
            f"Expected method_name 'sag_20', got {r.method_name!r}"
        )
        assert r.prompt == PROMPT, f"Prompt mismatch: {r.prompt!r}"
        assert r.seed == SEED, f"Seed mismatch: {r.seed}"
        assert "sag_scale" in r.metadata, "metadata missing 'sag_scale'"

        _assert_jsonl_exists(output_dir)

        print(f"  Output dir: {output_dir}")
        print(f"  Image size: {r.image.size}")
        print(f"  NFE: {r.nfe}  |  time_s: {r.time_s:.2f}s")
        print(f"  sag_scale: {r.metadata['sag_scale']}")
        print("[PASS] SAGBaseline")

    except Exception as exc:
        print(f"[FAIL] SAGBaseline: {exc}")
        raise


# ---------------------------------------------------------------------------
# Attend-and-Excite test
# ---------------------------------------------------------------------------

def test_attend_excite_baseline() -> None:
    """AttendExciteBaseline: single image, correct shape, token indices populated."""
    print("\n" + "=" * 60)
    print("TEST: AttendExciteBaseline (B6 — Attend-and-Excite, 50 steps)")
    print("=" * 60)

    try:
        baseline = AttendExciteBaseline(device=DEVICE)
        output_dir = tempfile.mkdtemp(prefix="test_ae_")

        results = baseline.generate(
            prompts=[PROMPT],
            seeds=[SEED],
            output_dir=output_dir,
        )

        assert len(results) == 1, f"Expected 1 result, got {len(results)}"

        r = results[0]
        assert r.image.size == (512, 512), (
            f"Expected image size (512, 512), got {r.image.size}"
        )
        assert r.nfe > 0, f"Expected nfe > 0, got {r.nfe}"
        assert r.time_s > 0, f"Expected time_s > 0, got {r.time_s}"
        assert r.method_name == "attend_excite_50", (
            f"Expected method_name 'attend_excite_50', got {r.method_name!r}"
        )
        assert r.prompt == PROMPT, f"Prompt mismatch: {r.prompt!r}"
        assert r.seed == SEED, f"Seed mismatch: {r.seed}"
        assert "token_indices" in r.metadata, "metadata missing 'token_indices'"
        assert isinstance(r.metadata["token_indices"], list), (
            "metadata['token_indices'] should be a list"
        )
        assert len(r.metadata["token_indices"]) > 0, (
            "token_indices should be non-empty for a non-trivial prompt"
        )
        assert "max_iter_to_alter" in r.metadata, (
            "metadata missing 'max_iter_to_alter'"
        )

        _assert_jsonl_exists(output_dir)

        print(f"  Output dir: {output_dir}")
        print(f"  Image size: {r.image.size}")
        print(f"  NFE: {r.nfe}  |  time_s: {r.time_s:.2f}s")
        print(f"  token_indices: {r.metadata['token_indices']}")
        print(f"  max_iter_to_alter: {r.metadata['max_iter_to_alter']}")
        print("[PASS] AttendExciteBaseline")

    except Exception as exc:
        print(f"[FAIL] AttendExciteBaseline: {exc}")
        raise


# ---------------------------------------------------------------------------
# Token-index helper unit test (no GPU needed)
# ---------------------------------------------------------------------------

def test_attend_excite_token_indices_unit() -> None:
    """_get_token_indices: BOS and EOS are excluded from returned indices."""
    print("\n" + "=" * 60)
    print("TEST: AttendExciteBaseline._get_token_indices (unit, no GPU)")
    print("=" * 60)

    try:
        # Instantiate without GPU to test the tokenizer helper alone.
        baseline = AttendExciteBaseline(device=DEVICE)
        indices = baseline._get_token_indices(PROMPT)

        # Must be a non-empty list of ints
        assert isinstance(indices, list), "Expected list"
        assert all(isinstance(i, int) for i in indices), "All indices must be int"
        assert len(indices) > 0, "Expected at least one content token"

        # BOS (0) and EOS (last position) must not be present
        token_ids = baseline.pipe.tokenizer(PROMPT).input_ids
        last_pos = len(token_ids) - 1
        assert 0 not in indices, "BOS index 0 must not appear in token_indices"
        assert last_pos not in indices, (
            f"EOS index {last_pos} must not appear in token_indices"
        )

        print(f"  Prompt: {PROMPT!r}")
        print(f"  Full token_ids: {token_ids}")
        print(f"  Content token_indices: {indices}")
        print("[PASS] _get_token_indices excludes BOS and EOS")

    except Exception as exc:
        print(f"[FAIL] _get_token_indices unit test: {exc}")
        raise


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_sag_baseline()
    test_attend_excite_baseline()
    test_attend_excite_token_indices_unit()

    print("\n" + "=" * 60)
    print("ALL ATTENTION BASELINE TESTS PASSED")
    print("=" * 60)
