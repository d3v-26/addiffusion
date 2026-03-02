"""Tests for search-based baselines: RandomSearchBaseline and OracleStopBaseline (Phase 2, Week 5).

Validates the B2 (random search) and B4 (oracle stop) baselines against a
single-prompt, single-seed generate() call.  Each test checks:

  - Exactly one result returned.
  - Output image is 512x512 PIL Image.
  - Reported NFE is positive and within expected bounds.
  - Wall-clock time is positive.
  - results.jsonl exists in the output directory.
  - Method-specific metadata fields are populated correctly.

Both tests use reduced parameters (n_samples=2, max_steps=5) to keep
smoke-test runtime manageable on the HPC cluster.

Run on HPC with: uv run python tests/test_baselines_search.py

References:
    - plan.md Phase 2 Week 5 (baseline B2 and B4 validation)
    - experiment.md §2 (gate G1 smoke tests)
    - research.md §4.2 (oracle upper bound motivation)
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

# Ensure project root is on sys.path for absolute src.* imports.
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.baselines.oracle import OracleStopBaseline
from src.baselines.random_search import RandomSearchBaseline

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PROMPT = "a blue sky"
_SEED = 42


def _check(condition: bool, label: str) -> bool:
    if condition:
        print(f"  [PASS] {label}")
    else:
        print(f"  [FAIL] {label}")
    return condition


# ---------------------------------------------------------------------------
# RandomSearchBaseline test
# ---------------------------------------------------------------------------


def test_random_search() -> bool:
    """Test RandomSearchBaseline with n_samples=2 (fast smoke test).

    Uses n_samples=2 and steps=20 to limit runtime.  Verifies:
      - One result returned.
      - Image size 512x512.
      - NFE == n_samples * steps == 40.
      - time_s > 0.
      - results.jsonl exists.
      - metadata contains "best_clip_score" key with a finite float value.
    """
    print("\n--- RandomSearchBaseline (n_samples=2) ---")
    tmp_dir = tempfile.mkdtemp()
    all_pass = True

    n_samples = 2
    steps = 20
    expected_nfe = n_samples * steps

    try:
        baseline = RandomSearchBaseline(n_samples=n_samples, steps=steps)
        results = baseline.generate([_PROMPT], [_SEED], tmp_dir)

        all_pass &= _check(len(results) == 1, "len(results) == 1")
        all_pass &= _check(
            results[0].image.size == (512, 512),
            f"image.size == (512, 512)  [got {results[0].image.size}]",
        )
        all_pass &= _check(
            results[0].nfe == expected_nfe,
            f"nfe == {expected_nfe}  [got {results[0].nfe}]",
        )
        all_pass &= _check(
            results[0].time_s > 0,
            f"time_s > 0  [got {results[0].time_s:.4f}s]",
        )
        jsonl_path = os.path.join(tmp_dir, "results.jsonl")
        all_pass &= _check(
            os.path.exists(jsonl_path),
            f"results.jsonl exists at {jsonl_path}",
        )

        meta = results[0].metadata
        has_clip_key = "best_clip_score" in meta
        all_pass &= _check(has_clip_key, "'best_clip_score' in metadata")
        if has_clip_key:
            clip_val = meta["best_clip_score"]
            all_pass &= _check(
                isinstance(clip_val, float) and -1.0 <= clip_val <= 1.0,
                f"best_clip_score is float in [-1,1]  [got {clip_val}]",
            )

    except Exception as exc:
        print(f"  [FAIL] Exception raised: {exc}")
        all_pass = False

    return all_pass


# ---------------------------------------------------------------------------
# OracleStopBaseline test
# ---------------------------------------------------------------------------


def test_oracle_stop() -> bool:
    """Test OracleStopBaseline with max_steps=5 (fast smoke test).

    Uses max_steps=5 to limit runtime.  Verifies:
      - One result returned.
      - Image size 512x512.
      - NFE > 0 and NFE <= max_steps.
      - time_s > 0.
      - results.jsonl exists.
      - metadata contains "peak_nfe" <= max_steps and "peak_clip" > 0.
    """
    print("\n--- OracleStopBaseline (max_steps=5) ---")
    tmp_dir = tempfile.mkdtemp()
    all_pass = True

    max_steps = 5

    try:
        baseline = OracleStopBaseline(max_steps=max_steps)
        results = baseline.generate([_PROMPT], [_SEED], tmp_dir)

        all_pass &= _check(len(results) == 1, "len(results) == 1")
        all_pass &= _check(
            results[0].image.size == (512, 512),
            f"image.size == (512, 512)  [got {results[0].image.size}]",
        )
        reported_nfe = results[0].nfe
        all_pass &= _check(
            0 < reported_nfe <= max_steps,
            f"0 < nfe <= {max_steps}  [got {reported_nfe}]",
        )
        all_pass &= _check(
            results[0].time_s > 0,
            f"time_s > 0  [got {results[0].time_s:.4f}s]",
        )
        jsonl_path = os.path.join(tmp_dir, "results.jsonl")
        all_pass &= _check(
            os.path.exists(jsonl_path),
            f"results.jsonl exists at {jsonl_path}",
        )

        meta = results[0].metadata
        has_peak_nfe = "peak_nfe" in meta
        has_peak_clip = "peak_clip" in meta
        all_pass &= _check(has_peak_nfe, "'peak_nfe' in metadata")
        all_pass &= _check(has_peak_clip, "'peak_clip' in metadata")
        if has_peak_nfe:
            all_pass &= _check(
                meta["peak_nfe"] <= max_steps,
                f"peak_nfe <= {max_steps}  [got {meta['peak_nfe']}]",
            )
        if has_peak_clip:
            all_pass &= _check(
                meta["peak_clip"] > 0,
                f"peak_clip > 0  [got {meta['peak_clip']}]",
            )

    except Exception as exc:
        print(f"  [FAIL] Exception raised: {exc}")
        all_pass = False

    return all_pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    results = {
        "RandomSearchBaseline": test_random_search(),
        "OracleStopBaseline": test_oracle_stop(),
    }

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    all_ok = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        all_ok &= passed

    print("=" * 50)
    if all_ok:
        print("All tests PASSED.")
        sys.exit(0)
    else:
        print("Some tests FAILED.")
        sys.exit(1)
