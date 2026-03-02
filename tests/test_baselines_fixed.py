"""Tests for fixed-step scheduler baselines (Phase 2, Week 5).

Validates DDIM20, DDIM50, DPMSolver20, DPMSolver50, Euler20, UniPC20, and
PNDM20 against a single-prompt, single-seed generate() call.  Each test checks:

  - Exactly one result returned.
  - Output image is 512x512 PIL Image.
  - Reported NFE matches the class constant.
  - Wall-clock time is positive.
  - results.jsonl exists in the output directory.

Run on HPC with: uv run python tests/test_baselines_fixed.py

References:
    - plan.md Phase 2 Week 5 (baseline validation)
    - experiment.md §2 (gate G1 smoke tests)
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

# Ensure project root is on sys.path for absolute src.* imports.
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.baselines.fixed_step import (
    DDIM20,
    DDIM50,
    DPMSolver20,
    DPMSolver50,
    Euler20,
    PNDM20,
    UniPC20,
)

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


def _run_baseline_test(cls, expected_nfe: int) -> bool:
    """Instantiate cls, run generate() on one prompt, check all assertions."""
    print(f"\n--- {cls.__name__} ---")
    tmp_dir = tempfile.mkdtemp()
    all_pass = True

    try:
        baseline = cls()
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

    except Exception as exc:
        print(f"  [FAIL] Exception raised: {exc}")
        all_pass = False

    return all_pass


# ---------------------------------------------------------------------------
# Individual test functions
# ---------------------------------------------------------------------------


def test_ddim20() -> bool:
    return _run_baseline_test(DDIM20, expected_nfe=20)


def test_ddim50() -> bool:
    return _run_baseline_test(DDIM50, expected_nfe=50)


def test_dpm_solver20() -> bool:
    return _run_baseline_test(DPMSolver20, expected_nfe=20)


def test_dpm_solver50() -> bool:
    return _run_baseline_test(DPMSolver50, expected_nfe=50)


def test_euler20() -> bool:
    return _run_baseline_test(Euler20, expected_nfe=20)


def test_unipc20() -> bool:
    return _run_baseline_test(UniPC20, expected_nfe=20)


def test_pndm20() -> bool:
    return _run_baseline_test(PNDM20, expected_nfe=20)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    results = {
        "DDIM20": test_ddim20(),
        "DDIM50": test_ddim50(),
        "DPMSolver20": test_dpm_solver20(),
        "DPMSolver50": test_dpm_solver50(),
        "Euler20": test_euler20(),
        "UniPC20": test_unipc20(),
        "PNDM20": test_pndm20(),
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
