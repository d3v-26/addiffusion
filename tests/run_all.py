"""Run all tests in sequence.

Usage:
    uv run python tests/run_all.py                           # All tests (requires GPU + models)
    uv run python tests/run_all.py --cpu-only                # CPU-only tests (no GPU needed)
    uv run python tests/run_all.py --model <model_id>        # Custom model path
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

# Tests that don't need GPU or model weights
CPU_TESTS = [
    "tests/test_networks.py",
    "tests/test_ppo.py",
]

# Tests that need GPU + model weights
GPU_TESTS = [
    "tests/test_pipeline.py",
    "tests/test_attention.py",
    "tests/test_refine.py",
    "tests/test_state.py",
    "tests/test_reward.py",
    "tests/test_episode.py",
]

# Full integration test (GPU + all model weights)
INTEGRATION_TESTS = [
    "tests/test_training_toy.py",
]


def run_test(test_path: str, model_id: str = None) -> bool:
    """Run a single test file. Returns True if passed."""
    cmd = [sys.executable, test_path]
    if model_id:
        cmd.append(model_id)

    print(f"\n{'=' * 60}")
    print(f"RUNNING: {test_path}")
    print(f"{'=' * 60}")

    t_start = time.time()
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    elapsed = time.time() - t_start

    passed = result.returncode == 0
    status = "PASSED" if passed else "FAILED"
    print(f"\n[{status}] {test_path} ({elapsed:.1f}s)")
    return passed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-only", action="store_true", help="Only run CPU tests")
    parser.add_argument("--no-integration", action="store_true", help="Skip integration tests")
    parser.add_argument("--model", type=str, default=None, help="Model ID override")
    args = parser.parse_args()

    model_id = args.model

    tests_to_run = list(CPU_TESTS)
    if not args.cpu_only:
        tests_to_run.extend(GPU_TESTS)
        if not args.no_integration:
            tests_to_run.extend(INTEGRATION_TESTS)

    results = {}
    t_total = time.time()

    for test in tests_to_run:
        results[test] = run_test(test, model_id if test not in CPU_TESTS else None)

    elapsed_total = time.time() - t_total

    # Summary
    print(f"\n{'=' * 60}")
    print(f"TEST SUMMARY ({elapsed_total:.1f}s total)")
    print(f"{'=' * 60}")

    passed = sum(1 for v in results.values() if v)
    failed = sum(1 for v in results.values() if not v)

    for test, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"  [{status}] {test}")

    print(f"\n{passed}/{passed + failed} tests passed")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
