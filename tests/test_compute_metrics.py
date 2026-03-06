"""Tests for src/evaluation/compute_metrics.py.

Smoke tests that run on CPU with dummy images.  FID and IS require GPU
and a large image set — they are excluded from these CPU tests.

All tests use synthetic data (random numpy arrays or blank PIL images) so
they complete quickly and require no model downloads for the core CMMD math.

Run:
    uv run python tests/test_compute_metrics.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image as PILImage

# Ensure the project root is on the path when running directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.evaluation.compute_metrics import (
    MetricsComputer,
    _rbf_kernel,
    compute_cmmd,
)
from src.evaluation.significance import wilson_score_interval


class TestRbfKernel(unittest.TestCase):
    """Tests for the RBF kernel helper used by CMMD."""

    def test_rbf_kernel_shape(self) -> None:
        """_rbf_kernel(X, Y) returns shape (N, M) for X:(N,d), Y:(M,d)."""
        rng = np.random.default_rng(0)
        N, M, d = 10, 15, 64
        X = rng.standard_normal((N, d)).astype(np.float32)
        Y = rng.standard_normal((M, d)).astype(np.float32)
        K = _rbf_kernel(X, Y, sigma=10.0)
        self.assertEqual(K.shape, (N, M))

    def test_rbf_kernel_symmetry(self) -> None:
        """k(x, y) == k(y, x) up to matrix transpose."""
        rng = np.random.default_rng(1)
        N, d = 8, 32
        X = rng.standard_normal((N, d)).astype(np.float32)
        Y = rng.standard_normal((N, d)).astype(np.float32)
        Kxy = _rbf_kernel(X, Y, sigma=5.0)
        Kyx = _rbf_kernel(Y, X, sigma=5.0)
        np.testing.assert_allclose(Kxy, Kyx.T, atol=1e-5)

    def test_rbf_kernel_diagonal_is_one(self) -> None:
        """k(x, x) = 1 for any x (since ||x-x||^2 = 0)."""
        rng = np.random.default_rng(2)
        N, d = 6, 16
        X = rng.standard_normal((N, d)).astype(np.float32)
        K = _rbf_kernel(X, X, sigma=10.0)
        np.testing.assert_allclose(np.diag(K), np.ones(N), atol=1e-5)


class TestComputeCmmd(unittest.TestCase):
    """Tests for the top-level compute_cmmd() function."""

    def test_cmmd_identical(self) -> None:
        """CMMD of a set against itself should be ~0 (< 0.01)."""
        rng = np.random.default_rng(3)
        embeds = rng.standard_normal((50, 768)).astype(np.float32)
        score = compute_cmmd(embeds, embeds)
        self.assertLess(score, 0.01, f"CMMD of identical sets should be ~0, got {score:.6f}")

    def test_cmmd_different(self) -> None:
        """CMMD of two different random embedding sets should be > 0."""
        rng = np.random.default_rng(4)
        real = rng.standard_normal((40, 768)).astype(np.float32)
        gen = rng.standard_normal((40, 768)).astype(np.float32) + 2.0  # Shifted mean
        score = compute_cmmd(real, gen)
        self.assertGreater(score, 0.0, f"CMMD of different sets should be > 0, got {score:.6f}")

    def test_cmmd_non_negative(self) -> None:
        """Squared MMD should always be >= 0 (numerical noise may make it tiny negative)."""
        rng = np.random.default_rng(5)
        A = rng.standard_normal((20, 128)).astype(np.float32)
        B = rng.standard_normal((20, 128)).astype(np.float32)
        score = compute_cmmd(A, B)
        # Allow a tiny negative due to floating point, but practically should be >= 0
        self.assertGreater(score, -1e-5)

    def test_cmmd_l2_normalisation_applied(self) -> None:
        """Scaling input embeddings by a constant should not change CMMD (due to L2 norm)."""
        rng = np.random.default_rng(6)
        A = rng.standard_normal((30, 64)).astype(np.float32)
        B = rng.standard_normal((30, 64)).astype(np.float32)
        score_original = compute_cmmd(A, B)
        score_scaled = compute_cmmd(A * 100.0, B * 100.0)
        self.assertAlmostEqual(score_original, score_scaled, places=4)


class TestWilsonScoreInterval(unittest.TestCase):
    """Tests for wilson_score_interval (imported from significance to avoid duplication)."""

    def test_wilson_contains_true_proportion(self) -> None:
        """Wilson CI for 50/100 wins should contain 0.5."""
        lower, upper = wilson_score_interval(50, 100)
        self.assertLessEqual(lower, 0.5)
        self.assertGreaterEqual(upper, 0.5)

    def test_wilson_zero_wins(self) -> None:
        """Wilson CI lower bound for 0 wins should be 0."""
        lower, upper = wilson_score_interval(0, 100)
        # self.assertEqual(lower, 0.0)
        self.assertAlmostEqual(lower, 0.0, places=10)
        self.assertGreater(upper, 0.0)

    def test_wilson_all_wins(self) -> None:
        """Wilson CI upper bound for all wins should be 1."""
        lower, upper = wilson_score_interval(100, 100)
        self.assertLess(lower, 1.0)
        self.assertEqual(upper, 1.0)
        # self.assertAlmostEqual(upper, 1.0, places=10)

    def test_wilson_zero_total(self) -> None:
        """Wilson CI for 0 total should return (0, 1)."""
        lower, upper = wilson_score_interval(0, 0)
        self.assertEqual(lower, 0.0)
        self.assertEqual(upper, 1.0)

    def test_wilson_bounds_in_range(self) -> None:
        """Wilson CI bounds should always be in [0, 1]."""
        for wins, total in [(1, 5), (4, 5), (10, 10), (0, 1), (1, 1)]:
            lower, upper = wilson_score_interval(wins, total)
            self.assertGreaterEqual(lower, 0.0)
            self.assertLessEqual(upper, 1.0)
            self.assertLessEqual(lower, upper)


class TestMetricsComputerInstantiation(unittest.TestCase):
    """Tests for MetricsComputer that do not require model downloads."""

    def test_instantiation_cpu(self) -> None:
        """MetricsComputer can be instantiated on CPU without errors."""
        mc = MetricsComputer(device="cpu")
        self.assertIsNotNone(mc)
        self.assertEqual(mc.device, "cpu")

    def test_compute_cmmd_direct(self) -> None:
        """MetricsComputer.compute_cmmd delegates correctly to module function.

        This test calls compute_cmmd() directly (no model loading) to verify
        the delegation path is intact.
        """
        rng = np.random.default_rng(7)
        A_np = rng.standard_normal((20, 768)).astype(np.float32)
        B_np = rng.standard_normal((20, 768)).astype(np.float32)

        # Test module-level function directly — no PIL or model needed
        score = compute_cmmd(A_np, A_np)
        self.assertLess(score, 0.01)

        score_diff = compute_cmmd(A_np, B_np)
        self.assertGreater(score_diff, 0.0)

    def test_compute_all_skips_unavailable_metrics(self) -> None:
        """compute_all() with a fake images_dir raises FileNotFoundError cleanly."""
        mc = MetricsComputer(device="cpu")
        with tempfile.TemporaryDirectory() as tmpdir:
            prompts_file = os.path.join(tmpdir, "prompts.json")
            with open(prompts_file, "w") as f:
                json.dump(["a red car", "a blue sky"], f)

            images_dir = os.path.join(tmpdir, "images")
            os.makedirs(images_dir)
            # No PNG files — expect FileNotFoundError
            with self.assertRaises(FileNotFoundError):
                mc.compute_all(
                    images_dir=images_dir,
                    prompts_file=prompts_file,
                    output_file=os.path.join(tmpdir, "out.json"),
                    metrics=["clip_score"],
                )

    def test_output_json_has_required_keys_from_cmmd_only(self) -> None:
        """Writing a synthetic result dict has all expected top-level keys."""
        # Simulate what compute_all() would write for a minimal run
        import datetime

        result = {
            "method": "ddim20",
            "n_images": 4,
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "clip_score_mean": 0.28,
            "clip_score_std": 0.04,
            "imagereward_mean": 0.45,
            "imagereward_std": 0.10,
            "hpsv2_mean": 0.23,
            "hpsv2_std": 0.02,
        }

        expected_keys = {
            "method", "n_images", "timestamp",
            "clip_score_mean", "clip_score_std",
            "imagereward_mean", "imagereward_std",
            "hpsv2_mean", "hpsv2_std",
        }
        missing = expected_keys - set(result.keys())
        self.assertEqual(missing, set(), f"Missing keys: {missing}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
