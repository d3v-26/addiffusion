"""Tests for src/evaluation/significance.py.

Pure CPU tests — no GPU or model loading required.  All computations use
synthetic numpy arrays.

Run:
    uv run python tests/test_significance.py
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

# Ensure project root is on the path when running directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.evaluation.significance import (
    paired_bootstrap_test,
    wilson_score_interval,
)


# ---------------------------------------------------------------------------
# Tests for paired_bootstrap_test
# ---------------------------------------------------------------------------

class TestPairedBootstrapTest(unittest.TestCase):
    """Tests for paired_bootstrap_test()."""

    def test_identical_arrays_high_p_value(self) -> None:
        """paired_bootstrap_test(arr, arr) should give p_value > 0.5.

        Under H0, the difference is exactly 0 so the p-value should be large.
        """
        arr = np.random.default_rng(0).standard_normal(100)
        result = paired_bootstrap_test(arr, arr, n_resamples=5_000, seed=42)
        self.assertGreater(result["p_value"], 0.5,
                           f"Expected p > 0.5 for identical arrays, got {result['p_value']}")

    def test_identical_arrays_zero_effect_size(self) -> None:
        """effect_size should be 0 when arrays are identical."""
        arr = np.ones(50)
        result = paired_bootstrap_test(arr, arr, n_resamples=2_000, seed=7)
        self.assertAlmostEqual(result["effect_size"], 0.0, places=5)

    def test_identical_arrays_zero_mean_diff(self) -> None:
        """mean_diff should be exactly 0 when arrays are identical."""
        arr = np.linspace(0, 1, 80)
        result = paired_bootstrap_test(arr, arr, n_resamples=2_000, seed=99)
        self.assertAlmostEqual(result["mean_diff"], 0.0, places=10)

    def test_clearly_different_arrays_low_p_value(self) -> None:
        """Arrays with very different means should yield p_value < 0.05."""
        arr_a = np.ones(100)
        arr_b = np.zeros(100)
        result = paired_bootstrap_test(arr_a, arr_b, n_resamples=5_000, seed=42)
        self.assertLess(result["p_value"], 0.05,
                        f"Expected p < 0.05 for clearly different arrays, got {result['p_value']}")

    def test_clearly_different_arrays_positive_effect_size(self) -> None:
        """Effect size should be positive when metric_a > metric_b."""
        arr_a = np.ones(100)
        arr_b = np.zeros(100)
        result = paired_bootstrap_test(arr_a, arr_b, n_resamples=5_000, seed=42)
        self.assertGreater(result["effect_size"], 0.0)

    def test_clearly_different_arrays_positive_mean_diff(self) -> None:
        """mean_diff = mean(A) - mean(B) should be 1.0 for ones vs zeros."""
        arr_a = np.ones(100)
        arr_b = np.zeros(100)
        result = paired_bootstrap_test(arr_a, arr_b)
        self.assertAlmostEqual(result["mean_diff"], 1.0, places=10)

    def test_result_keys(self) -> None:
        """Result dict should contain all required keys."""
        arr = np.random.default_rng(5).standard_normal(50)
        result = paired_bootstrap_test(arr, arr * 0.9, n_resamples=1_000, seed=1)
        required = {"p_value", "ci_95", "effect_size", "mean_diff", "bootstrap_mean"}
        missing = required - set(result.keys())
        self.assertEqual(missing, set(), f"Missing keys: {missing}")

    def test_ci_contains_mean_diff(self) -> None:
        """The 95% CI should contain the observed mean_diff."""
        rng = np.random.default_rng(10)
        arr_a = rng.standard_normal(80) + 0.3
        arr_b = rng.standard_normal(80)
        result = paired_bootstrap_test(arr_a, arr_b, n_resamples=5_000, seed=42)
        lower, upper = result["ci_95"]
        self.assertLessEqual(lower, result["mean_diff"])
        self.assertGreaterEqual(upper, result["mean_diff"])

    def test_alternative_greater(self) -> None:
        """'greater' alternative should give low p-value when A > B."""
        arr_a = np.ones(100)
        arr_b = np.zeros(100)
        result = paired_bootstrap_test(arr_a, arr_b, n_resamples=5_000,
                                       alternative="greater", seed=42)
        self.assertLess(result["p_value"], 0.05)

    def test_alternative_less(self) -> None:
        """'less' alternative should give high p-value when A > B (wrong direction)."""
        arr_a = np.ones(100)
        arr_b = np.zeros(100)
        result = paired_bootstrap_test(arr_a, arr_b, n_resamples=5_000,
                                       alternative="less", seed=42)
        self.assertGreater(result["p_value"], 0.5)

    def test_mismatched_shapes_raises(self) -> None:
        """Mismatched array shapes should raise ValueError."""
        arr_a = np.ones(50)
        arr_b = np.ones(60)
        with self.assertRaises(ValueError):
            paired_bootstrap_test(arr_a, arr_b)

    def test_p_value_uniform_under_null(self) -> None:
        """Under H0 (A == B), p-values should be approximately uniform.

        We verify that at least some p-values are > 0.5 across repeated
        trials with different random seeds, indicating the test is not
        systematically biased toward rejection.
        """
        rng = np.random.default_rng(0)
        p_values = []
        for trial in range(20):
            arr = rng.standard_normal(100)
            result = paired_bootstrap_test(arr, arr, n_resamples=2_000, seed=trial)
            p_values.append(result["p_value"])

        n_above_half = sum(p > 0.5 for p in p_values)
        self.assertGreater(n_above_half, 5,
                           f"Expected many p-values > 0.5 under H0, got {n_above_half}/20. "
                           f"p_values: {p_values}")

    def test_bootstrap_ci_coverage(self) -> None:
        """Bootstrap CI should contain the true mean difference ~95% of the time.

        Runs 20 independent trials and checks coverage >= 80% (conservative
        threshold for 20 trials; exact 95% would need hundreds).
        """
        rng = np.random.default_rng(2024)
        true_diff = 0.3
        covered = 0

        for trial in range(20):
            arr_a = rng.standard_normal(100) + true_diff
            arr_b = rng.standard_normal(100)
            result = paired_bootstrap_test(arr_a, arr_b, n_resamples=3_000, seed=trial)
            lower, upper = result["ci_95"]
            if lower <= true_diff <= upper:
                covered += 1

        coverage = covered / 20
        self.assertGreaterEqual(coverage, 0.80,
                                f"Expected >=80% coverage, got {coverage:.0%} ({covered}/20)")


# ---------------------------------------------------------------------------
# Tests for wilson_score_interval
# ---------------------------------------------------------------------------

class TestWilsonScoreInterval(unittest.TestCase):
    """Tests for wilson_score_interval()."""

    def test_fifty_fifty_contains_half(self) -> None:
        """CI for 50/100 wins should contain 0.5."""
        lower, upper = wilson_score_interval(50, 100)
        self.assertLessEqual(lower, 0.5)
        self.assertGreaterEqual(upper, 0.5)

    def test_zero_wins_lower_is_zero(self) -> None:
        """Lower bound for 0/100 wins should be 0."""
        lower, upper = wilson_score_interval(0, 100)
        self.assertEqual(lower, 0.0)
        self.assertGreater(upper, 0.0)

    def test_all_wins_upper_is_one(self) -> None:
        """Upper bound for 100/100 wins should be 1."""
        lower, upper = wilson_score_interval(100, 100)
        self.assertLess(lower, 1.0)
        self.assertEqual(upper, 1.0)

    def test_bounds_always_in_range(self) -> None:
        """Lower and upper bounds should always be in [0, 1]."""
        test_cases = [(0, 1), (1, 1), (0, 5), (3, 5), (5, 5), (0, 100), (100, 100)]
        for wins, total in test_cases:
            lower, upper = wilson_score_interval(wins, total)
            self.assertGreaterEqual(lower, 0.0, f"lower < 0 for wins={wins}, total={total}")
            self.assertLessEqual(upper, 1.0, f"upper > 1 for wins={wins}, total={total}")
            self.assertLessEqual(lower, upper, f"lower > upper for wins={wins}, total={total}")

    def test_zero_total_returns_full_interval(self) -> None:
        """Wilson CI for 0 total observations should return (0, 1)."""
        lower, upper = wilson_score_interval(0, 0)
        self.assertEqual(lower, 0.0)
        self.assertEqual(upper, 1.0)

    def test_higher_confidence_wider_interval(self) -> None:
        """A 99% CI should be wider than a 90% CI for the same data."""
        lo90, hi90 = wilson_score_interval(50, 100, confidence=0.90)
        lo99, hi99 = wilson_score_interval(50, 100, confidence=0.99)
        self.assertLess(lo99, lo90)
        self.assertGreater(hi99, hi90)

    def test_interval_width_decreases_with_more_data(self) -> None:
        """More data should yield a narrower CI at the same win rate."""
        lo_small, hi_small = wilson_score_interval(5, 10)
        lo_large, hi_large = wilson_score_interval(500, 1000)
        width_small = hi_small - lo_small
        width_large = hi_large - lo_large
        self.assertGreater(width_small, width_large)


if __name__ == "__main__":
    unittest.main(verbosity=2)
