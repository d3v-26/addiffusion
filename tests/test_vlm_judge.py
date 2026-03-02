"""Tests for src/evaluation/vlm_judge.py.

Uses unittest.mock to simulate OpenAI API responses — no real API key or
network calls are required.  All tests run on CPU with dummy PIL images.

Run:
    uv run python tests/test_vlm_judge.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from PIL import Image as PILImage

# Ensure project root is on the path when running directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.evaluation.vlm_judge import (
    VLMJudge,
    _encode_image,
    fit_bradley_terry,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dummy_image(width: int = 64, height: int = 64) -> PILImage.Image:
    """Return a small random-ish PIL image suitable for testing."""
    import numpy as np
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    return PILImage.fromarray(arr)


def _make_mock_response(content: str) -> MagicMock:
    """Build a mock object that mimics openai.ChatCompletion response structure."""
    msg = MagicMock()
    msg.content = content

    choice = MagicMock()
    choice.message = msg

    response = MagicMock()
    response.choices = [choice]
    return response


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEncodeImage(unittest.TestCase):
    """Tests for _encode_image helper."""

    def test_encode_image_returns_nonempty_base64(self) -> None:
        """_encode_image should return a non-empty base64 string."""
        img = PILImage.new("RGB", (64, 64), color=(128, 0, 255))
        encoded = _encode_image(img)
        self.assertIsInstance(encoded, str)
        self.assertGreater(len(encoded), 0)

    def test_encode_image_is_valid_base64(self) -> None:
        """_encode_image output should decode without error."""
        import base64
        img = PILImage.new("RGB", (32, 32))
        encoded = _encode_image(img)
        decoded = base64.b64decode(encoded)
        self.assertGreater(len(decoded), 0)

    def test_encode_image_larger_image(self) -> None:
        """Larger images produce larger base64 strings."""
        small = _encode_image(PILImage.new("RGB", (32, 32)))
        large = _encode_image(PILImage.new("RGB", (256, 256)))
        self.assertGreater(len(large), len(small))


class TestComparePairMock(unittest.TestCase):
    """Tests for VLMJudge.compare_pair() using mocked OpenAI client."""

    def _make_judge(self, mock_client: MagicMock) -> VLMJudge:
        """Instantiate VLMJudge with a pre-wired mock OpenAI client."""
        with patch("src.evaluation.vlm_judge.OpenAI", return_value=mock_client):
            judge = VLMJudge(model="gpt-4o-2024-11-20", rate_limit_rps=0.0)
        return judge

    def test_compare_pair_winner_a(self) -> None:
        """compare_pair returns winner='A' when API responds 'A\nLooks better'."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_mock_response(
            "A\nLooks better overall."
        )

        judge = self._make_judge(mock_client)
        img_a = _make_dummy_image()
        img_b = _make_dummy_image()

        # Force no-swap by seeding random — patch random.random to return 0.9 (>= 0.5)
        import random as _random
        with patch.object(_random, "random", return_value=0.9):
            result = judge.compare_pair(img_a, img_b, prompt="a red apple")

        self.assertEqual(result["winner"], "A")
        self.assertIn("justification", result)
        self.assertIn("order", result)
        self.assertEqual(result["order"], "AB")

    def test_compare_pair_winner_b(self) -> None:
        """compare_pair returns winner='B' when API responds 'B\nMore detailed'."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_mock_response(
            "B\nMore detailed rendering."
        )

        judge = self._make_judge(mock_client)
        img_a = _make_dummy_image()
        img_b = _make_dummy_image()

        import random as _random
        with patch.object(_random, "random", return_value=0.9):  # No swap
            result = judge.compare_pair(img_a, img_b, prompt="a mountain lake")

        self.assertEqual(result["winner"], "B")

    def test_compare_pair_tie(self) -> None:
        """compare_pair returns winner='Tie' when API responds 'Tie\n...'."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_mock_response(
            "Tie\nBoth look similar."
        )

        judge = self._make_judge(mock_client)
        img_a = _make_dummy_image()
        img_b = _make_dummy_image()

        import random as _random
        with patch.object(_random, "random", return_value=0.9):
            result = judge.compare_pair(img_a, img_b, prompt="a blue sky")

        self.assertEqual(result["winner"], "Tie")

    def test_compare_pair_swap_corrects_winner(self) -> None:
        """When A/B are swapped, a 'A' verdict from the model should become 'B'."""
        mock_client = MagicMock()
        # Model sees images in BA order and picks "A" (which is actually method B)
        mock_client.chat.completions.create.return_value = _make_mock_response(
            "A\nLeft image is clearer."
        )

        judge = self._make_judge(mock_client)
        img_a = _make_dummy_image()
        img_b = _make_dummy_image()

        import random as _random
        with patch.object(_random, "random", return_value=0.1):  # Force swap (< 0.5)
            result = judge.compare_pair(img_a, img_b, prompt="a forest path")

        # Presented "A" = actual method B => winner should be "B"
        self.assertEqual(result["winner"], "B")
        self.assertEqual(result["order"], "BA")

    def test_order_randomization(self) -> None:
        """compare_pair should produce both 'AB' and 'BA' orders across many calls."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_mock_response(
            "A\nTest response."
        )

        judge = self._make_judge(mock_client)
        img_a = _make_dummy_image()
        img_b = _make_dummy_image()

        orders = set()
        for _ in range(20):
            result = judge.compare_pair(img_a, img_b, prompt="test prompt")
            orders.add(result["order"])

        self.assertIn("AB", orders, "Expected some 'AB' order calls")
        self.assertIn("BA", orders, "Expected some 'BA' order calls")


class TestEvaluateDatasetMock(unittest.TestCase):
    """Tests for VLMJudge.evaluate_dataset() using mocked API and temp directories."""

    def _make_judge(self, mock_client: MagicMock) -> VLMJudge:
        with patch("src.evaluation.vlm_judge.OpenAI", return_value=mock_client):
            judge = VLMJudge(model="gpt-4o-2024-11-20", rate_limit_rps=0.0)
        return judge

    def _setup_image_dirs(
        self,
        tmpdir: str,
        n: int = 4,
    ) -> tuple[str, str]:
        """Create two directories with n dummy PNG images each."""
        dir_a = os.path.join(tmpdir, "method_a")
        dir_b = os.path.join(tmpdir, "method_b")
        os.makedirs(dir_a)
        os.makedirs(dir_b)
        for i in range(n):
            img = _make_dummy_image()
            img.save(os.path.join(dir_a, f"{i:04d}.png"))
            img.save(os.path.join(dir_b, f"{i:04d}.png"))
        return dir_a, dir_b

    def test_win_rate_all_a(self) -> None:
        """When all API responses return 'A', win_rate_a should be 1.0."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_mock_response(
            "A\nAlways A."
        )

        judge = self._make_judge(mock_client)

        with tempfile.TemporaryDirectory() as tmpdir:
            dir_a, dir_b = self._setup_image_dirs(tmpdir, n=4)
            prompts = ["prompt one", "prompt two", "prompt three", "prompt four"]
            output_file = os.path.join(tmpdir, "results.json")

            # Patch random to never swap, so "A" from model always = method A
            import random as _random
            with patch.object(_random, "random", return_value=0.9):
                summary = judge.evaluate_dataset(
                    method_a_dir=dir_a,
                    method_b_dir=dir_b,
                    prompts=prompts,
                    output_file=output_file,
                    n_pairs=4,
                    randomize_order=False,
                )

        self.assertAlmostEqual(summary["win_rate_a"], 1.0)
        self.assertAlmostEqual(summary["win_rate_b"], 0.0)
        self.assertEqual(summary["n_evaluated"], 4)

    def test_evaluate_dataset_creates_jsonl_cache(self) -> None:
        """evaluate_dataset should create a JSONL cache file alongside the JSON summary."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_mock_response(
            "Tie\nSimilar quality."
        )

        judge = self._make_judge(mock_client)

        with tempfile.TemporaryDirectory() as tmpdir:
            dir_a, dir_b = self._setup_image_dirs(tmpdir, n=2)
            prompts = ["prompt one", "prompt two"]
            output_file = os.path.join(tmpdir, "results.json")

            import random as _random
            with patch.object(_random, "random", return_value=0.9):
                judge.evaluate_dataset(
                    method_a_dir=dir_a,
                    method_b_dir=dir_b,
                    prompts=prompts,
                    output_file=output_file,
                    n_pairs=2,
                    randomize_order=False,
                )

            jsonl_path = os.path.join(tmpdir, "results.jsonl")
            self.assertTrue(os.path.exists(jsonl_path))

            with open(jsonl_path) as f:
                lines = [l.strip() for l in f if l.strip()]
            self.assertEqual(len(lines), 2)

    def test_evaluate_dataset_summary_keys(self) -> None:
        """evaluate_dataset summary should contain all required keys."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_mock_response("B\nBetter.")

        judge = self._make_judge(mock_client)

        with tempfile.TemporaryDirectory() as tmpdir:
            dir_a, dir_b = self._setup_image_dirs(tmpdir, n=2)
            output_file = os.path.join(tmpdir, "res.json")

            import random as _random
            with patch.object(_random, "random", return_value=0.9):
                summary = judge.evaluate_dataset(
                    method_a_dir=dir_a,
                    method_b_dir=dir_b,
                    prompts=["p1", "p2"],
                    output_file=output_file,
                    n_pairs=2,
                    randomize_order=False,
                )

        required = {
            "win_rate_a", "win_rate_b", "tie_rate",
            "n_evaluated", "method_a", "method_b", "wilson_ci_a",
        }
        missing = required - set(summary.keys())
        self.assertEqual(missing, set(), f"Missing keys: {missing}")


class TestBradleyTerry(unittest.TestCase):
    """Tests for fit_bradley_terry()."""

    def test_heavy_a_wins(self) -> None:
        """With 80 wins for A vs 15 for B, win_prob_a should be > 0.7."""
        result = fit_bradley_terry(80, 15, 5)
        self.assertIn("win_prob_a", result)
        self.assertGreater(result["win_prob_a"], 0.7)

    def test_equal_wins(self) -> None:
        """With equal wins, win_prob_a should be close to 0.5."""
        result = fit_bradley_terry(50, 50, 0)
        self.assertAlmostEqual(result["win_prob_a"], 0.5, places=2)

    def test_zero_comparisons(self) -> None:
        """With 0 comparisons, win_prob_a should be 0.5 (no information)."""
        result = fit_bradley_terry(0, 0, 0)
        self.assertAlmostEqual(result["win_prob_a"], 0.5, places=2)

    def test_heavy_b_wins(self) -> None:
        """With 10 wins for A vs 90 for B, win_prob_a should be < 0.3."""
        result = fit_bradley_terry(10, 90, 0)
        self.assertLess(result["win_prob_a"], 0.3)

    def test_result_keys(self) -> None:
        """fit_bradley_terry should return all required keys."""
        result = fit_bradley_terry(60, 30, 10)
        for key in ("strength_a", "strength_b", "win_prob_a"):
            self.assertIn(key, result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
