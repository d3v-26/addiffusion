"""GPT-4o pairwise image evaluation for AdDiffusion (Phase 2, Week 6).

Implements VLM-as-Judge using OpenAI's GPT-4o model to compare pairs of
generated images and determine which is higher quality. Results are cached
incrementally to JSONL so interrupted runs can resume.

Key design decisions:
    - Model: gpt-4o-2024-11-20 (NOT the deprecated gpt-4-vision-preview)
    - API: client.chat.completions.create() (OpenAI SDK, not Anthropic)
    - Images: base64-encoded PNG passed as image_url content blocks
    - Positional bias mitigation: random A/B order swap with 50% probability
    - Incremental JSONL caching for resume support

References:
    - plan.md Phase 2 Week 6 (VLM-as-Judge evaluation)
    - research.md §4.3 (human / VLM evaluation protocol)
    - CLAUDE.md Fallback Strategies (GPT-4o as tiebreaker)

Usage:
    OPENAI_API_KEY=sk-... uv run python src/evaluation/vlm_judge.py \\
        --method_a outputs/baselines/ddim50/ \\
        --method_b outputs/baselines/agent/ \\
        --prompts data/drawbench/prompts.json \\
        --output outputs/evaluation/vlm_judge/agent_vs_ddim50.json \\
        --n_pairs 200
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image as PILImage
from openai import OpenAI


# ---------------------------------------------------------------------------
# Image encoding helper
# ---------------------------------------------------------------------------

def _encode_image(img: PILImage.Image) -> str:
    """Base64-encode a PIL image as PNG.

    Args:
        img: PIL image to encode.

    Returns:
        Base64 string (UTF-8 decoded).
    """
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ---------------------------------------------------------------------------
# Bradley-Terry model
# ---------------------------------------------------------------------------

def fit_bradley_terry(wins_a: int, wins_b: int, ties: int) -> dict:
    """Simple Bradley-Terry model for two-player pairwise comparison.

    Fits log-likelihood to estimate player strengths, accounting for ties via
    the Davidson extension where applicable.  For simplicity this implementation
    treats each tie as half-a-win for each player.

    Args:
        wins_a: Number of comparisons won by method A.
        wins_b: Number of comparisons won by method B.
        ties:   Number of tied comparisons.

    Returns:
        Dict with keys:
            strength_a  : Fitted strength parameter for A (log-scale)
            strength_b  : Fitted strength parameter for B (log-scale, fixed to 0)
            win_prob_a  : Predicted win probability for A against B
    """
    from scipy.optimize import minimize

    # Treat ties as 0.5 wins each for numerical stability
    effective_wins_a = wins_a + 0.5 * ties
    effective_wins_b = wins_b + 0.5 * ties
    total = effective_wins_a + effective_wins_b

    if total == 0:
        return {"strength_a": 0.0, "strength_b": 0.0, "win_prob_a": 0.5}

    # Log-likelihood: L(s_a) = wins_a * log(p_a) + wins_b * log(1-p_a)
    # where p_a = exp(s_a) / (exp(s_a) + 1)  (s_b fixed at 0)
    def neg_log_likelihood(params: np.ndarray) -> float:
        s_a = params[0]
        log_p_a = s_a - np.logaddexp(s_a, 0.0)       # log sigmoid(s_a)
        log_p_b = 0.0 - np.logaddexp(s_a, 0.0)       # log sigmoid(-s_a)
        nll = -(effective_wins_a * log_p_a + effective_wins_b * log_p_b)
        return float(nll)

    result = minimize(neg_log_likelihood, x0=np.array([0.0]), method="L-BFGS-B")
    s_a = float(result.x[0])
    win_prob_a = 1.0 / (1.0 + math.exp(-s_a))

    return {
        "strength_a": s_a,
        "strength_b": 0.0,
        "win_prob_a": win_prob_a,
    }


# ---------------------------------------------------------------------------
# VLMJudge
# ---------------------------------------------------------------------------

class VLMJudge:
    """GPT-4o pairwise image evaluator.

    Compares pairs of images from two methods on a shared prompt set and
    aggregates win rates with Wilson score confidence intervals.

    Args:
        model:           OpenAI model ID to use.
        rate_limit_rps:  Maximum API requests per second (sleep between calls).
    """

    PROMPT_TEMPLATE = (
        'You are evaluating two AI-generated images for the prompt: "{prompt}"\n\n'
        "Image A is on the left, Image B is on the right.\n\n"
        "Consider: (1) text-image alignment, (2) visual quality and coherence,\n"
        "(3) absence of artifacts, (4) aesthetic appeal.\n\n"
        "Which image is better overall? Respond with exactly one of: A, B, or Tie\n"
        "Then on the next line, write a brief justification (1-2 sentences)."
    )

    def __init__(
        self,
        model: str = "gpt-4o-2024-11-20",
        rate_limit_rps: float = 1.0,
    ) -> None:
        self.model = model
        self.rate_limit_rps = rate_limit_rps
        self._sleep_s = 1.0 / rate_limit_rps if rate_limit_rps > 0 else 0.0

        self._client = OpenAI()  # Reads OPENAI_API_KEY from environment

    def compare_pair(
        self,
        image_a: PILImage.Image,
        image_b: PILImage.Image,
        prompt: str,
    ) -> dict:
        """Compare two images for the given prompt using GPT-4o.

        A/B order is randomly swapped with 50% probability to mitigate
        positional bias.  The 'order' field records the actual presentation
        order so callers can un-swap results if needed.

        Args:
            image_a: Image from method A.
            image_b: Image from method B.
            prompt:  The text prompt both images were generated from.

        Returns:
            Dict with keys:
                winner       : "A", "B", or "Tie" (in terms of the original
                               method_a / method_b labelling, after un-swapping)
                justification: One-to-two sentence explanation from the model.
                order        : "AB" if images were shown in natural order, "BA"
                               if they were swapped for this call.
        """
        # Randomly swap to mitigate positional bias
        swapped = random.random() < 0.5
        if swapped:
            left_img, right_img = image_b, image_a
            order = "BA"
        else:
            left_img, right_img = image_a, image_b
            order = "AB"

        prompt_text = self.PROMPT_TEMPLATE.format(prompt=prompt)

        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{_encode_image(left_img)}",
                            "detail": "high",
                        },
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{_encode_image(right_img)}",
                            "detail": "high",
                        },
                    },
                ],
            }],
            max_tokens=256,
        )

        answer = response.choices[0].message.content.strip()
        lines = answer.split("\n", 1)
        verdict_raw = lines[0].strip().upper()
        justification = lines[1].strip() if len(lines) > 1 else ""

        # Normalise verdict to A/B/Tie
        if verdict_raw.startswith("A"):
            verdict_presented = "A"
        elif verdict_raw.startswith("B"):
            verdict_presented = "B"
        else:
            verdict_presented = "Tie"

        # Un-swap: if order was BA, "A" in presentation = method B, etc.
        if verdict_presented == "Tie":
            winner = "Tie"
        elif swapped:
            winner = "B" if verdict_presented == "A" else "A"
        else:
            winner = verdict_presented

        return {
            "winner": winner,
            "justification": justification,
            "order": order,
        }

    def evaluate_dataset(
        self,
        method_a_dir: str,
        method_b_dir: str,
        prompts: list[str],
        output_file: str,
        n_pairs: int = 200,
        randomize_order: bool = True,
        resume: bool = True,
    ) -> dict:
        """Evaluate n_pairs of images from two methods.

        Results are written incrementally to output_file as JSONL so that
        interrupted runs can be resumed.

        Args:
            method_a_dir:    Directory of PNG images from method A (sorted).
            method_b_dir:    Directory of PNG images from method B (sorted).
            prompts:         List of prompts (one per image pair).
            output_file:     JSONL cache file; also stores final summary JSON.
            n_pairs:         Number of image pairs to evaluate.
            randomize_order: If True, randomly shuffle which pairs are selected.
            resume:          If True, skip pairs already in output_file.

        Returns:
            Dict with keys:
                win_rate_a   : Fraction of comparisons won by A (excl. ties)
                win_rate_b   : Fraction of comparisons won by B (excl. ties)
                tie_rate     : Fraction of tied comparisons
                n_evaluated  : Total comparisons made
                method_a     : Basename of method_a_dir
                method_b     : Basename of method_b_dir
                wilson_ci_a  : (lower, upper) 95% Wilson CI for win_rate_a
        """
        from src.evaluation.significance import wilson_score_interval

        paths_a = sorted(Path(method_a_dir).glob("*.png"))
        paths_b = sorted(Path(method_b_dir).glob("*.png"))
        n_available = min(len(paths_a), len(paths_b), len(prompts))
        indices = list(range(n_available))

        if randomize_order:
            random.shuffle(indices)

        indices = indices[:n_pairs]

        # Load previously cached results if resuming
        completed_indices: set[int] = set()
        cached_results: list[dict] = []

        jsonl_path = Path(output_file).with_suffix(".jsonl")
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

        if resume and jsonl_path.exists():
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    cached_results.append(rec)
                    completed_indices.add(rec["pair_index"])

        print(
            f"[vlm_judge] Resuming with {len(completed_indices)} already evaluated pairs."
            if completed_indices
            else "[vlm_judge] Starting fresh evaluation."
        )

        new_results: list[dict] = []
        with open(jsonl_path, "a", encoding="utf-8") as jsonl_f:
            for pair_idx in indices:
                if pair_idx in completed_indices:
                    continue

                img_a = PILImage.open(paths_a[pair_idx]).convert("RGB")
                img_b = PILImage.open(paths_b[pair_idx]).convert("RGB")
                prompt = prompts[pair_idx]

                comparison = self.compare_pair(img_a, img_b, prompt)
                record = {
                    "pair_index": pair_idx,
                    "prompt": prompt,
                    "image_a": paths_a[pair_idx].name,
                    "image_b": paths_b[pair_idx].name,
                    **comparison,
                }
                jsonl_f.write(json.dumps(record) + "\n")
                jsonl_f.flush()
                new_results.append(record)

                print(
                    f"  [{len(cached_results) + len(new_results)}/{len(indices)}] "
                    f"pair {pair_idx}: {comparison['winner']}"
                )

                time.sleep(self._sleep_s)

        all_results = cached_results + new_results
        wins_a = sum(1 for r in all_results if r["winner"] == "A")
        wins_b = sum(1 for r in all_results if r["winner"] == "B")
        ties = sum(1 for r in all_results if r["winner"] == "Tie")
        n_eval = len(all_results)

        total_decisive = wins_a + wins_b + ties
        win_rate_a = wins_a / total_decisive if total_decisive > 0 else 0.0
        win_rate_b = wins_b / total_decisive if total_decisive > 0 else 0.0
        tie_rate = ties / total_decisive if total_decisive > 0 else 0.0

        ci_a = wilson_score_interval(wins_a, n_eval)

        bt = fit_bradley_terry(wins_a, wins_b, ties)

        summary = {
            "win_rate_a": win_rate_a,
            "win_rate_b": win_rate_b,
            "tie_rate": tie_rate,
            "n_evaluated": n_eval,
            "wins_a": wins_a,
            "wins_b": wins_b,
            "ties": ties,
            "method_a": Path(method_a_dir).name,
            "method_b": Path(method_b_dir).name,
            "wilson_ci_a": list(ci_a),
            "bradley_terry": bt,
        }

        # Write summary JSON
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(
            f"[vlm_judge] Done. win_rate_a={win_rate_a:.3f}, "
            f"win_rate_b={win_rate_b:.3f}, tie_rate={tie_rate:.3f}"
        )
        return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GPT-4o pairwise image evaluation (VLM-as-Judge)."
    )
    parser.add_argument(
        "--method_a",
        required=True,
        help="Directory of PNG images for method A.",
    )
    parser.add_argument(
        "--method_b",
        required=True,
        help="Directory of PNG images for method B.",
    )
    parser.add_argument(
        "--prompts",
        required=True,
        help="JSON file with list of prompt strings.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSON file for summary; JSONL cache stored alongside.",
    )
    parser.add_argument(
        "--n_pairs",
        type=int,
        default=200,
        help="Number of image pairs to evaluate (default: 200).",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-2024-11-20",
        help="OpenAI model ID (default: gpt-4o-2024-11-20).",
    )
    parser.add_argument(
        "--rate_limit_rps",
        type=float,
        default=1.0,
        help="API calls per second (default: 1.0).",
    )
    parser.add_argument(
        "--no_resume",
        action="store_true",
        help="Do not resume from existing cache; start fresh.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for CLI usage."""
    args = _parse_args()

    with open(args.prompts, "r", encoding="utf-8") as f:
        prompts: list[str] = json.load(f)

    judge = VLMJudge(model=args.model, rate_limit_rps=args.rate_limit_rps)
    judge.evaluate_dataset(
        method_a_dir=args.method_a,
        method_b_dir=args.method_b,
        prompts=prompts,
        output_file=args.output,
        n_pairs=args.n_pairs,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
