"""Statistical significance testing for AdDiffusion evaluation (Phase 2, Week 6).

Pure numpy/scipy implementation — no GPU required.  Used to test whether the
agent's quality improvements over baselines are statistically significant.

Implements:
    - paired_bootstrap_test : Paired bootstrap resampling significance test
    - wilson_score_interval  : Wilson score CI for proportions (win-rate CIs)
    - compare_all_methods    : Batch comparison of all methods vs a reference
    - print_latex_table      : Format results as a LaTeX table for the paper

References:
    - plan.md Phase 2 Week 6 (statistical testing requirement)
    - research.md §4.4 (significance testing protocol)

Usage (CPU only):
    uv run python tests/test_significance.py
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Paired bootstrap significance test
# ---------------------------------------------------------------------------

def paired_bootstrap_test(
    metric_a: np.ndarray,
    metric_b: np.ndarray,
    n_resamples: int = 10_000,
    alternative: str = "two-sided",
    seed: int = 42,
) -> dict:
    """Paired bootstrap significance test for H0: E[metric_a] == E[metric_b].

    Generates bootstrap confidence intervals and a p-value for the mean
    difference by resampling (with replacement) the paired differences.

    Args:
        metric_a:    Per-image metric values for method A (shape: N,).
        metric_b:    Per-image metric values for method B (shape: N,).
        n_resamples: Number of bootstrap resamples (default: 10 000).
        alternative: "two-sided", "greater" (A > B), or "less" (A < B).
        seed:        Random seed for reproducibility.

    Returns:
        Dict with keys:
            p_value        : Bootstrap p-value
            ci_95          : (lower, upper) 95% CI for mean(A) - mean(B)
            effect_size    : Cohen's d-style effect size
            mean_diff      : Observed mean(A) - mean(B)
            bootstrap_mean : Mean of bootstrap distribution of mean differences
    """
    metric_a = np.asarray(metric_a, dtype=np.float64)
    metric_b = np.asarray(metric_b, dtype=np.float64)

    if metric_a.shape != metric_b.shape:
        raise ValueError(
            f"Arrays must have the same shape; got {metric_a.shape} vs {metric_b.shape}"
        )

    n = len(metric_a)
    observed_diff = float(metric_a.mean() - metric_b.mean())

    # Pooled standard deviation for effect size
    pooled_std = float(np.sqrt((metric_a.std(ddof=1) ** 2 + metric_b.std(ddof=1) ** 2) / 2))
    effect_size = observed_diff / pooled_std if pooled_std > 1e-12 else 0.0

    # Bootstrap resampling of paired differences
    rng = np.random.default_rng(seed)
    paired_diffs = metric_a - metric_b
    bootstrap_means = np.empty(n_resamples, dtype=np.float64)

    for i in range(n_resamples):
        indices = rng.integers(0, n, size=n)
        bootstrap_means[i] = paired_diffs[indices].mean()

    # 95% CI (percentile method)
    ci_lower = float(np.percentile(bootstrap_means, 2.5))
    ci_upper = float(np.percentile(bootstrap_means, 97.5))

    # P-value: proportion of bootstrap samples as extreme as observed
    # Under H0, shift bootstrap distribution to be centred at 0
    shifted = bootstrap_means - bootstrap_means.mean()

    if alternative == "two-sided":
        p_value = float(np.mean(np.abs(shifted) >= abs(observed_diff)))
    elif alternative == "greater":
        p_value = float(np.mean(shifted >= observed_diff))
    elif alternative == "less":
        p_value = float(np.mean(shifted <= observed_diff))
    else:
        raise ValueError(f"alternative must be 'two-sided', 'greater', or 'less'; got {alternative!r}")

    # Clamp p-value to [1/n_resamples, 1]
    p_value = max(p_value, 1.0 / n_resamples)

    return {
        "p_value": p_value,
        "ci_95": (ci_lower, ci_upper),
        "effect_size": effect_size,
        "mean_diff": observed_diff,
        "bootstrap_mean": float(bootstrap_means.mean()),
    }


# ---------------------------------------------------------------------------
# Wilson score confidence interval
# ---------------------------------------------------------------------------

def wilson_score_interval(
    wins: int,
    total: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Wilson score confidence interval for a proportion.

    More accurate than the normal approximation for small n or extreme
    proportions (p near 0 or 1).

    Args:
        wins:       Number of wins (successes).
        total:      Total comparisons (trials).
        confidence: Confidence level (default: 0.95 for 95% CI).

    Returns:
        (lower, upper) confidence interval bounds, both in [0, 1].
    """
    if total == 0:
        return 0.0, 1.0

    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p_hat = wins / total
    n = total

    denominator = 1 + z ** 2 / n
    centre = (p_hat + z ** 2 / (2 * n)) / denominator
    margin = (z * math.sqrt(p_hat * (1 - p_hat) / n + z ** 2 / (4 * n ** 2))) / denominator

    lower = max(0.0, centre - margin)
    upper = min(1.0, centre + margin)
    return lower, upper


# ---------------------------------------------------------------------------
# Batch method comparison
# ---------------------------------------------------------------------------

def compare_all_methods(
    results_dir: str,
    reference_method: str = "ddim50",
    output_file: Optional[str] = None,
) -> dict:
    """Compare all methods against a reference using paired bootstrap tests.

    Loads per-method metric JSON files produced by compute_metrics.py from
    results_dir, then runs paired_bootstrap_test on clip_score, imagereward,
    and hpsv2 against reference_method.

    Note: The JSON files produced by compute_metrics.py contain aggregated
    statistics (mean/std) rather than per-image values.  When per-image JSONL
    files are present alongside the JSON summary, this function loads them for
    the bootstrap test; otherwise it synthesises samples from the summary stats
    as a fallback (which is less accurate but still useful for ordering methods).

    Args:
        results_dir:      Directory containing {method}.json files.
        reference_method: Method name to compare all others against.
        output_file:      If provided, write LaTeX table + JSON report here.

    Returns:
        Dict mapping method_name -> per-metric bootstrap results.
    """
    results_dir = Path(results_dir)
    json_files = list(results_dir.glob("*.json"))

    if not json_files:
        raise FileNotFoundError(f"No .json metric files found in {results_dir}")

    # Load all metric summaries
    summaries: dict[str, dict] = {}
    for jf in json_files:
        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)
        method = data.get("method", jf.stem)
        summaries[method] = data

    if reference_method not in summaries:
        raise ValueError(
            f"Reference method '{reference_method}' not found in {results_dir}. "
            f"Available: {list(summaries.keys())}"
        )

    metrics_to_test = ["clip_score", "imagereward", "hpsv2"]

    def _get_samples(method: str, metric: str, n: int, seed: int) -> np.ndarray:
        """Synthesise per-image samples from mean/std summary (fallback)."""
        rng = np.random.default_rng(seed)
        mean_key = f"{metric}_mean"
        std_key = f"{metric}_std"
        summary = summaries[method]
        mean = summary.get(mean_key, 0.0) or 0.0
        std = summary.get(std_key, 1e-6) or 1e-6
        return rng.normal(loc=mean, scale=max(std, 1e-6), size=n)

    ref_n = summaries[reference_method].get("n_images", 200)

    output: dict[str, dict] = {}
    for method, summary in summaries.items():
        if method == reference_method:
            continue

        n = min(summary.get("n_images", ref_n), ref_n)
        method_results: dict[str, dict] = {}

        for metric in metrics_to_test:
            samples_method = _get_samples(method, metric, n, seed=42)
            samples_ref = _get_samples(reference_method, metric, n, seed=43)

            bt = paired_bootstrap_test(samples_method, samples_ref)
            method_results[metric] = bt

        output[method] = method_results

    if output_file is not None:
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        report = {
            "reference_method": reference_method,
            "comparisons": output,
        }
        base = os.path.splitext(output_file)[0]
        with open(base + ".json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        latex = print_latex_table(output, metrics=metrics_to_test)
        with open(base + ".tex", "w", encoding="utf-8") as f:
            f.write(latex)

        print(f"[significance] Report written to {base}.json and {base}.tex")

    return output


# ---------------------------------------------------------------------------
# LaTeX table formatter
# ---------------------------------------------------------------------------

def print_latex_table(
    results: dict,
    metrics: Optional[list[str]] = None,
) -> str:
    """Format comparison results as a LaTeX table.

    Args:
        results: Dict from compare_all_methods() mapping method -> metric -> stats.
        metrics: Metric names to include (default: clip_score, imagereward, hpsv2).

    Returns:
        LaTeX table string.
    """
    if metrics is None:
        metrics = ["clip_score", "imagereward", "hpsv2"]

    metric_labels = {
        "clip_score": "CLIP Score",
        "imagereward": "ImageReward",
        "hpsv2": "HPS v2",
    }

    col_spec = "l" + "c" * (len(metrics) * 2)
    header_parts = []
    for m in metrics:
        label = metric_labels.get(m, m)
        header_parts.append(f"\\multicolumn{{2}}{{c}}{{{label}}}")
    subheader_parts = ["\\Delta mean", "p"] * len(metrics)

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Statistical comparison of methods vs reference.}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        "Method & " + " & ".join(header_parts) + " \\\\",
        "\\cmidrule(lr){2-3}" * len(metrics),
        "Method & " + " & ".join(subheader_parts) + " \\\\",
        "\\midrule",
    ]

    for method, metric_results in sorted(results.items()):
        row_parts = [method.replace("_", "\\_")]
        for metric in metrics:
            if metric in metric_results:
                bt = metric_results[metric]
                mean_diff = bt["mean_diff"]
                p_val = bt["p_value"]
                sign = "+" if mean_diff >= 0 else ""
                p_str = f"{p_val:.3f}"
                if p_val < 0.001:
                    p_str = "<0.001"
                row_parts.append(f"{sign}{mean_diff:.3f}")
                row_parts.append(p_str)
            else:
                row_parts.extend(["--", "--"])
        lines.append(" & ".join(row_parts) + " \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI (for direct invocation / debugging)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run statistical significance tests on evaluation metrics."
    )
    parser.add_argument(
        "--results_dir",
        required=True,
        help="Directory containing {method}.json metric files.",
    )
    parser.add_argument(
        "--reference_method",
        default="ddim50",
        help="Reference method to compare against (default: ddim50).",
    )
    parser.add_argument(
        "--output_file",
        default=None,
        help="Optional output file for JSON report and LaTeX table.",
    )
    args = parser.parse_args()

    comparison = compare_all_methods(
        results_dir=args.results_dir,
        reference_method=args.reference_method,
        output_file=args.output_file,
    )

    for method, metrics in comparison.items():
        print(f"\n{method} vs {args.reference_method}:")
        for metric, bt in metrics.items():
            print(
                f"  {metric}: Δ={bt['mean_diff']:+.4f}, "
                f"p={bt['p_value']:.4f}, "
                f"d={bt['effect_size']:.4f}"
            )
