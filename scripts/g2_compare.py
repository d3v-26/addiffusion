"""G2 decision gate: compare agent metrics against DDIM-20.

Loads per-method metric JSON files produced by eval_metrics.slurm and runs
paired bootstrap significance tests to determine whether G2 passes.

G2 pass condition (phase2.md):
    >= 1 quality metric beats DDIM-20 at p < 0.05 AND mean agent NFE <= 30.

Usage:
    uv run python scripts/g2_compare.py \\
        --agent_metrics  outputs/evaluation/metrics/agent.json \\
        --ddim20_metrics outputs/evaluation/metrics/ddim20.json \\
        --agent_results  outputs/baselines/agent/results.jsonl

The script prints a pass/fail verdict and a per-metric summary table.

References:
    - plan.md Decision Gate G2
    - phase2.md G2 criteria table
    - src/evaluation/significance.py (paired_bootstrap_test)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.evaluation.significance import paired_bootstrap_test


# G2 thresholds
NFE_BUDGET = 30       # Mean agent NFE must be <= this
P_THRESHOLD = 0.05    # Significance threshold


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def load_per_image_metrics(metrics_json: dict, metric_key: str) -> np.ndarray | None:
    """Extract per-image metric values from a compute_metrics.py output JSON.

    compute_metrics.py stores per-image scores under 'per_image' or at the
    top level as a list.  Falls back to None if not available.
    """
    per_image = metrics_json.get("per_image", {})
    if metric_key in per_image:
        return np.array(per_image[metric_key], dtype=float)
    # Some metrics store values directly as a list at top level
    if metric_key in metrics_json and isinstance(metrics_json[metric_key], list):
        return np.array(metrics_json[metric_key], dtype=float)
    return None


def load_mean_nfe(results_jsonl: str) -> float:
    """Compute mean NFE from the agent's results.jsonl."""
    nfes = []
    with open(results_jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            nfes.append(record["nfe"])
    return float(np.mean(nfes))


def summarise_actions(results_jsonl: str) -> dict:
    """Return action distribution from results.jsonl."""
    counts = {"continue": 0, "stop": 0, "refine": 0}
    total = 0
    with open(results_jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            for a in record.get("action_sequence", []):
                counts[a] = counts.get(a, 0) + 1
                total += 1
    if total == 0:
        return counts
    return {k: v / total for k, v in counts.items()}


def main():
    parser = argparse.ArgumentParser(description="G2 decision gate comparison.")
    parser.add_argument("--agent_metrics",  required=True, help="outputs/evaluation/metrics/agent.json")
    parser.add_argument("--ddim20_metrics", required=True, help="outputs/evaluation/metrics/ddim20.json")
    parser.add_argument("--agent_results",  required=True, help="outputs/baselines/agent/results.jsonl")
    parser.add_argument("--n_resamples", type=int, default=10_000, help="Bootstrap resamples (default: 10000)")
    args = parser.parse_args()

    agent_m  = load_json(args.agent_metrics)
    ddim20_m = load_json(args.ddim20_metrics)

    # ------------------------------------------------------------------ #
    # NFE check                                                             #
    # ------------------------------------------------------------------ #
    mean_nfe = load_mean_nfe(args.agent_results)
    nfe_pass = mean_nfe <= NFE_BUDGET
    action_dist = summarise_actions(args.agent_results)

    print("=" * 65)
    print("  AdDiffusion — G2 Decision Gate")
    print("=" * 65)
    print(f"\n  Mean agent NFE : {mean_nfe:.2f}  (threshold: <= {NFE_BUDGET})  {'PASS' if nfe_pass else 'FAIL'}")
    print(f"  Action dist    : continue={action_dist.get('continue',0):.2%}  "
          f"stop={action_dist.get('stop',0):.2%}  "
          f"refine={action_dist.get('refine',0):.2%}")

    # ------------------------------------------------------------------ #
    # Per-metric quality comparisons                                        #
    # ------------------------------------------------------------------ #
    # Metrics where higher is better
    higher_better = {
        "clip_score":   "CLIP Score   (agent >= ddim20)",
        "imagereward":  "ImageReward  (agent >= ddim20)",
        "hpsv2":        "HPS v2       (agent >= ddim20)",
        "aesthetic":    "Aesthetic    (agent >= ddim20)",
    }
    # FID: lower is better → test agent < ddim20
    lower_better = {
        "fid": "FID          (agent <= ddim20)",
    }

    print(f"\n{'Metric':<20} {'Agent':>8} {'DDIM-20':>8} {'Δ':>8} {'p-value':>10} {'Result':>8}")
    print("-" * 65)

    quality_pass = False
    quality_pass_reasons = []

    def run_test(agent_vals, ddim20_vals, metric_label, alternative, higher_is_better):
        nonlocal quality_pass
        if agent_vals is None or ddim20_vals is None:
            print(f"  {metric_label:<20} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>10} {'SKIP':>8}")
            return

        # Align lengths
        n = min(len(agent_vals), len(ddim20_vals))
        if n < len(agent_vals) or n < len(ddim20_vals):
            print(f"  [WARNING] Length mismatch for {metric_label}: agent={len(agent_vals)}, ddim20={len(ddim20_vals)}. Truncating to {n}.")
        agent_vals = agent_vals[:n]
        ddim20_vals = ddim20_vals[:n]

        result = paired_bootstrap_test(
            agent_vals, ddim20_vals,
            n_resamples=args.n_resamples,
            alternative=alternative,
        )

        mean_agent  = float(np.mean(agent_vals))
        mean_ddim20 = float(np.mean(ddim20_vals))
        delta = mean_agent - mean_ddim20
        p = result["p_value"]

        if higher_is_better:
            passed = (mean_agent >= mean_ddim20) and (p < P_THRESHOLD)
        else:
            passed = (mean_agent <= mean_ddim20) and (p < P_THRESHOLD)

        verdict = "PASS" if passed else ("ns" if p >= P_THRESHOLD else "FAIL")
        print(f"  {metric_label:<20} {mean_agent:>8.4f} {mean_ddim20:>8.4f} {delta:>+8.4f} {p:>10.4f} {verdict:>8}")

        if passed:
            quality_pass = True
            quality_pass_reasons.append(metric_label.strip())

    for key, label in higher_better.items():
        run_test(
            load_per_image_metrics(agent_m,  key),
            load_per_image_metrics(ddim20_m, key),
            label, alternative="greater", higher_is_better=True,
        )

    for key, label in lower_better.items():
        run_test(
            load_per_image_metrics(agent_m,  key),
            load_per_image_metrics(ddim20_m, key),
            label, alternative="less", higher_is_better=False,
        )

    # ------------------------------------------------------------------ #
    # G2 verdict                                                            #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 65)
    g2_pass = nfe_pass and quality_pass
    if g2_pass:
        print(f"  G2: PASS")
        print(f"  Agent beats DDIM-20 on: {', '.join(quality_pass_reasons)}")
        print(f"  Mean NFE {mean_nfe:.1f} <= {NFE_BUDGET}.  Proceed to Phase 3 full evaluation.")
    else:
        if not nfe_pass:
            print(f"  G2: FAIL  — mean NFE {mean_nfe:.1f} exceeds budget {NFE_BUDGET}.")
            print(f"  Action: check stop action frequency; increase c_nfe penalty.")
        if not quality_pass:
            print(f"  G2: FAIL  — no quality metric beats DDIM-20 at p < {P_THRESHOLD}.")
            print(f"  Action: try quality_heavy reward profile or extend training to 1K iters.")
    print("=" * 65)

    sys.exit(0 if g2_pass else 1)


if __name__ == "__main__":
    main()
