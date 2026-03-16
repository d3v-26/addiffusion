"""Run one or more baselines on a prompt set (Phase 2, Week 6).

Generates images for each specified baseline and writes outputs to
outputs/baselines/{method_name}/.  Each baseline saves:
    - PNG files: {seed}_{idx:04d}.png
    - results.jsonl: one JSON record per image
    - metadata.json: method config and timestamp

Usage:
    uv run python scripts/run_baselines.py \\
        --baselines ddim20,ddim50 \\
        --prompts_file data/drawbench/prompts.json \\
        --output_dir outputs/baselines/ \\
        --seeds 42

Supported baselines: ddim20, ddim50, dpm20, dpm50, euler20, unipc20, pndm20,
                     sag, attend_excite, lcm, sdxl_turbo, random_search, oracle

References:
    - plan.md Phase 2 Week 6 (baseline generation pipeline)
    - src/baselines/base.py (BaseBaseline interface)
    - experiment.md §3 (execution commands)
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
import time
import traceback


# ---------------------------------------------------------------------------
# Baseline registry
# ---------------------------------------------------------------------------

BASELINE_REGISTRY: dict[str, str] = {
    "ddim20":         "src.baselines.fixed_step.DDIM20",
    "ddim50":         "src.baselines.fixed_step.DDIM50",
    "dpm20":          "src.baselines.fixed_step.DPMSolver20",
    "dpm50":          "src.baselines.fixed_step.DPMSolver50",
    "euler20":        "src.baselines.fixed_step.Euler20",
    "unipc20":        "src.baselines.fixed_step.UniPC20",
    "pndm20":         "src.baselines.fixed_step.PNDM20",
    "sag":            "src.baselines.sag.SAGBaseline",
    "attend_excite":  "src.baselines.attend_excite.AttendExciteBaseline",
    "lcm":            "src.baselines.lcm.LCMBaseline",
    "sdxl_turbo":     "src.baselines.sdxl_turbo.SDXLTurboBaseline",
    "random_search":  "src.baselines.random_search.RandomSearchBaseline",
    "oracle":         "src.baselines.oracle.OracleStopBaseline",
}


# ---------------------------------------------------------------------------
# Dynamic class loader
# ---------------------------------------------------------------------------

def _load_class(dotted_path: str):
    """Load a class from a dotted module path string.

    Example:
        _load_class("src.baselines.fixed_step.DDIM20")
        => class DDIM20 from src.baselines.fixed_step

    Args:
        dotted_path: Fully qualified class path.

    Returns:
        The class object.

    Raises:
        ImportError: If the module cannot be imported.
        AttributeError: If the class does not exist in the module.
    """
    module_path, class_name = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


# ---------------------------------------------------------------------------
# Seeds helper
# ---------------------------------------------------------------------------

def _parse_seeds(seeds_str: str, n_prompts: int) -> list[int]:
    """Parse the --seeds argument into a list of n_prompts integers.

    If a single integer is given, the same seed is used for all prompts.
    If a comma-separated list is given, it is used as-is (truncated or
    repeated to match n_prompts).

    Args:
        seeds_str: String from --seeds argument (e.g. "42" or "42,43,44").
        n_prompts: Number of prompts (= desired list length).

    Returns:
        List of integer seeds, length == n_prompts.
    """
    parts = [int(x.strip()) for x in seeds_str.split(",")]
    if len(parts) == 1:
        return [parts[0]] * n_prompts
    # Cycle or truncate to match n_prompts
    if len(parts) < n_prompts:
        repeated = (parts * ((n_prompts // len(parts)) + 1))[:n_prompts]
        return repeated
    return parts[:n_prompts]


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_baseline(
    name: str,
    prompts: list[str],
    seeds: list[int],
    output_dir: str,
    device: str,
) -> None:
    """Instantiate and run a single baseline.

    Args:
        name:       Baseline name key (must be in BASELINE_REGISTRY).
        prompts:    List of text prompts.
        seeds:      List of per-image random seeds (same length as prompts).
        output_dir: Root output directory; images go to {output_dir}/{name}/.
        device:     Torch device string (e.g. "cuda", "cpu").
    """
    if name not in BASELINE_REGISTRY:
        raise ValueError(
            f"Unknown baseline '{name}'. "
            f"Available: {sorted(BASELINE_REGISTRY.keys())}"
        )

    class_path = BASELINE_REGISTRY[name]
    print(f"[run_baselines] Loading {class_path} ...")
    BaselineClass = _load_class(class_path)

    print(f"[run_baselines] Instantiating {name} on device={device} ...")
    baseline = BaselineClass(device=device)

    method_output_dir = os.path.join(output_dir, name)
    os.makedirs(method_output_dir, exist_ok=True)

    print(
        f"[run_baselines] Running {name} on {len(prompts)} prompts -> {method_output_dir}"
    )
    t0 = time.time()
    results = baseline.generate(prompts=prompts, seeds=seeds, output_dir=method_output_dir)
    elapsed = time.time() - t0

    print(
        f"[run_baselines] {name}: {len(results)} images in {elapsed:.1f}s "
        f"({elapsed/len(results):.2f}s/image)"
    )


def main() -> None:
    """Entry point: parse args, load prompts, run each baseline in sequence."""
    parser = argparse.ArgumentParser(
        description="Run baseline image generators for AdDiffusion evaluation."
    )
    parser.add_argument(
        "--baselines",
        required=True,
        help="Comma-separated list of baseline names (e.g. ddim20,ddim50).",
    )
    parser.add_argument(
        "--prompts_file",
        required=True,
        help="JSON file containing a list of prompt strings.",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/baselines/",
        help="Root output directory (default: outputs/baselines/).",
    )
    parser.add_argument(
        "--seeds",
        default="42",
        help=(
            "Random seed(s) for generation.  Pass a single integer to use the "
            "same seed for all prompts, or a comma-separated list for per-image "
            "seeds (default: 42)."
        ),
    )
    parser.add_argument(
        "--device",
        default=None,
        help=(
            "Torch device (default: 'cuda' if available, else 'cpu').  "
            "Override with e.g. --device cpu for testing."
        ),
    )
    args = parser.parse_args()

    # Resolve device
    if args.device is None:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    else:
        device = args.device

    # Load prompts
    with open(args.prompts_file, "r", encoding="utf-8") as f:
        prompts: list[str] = json.load(f)

    print(f"[run_baselines] Loaded {len(prompts)} prompts from {args.prompts_file}")

    # Parse baseline names
    baseline_names = [b.strip() for b in args.baselines.split(",") if b.strip()]
    print(f"[run_baselines] Baselines to run: {baseline_names}")

    # Parse seeds
    seeds = _parse_seeds(args.seeds, len(prompts))

    os.makedirs(args.output_dir, exist_ok=True)

    # Run each baseline, catching errors to avoid crashing the whole job
    errors: dict[str, str] = {}
    for name in baseline_names:
        print(f"\n{'='*60}")
        print(f"  Baseline: {name}")
        print(f"{'='*60}")
        try:
            run_baseline(
                name=name,
                prompts=prompts,
                seeds=seeds,
                output_dir=args.output_dir,
                device=device,
            )
        except Exception as exc:
            tb = traceback.format_exc()
            print(f"[run_baselines] ERROR in baseline '{name}': {exc}", file=sys.stderr)
            print(tb, file=sys.stderr)
            errors[name] = str(exc)

    # Final summary
    print(f"\n{'='*60}")
    print("  Summary")
    print(f"{'='*60}")
    succeeded = [n for n in baseline_names if n not in errors]
    print(f"  Succeeded : {succeeded}")
    if errors:
        print(f"  Failed    : {list(errors.keys())}")
        for name, msg in errors.items():
            print(f"    {name}: {msg}")
        sys.exit(1)
    else:
        print("  All baselines completed successfully.")


if __name__ == "__main__":
    main()
