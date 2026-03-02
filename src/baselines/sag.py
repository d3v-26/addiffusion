"""Self-Attention Guidance (SAG) baseline — B6.

SAG augments standard DDIM sampling with a self-attention-based perturbation
that improves synthesis quality without requiring additional text guidance.

Reference: Hong et al., "Improving Sample Quality of Diffusion Models Using
Self-Attention Guidance", ICCV 2023.

NFE: 20 scheduler steps (D-15: each step is 1 NFE regardless of CFG doublings).
Requires: ~10 GB VRAM (SD 1.5 fp16).

Run:
    uv run python -c "
    from src.baselines.sag import SAGBaseline
    b = SAGBaseline()
    b.generate(['a red bicycle'], [0], '/tmp/sag_out')
    "
"""

from __future__ import annotations

import time

import torch
from PIL import Image  # noqa: F401 — PIL.Image.Image used in BaselineResult

from src.baselines.base import BaseBaseline, BaselineResult, save_results


class SAGBaseline(BaseBaseline):
    """Self-Attention Guidance baseline at 20 DDIM steps.

    Args:
        device: PyTorch device string, e.g. ``"cuda"`` or ``"cpu"``.
    """

    name = "sag_20"
    nfe = 20

    def __init__(self, device: str = "cuda") -> None:
        from diffusers import StableDiffusionSAGPipeline

        self.device = device
        self.pipe = StableDiffusionSAGPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False,
        ).to(device)

    def generate(
        self,
        prompts: list[str],
        seeds: list[int],
        output_dir: str,
        sag_scale: float = 0.75,
        **kwargs,
    ) -> list[BaselineResult]:
        """Generate one image per (prompt, seed) pair using SAG.

        Args:
            prompts: Text prompts. ``len(prompts)`` must equal ``len(seeds)``.
            seeds: Per-image random seeds for reproducibility.
            output_dir: Directory for PNG outputs and ``results.jsonl``.
            sag_scale: Self-attention guidance scale (default 0.75, per paper).
            **kwargs: Ignored additional keyword arguments.

        Returns:
            List of :class:`~src.baselines.base.BaselineResult`, one per image.
        """
        results: list[BaselineResult] = []

        for prompt, seed in zip(prompts, seeds):
            gen = torch.Generator(self.device).manual_seed(seed)
            t0 = time.time()
            image = self.pipe(
                prompt,
                num_inference_steps=self.nfe,
                sag_scale=sag_scale,
                generator=gen,
            ).images[0]
            elapsed = time.time() - t0

            results.append(
                BaselineResult(
                    image=image,
                    prompt=prompt,
                    seed=seed,
                    nfe=self.nfe,
                    time_s=elapsed,
                    method_name=self.name,
                    metadata={"sag_scale": sag_scale},
                )
            )

        return save_results(results, output_dir)
