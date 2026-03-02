"""SDXL-Turbo baseline — B5.

SDXL-Turbo is an adversarially distilled variant of Stable Diffusion XL that
produces 512x512 images in a single or very small number of denoising steps.
It uses Adversarial Diffusion Distillation (ADD) which eliminates the need for
classifier-free guidance.

Reference: Sauer et al., "Adversarial Diffusion Distillation", arXiv 2311.17042.

Model: ``stabilityai/sdxl-turbo`` (HuggingFace Hub, fp16 variant).

NFE: 4 scheduler steps.
guidance_scale: 0.0 (mandatory — the model was trained without CFG; any non-zero
value produces degraded outputs per the official Stability AI documentation).
Output resolution: 512x512 (default for SDXL-Turbo).

Requires: ~24 GB VRAM (SDXL fp16).  Use an A100-80GB node.

Run:
    uv run python -c "
    from src.baselines.sdxl_turbo import SDXLTurboBaseline
    b = SDXLTurboBaseline()
    b.generate(['a red bicycle'], [0], '/tmp/sdxl_turbo_out')
    "
"""

from __future__ import annotations

import time

import torch
from PIL import Image  # noqa: F401 — PIL.Image.Image used in BaselineResult

from src.baselines.base import BaseBaseline, BaselineResult, save_results


class SDXLTurboBaseline(BaseBaseline):
    """SDXL-Turbo baseline: adversarially distilled SDXL at 4 steps.

    Classifier-free guidance is disabled (``guidance_scale=0.0``) as required
    by the adversarial distillation training objective.

    Args:
        device: PyTorch device string, e.g. ``"cuda"`` or ``"cpu"``.

    Note:
        Requires approximately 24 GB VRAM.  Schedule on an A100-80GB node when
        running on the HPC cluster.
    """

    name = "sdxl_turbo_4"
    nfe = 4

    def __init__(self, device: str = "cuda") -> None:
        from diffusers import AutoPipelineForText2Image

        self.device = device
        self.pipe = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sdxl-turbo",
            torch_dtype=torch.float16,
            variant="fp16",
        ).to(device)

    def generate(
        self,
        prompts: list[str],
        seeds: list[int],
        output_dir: str,
        **kwargs,
    ) -> list[BaselineResult]:
        """Generate one image per (prompt, seed) pair using SDXL-Turbo.

        Args:
            prompts: Text prompts. ``len(prompts)`` must equal ``len(seeds)``.
            seeds: Per-image random seeds for reproducibility.
            output_dir: Directory for PNG outputs and ``results.jsonl``.
            **kwargs: Ignored additional keyword arguments.

        Returns:
            List of :class:`~src.baselines.base.BaselineResult`, one per image.

        Note:
            ``guidance_scale`` is fixed at ``0.0`` (mandatory for SDXL-Turbo).
            Passing a non-zero value via ``kwargs`` is silently ignored.
            Default output resolution is 512x512.
        """
        results: list[BaselineResult] = []

        for prompt, seed in zip(prompts, seeds):
            gen = torch.Generator(self.device).manual_seed(seed)
            t0 = time.time()
            image = self.pipe(
                prompt=prompt,
                num_inference_steps=self.nfe,
                guidance_scale=0.0,  # Mandatory: adversarial distillation, no CFG
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
                    metadata={},
                )
            )

        return save_results(results, output_dir)
