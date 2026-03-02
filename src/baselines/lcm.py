"""LCM-LoRA baseline with Stable Diffusion 1.5 — B5.

Latent Consistency Models (LCM) via LoRA adapter enable high-quality image
generation in as few as 4 denoising steps by distilling a consistency model
into the base SD 1.5 weights using a lightweight LoRA patch.

Reference: Luo et al., "Latent Consistency Models: Synthesizing High-Resolution
Images with Few-Step Inference", arXiv 2310.04378.

Adapter: ``latent-consistency/lcm-lora-sdv1-5`` (HuggingFace Hub).

NFE: 4 scheduler steps.
guidance_scale: 0 (CFG fully disabled — LCM was trained without classifier-free
guidance; using guidance_scale > 0 degrades output quality).
Requires: ~10 GB VRAM (SD 1.5 fp16).

Run:
    uv run python -c "
    from src.baselines.lcm import LCMBaseline
    b = LCMBaseline()
    b.generate(['a red bicycle'], [0], '/tmp/lcm_out')
    "
"""

from __future__ import annotations

import time

import torch
from PIL import Image  # noqa: F401 — PIL.Image.Image used in BaselineResult

from src.baselines.base import BaseBaseline, BaselineResult, save_results


class LCMBaseline(BaseBaseline):
    """LCM-LoRA baseline: SD 1.5 + LCM-LoRA at 4 denoising steps.

    The LoRA weights are fused into the base model at init time so that
    inference has no additional overhead compared to standard SD 1.5.

    Args:
        device: PyTorch device string, e.g. ``"cuda"`` or ``"cpu"``.
    """

    name = "lcm_4"
    nfe = 4

    def __init__(self, device: str = "cuda") -> None:
        from diffusers import LCMScheduler, StableDiffusionPipeline

        self.device = device

        pipe = StableDiffusionPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False,
        ).to(device)

        # Load and fuse the LCM-LoRA adapter into the base weights.
        # fuse_lora() merges the adapter matrices so no extra compute at runtime.
        pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
        pipe.fuse_lora()

        # Replace the scheduler with the LCM-compatible variant.
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

        self.pipe = pipe

    def generate(
        self,
        prompts: list[str],
        seeds: list[int],
        output_dir: str,
        **kwargs,
    ) -> list[BaselineResult]:
        """Generate one image per (prompt, seed) pair using LCM-LoRA.

        Args:
            prompts: Text prompts. ``len(prompts)`` must equal ``len(seeds)``.
            seeds: Per-image random seeds for reproducibility.
            output_dir: Directory for PNG outputs and ``results.jsonl``.
            **kwargs: Ignored additional keyword arguments.

        Returns:
            List of :class:`~src.baselines.base.BaselineResult`, one per image.

        Note:
            ``guidance_scale`` is fixed at ``0`` (CFG disabled).  Passing a
            non-zero value via ``kwargs`` is silently ignored to prevent
            accidental quality degradation.
        """
        results: list[BaselineResult] = []

        for prompt, seed in zip(prompts, seeds):
            gen = torch.Generator(self.device).manual_seed(seed)
            t0 = time.time()
            image = self.pipe(
                prompt,
                num_inference_steps=self.nfe,
                guidance_scale=0,  # LCM: CFG disabled (official recommendation)
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
