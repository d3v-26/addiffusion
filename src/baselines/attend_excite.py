"""Attend-and-Excite baseline — B6.

Attend-and-Excite enforces that each subject token in the prompt attends to
a distinct spatial region of the image, improving compositional text-to-image
generation.

Reference: Chefer et al., "Attend-and-Excite: Attention-Based Semantic
Guidance for Text-to-Image Diffusion Models", SIGGRAPH 2023.

NFE: 50 scheduler steps (standard quality setting from the paper).
Requires: ~10 GB VRAM (SD 1.5 fp16).

Run:
    uv run python -c "
    from src.baselines.attend_excite import AttendExciteBaseline
    b = AttendExciteBaseline()
    b.generate(['a cat and a dog'], [0], '/tmp/ae_out')
    "
"""

from __future__ import annotations

import time

import torch
from PIL import Image  # noqa: F401 — PIL.Image.Image used in BaselineResult

from src.baselines.base import BaseBaseline, BaselineResult, save_results


class AttendExciteBaseline(BaseBaseline):
    """Attend-and-Excite baseline at 50 DDIM steps.

    Token indices for excitation are extracted automatically from each prompt
    using the pipeline tokenizer: all non-special tokens (i.e. everything
    except BOS at position 0 and EOS at the final position) are passed to the
    pipeline so that all content tokens are encouraged to attend to distinct
    regions.

    Args:
        device: PyTorch device string, e.g. ``"cuda"`` or ``"cpu"``.
    """

    name = "attend_excite_50"
    nfe = 50

    def __init__(self, device: str = "cuda") -> None:
        from diffusers import StableDiffusionAttendAndExcitePipeline

        self.device = device
        self.pipe = StableDiffusionAttendAndExcitePipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False,
        ).to(device)

    def _get_token_indices(self, prompt: str) -> list[int]:
        """Return indices of content tokens, excluding BOS/EOS.

        The CLIP tokenizer adds a BOS token at position 0 and an EOS token at
        the final position.  Attend-and-Excite operates on content tokens only,
        so we return positions ``1 .. len-2`` (inclusive).

        Args:
            prompt: A single text prompt string.

        Returns:
            List of integer token positions for all content tokens.
        """
        token_ids = self.pipe.tokenizer(prompt).input_ids
        # Position 0 is BOS; position len-1 is EOS.  Content is in between.
        return list(range(1, len(token_ids) - 1))

    def generate(
        self,
        prompts: list[str],
        seeds: list[int],
        output_dir: str,
        max_iter_to_alter: int = 25,
        **kwargs,
    ) -> list[BaselineResult]:
        """Generate one image per (prompt, seed) pair using Attend-and-Excite.

        Args:
            prompts: Text prompts. ``len(prompts)`` must equal ``len(seeds)``.
            seeds: Per-image random seeds for reproducibility.
            output_dir: Directory for PNG outputs and ``results.jsonl``.
            max_iter_to_alter: Number of denoising steps during which
                attention excitation updates are applied (default 25, per
                paper; excitation is frozen for the remaining steps).
            **kwargs: Ignored additional keyword arguments.

        Returns:
            List of :class:`~src.baselines.base.BaselineResult`, one per image.
        """
        results: list[BaselineResult] = []

        for prompt, seed in zip(prompts, seeds):
            gen = torch.Generator(self.device).manual_seed(seed)
            token_indices = self._get_token_indices(prompt)
            t0 = time.time()
            image = self.pipe(
                prompt=prompt,
                token_indices=token_indices,
                num_inference_steps=self.nfe,
                generator=gen,
                max_iter_to_alter=max_iter_to_alter,
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
                    metadata={
                        "token_indices": token_indices,
                        "max_iter_to_alter": max_iter_to_alter,
                    },
                )
            )

        return save_results(results, output_dir)
