"""Oracle-Stop baseline: CLIP-guided optimal early exit (Phase 2, Week 5).

Baseline B4 from plan.md Phase 2 Week 5.  The oracle runs ALL denoising steps
via AdaptiveDiffusionPipeline, decodes and scores the one-step clean prediction
z0_pred at every step, and returns the image from the step with the peak CLIP
score.  This represents an upper bound on the agent's stop policy.

Because the oracle has access to future information (final CLIP score), it
cannot be deployed at inference time.  Its role is to establish the quality
ceiling for early stopping and measure how far the trained agent is from
theoretically optimal.

References:
    - plan.md Phase 2 Week 5 (baseline B4 — oracle stop)
    - research.md §4.2 (oracle upper bound, hypothesis H1)
    - discovery.md D-15 (NFE = scheduler steps, not network evals)
    - discovery.md D-14 (latent space; decode only for scoring)
    - discovery.md D-16 (CLIP uninformative at early timesteps — oracle sees this)
"""

from __future__ import annotations

import time
from typing import Optional

import torch
import torch.nn.functional as F
from PIL import Image

from src.baselines.base import BaseBaseline, BaselineResult, load_sd15_pipeline, save_results


class OracleStopBaseline(BaseBaseline):
    """Oracle early-stopping baseline scored by CLIP similarity (baseline B4).

    Runs full denoising, decodes z0_pred at every step, and returns the image
    at the step index where CLIP similarity peaks.  Reports the NFE at that
    peak rather than max_steps.

    Args:
        max_steps: Total scheduler steps to unroll (default 50, matching
                   AdDiffusion episode length).
        device:    Torch device string (default "cuda").

    References:
        - plan.md Phase 2 Week 5 (baseline B4)
        - research.md §4.2 (oracle upper bound)
    """

    name = "oracle_stop"
    nfe = -1  # Dynamic: reports NFE at CLIP peak per image

    def __init__(self, max_steps: int = 50, device: str = "cuda") -> None:
        self.max_steps = max_steps
        self.device = device
        self._pipeline = None
        self._clip_model = None
        self._preprocess = None
        self._tokenizer = None

    # ------------------------------------------------------------------
    # Lazy loaders
    # ------------------------------------------------------------------

    def _load_pipeline(self) -> None:
        """Load AdaptiveDiffusionPipeline (lazy, avoids double VRAM at init)."""
        from src.diffusion.pipeline import AdaptiveDiffusionPipeline

        self._pipeline = AdaptiveDiffusionPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5",
            scheduler_name="ddim",
            device=self.device,
        )

    def _load_clip(self) -> None:
        """Load CLIP ViT-L/14 (openai) via open_clip (lazy)."""
        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="openai"
        )
        self._clip_model = model.to(self.device).eval()
        self._preprocess = preprocess
        self._tokenizer = open_clip.get_tokenizer("ViT-L-14")

    # ------------------------------------------------------------------
    # CLIP scoring
    # ------------------------------------------------------------------

    def _clip_score(self, image: Image.Image, text: str) -> float:
        """Return CLIP cosine similarity between a PIL image and a text string.

        Args:
            image: PIL Image (already decoded from z0_pred).
            text:  Prompt string.

        Returns:
            Scalar float in [-1, 1].
        """
        img_tensor = self._preprocess(image).unsqueeze(0).to(self.device)
        text_tokens = self._tokenizer([text]).to(self.device)
        with torch.no_grad():
            img_feat = F.normalize(
                self._clip_model.encode_image(img_tensor), dim=-1
            )
            txt_feat = F.normalize(
                self._clip_model.encode_text(text_tokens), dim=-1
            )
        return float((img_feat * txt_feat).sum())

    # ------------------------------------------------------------------
    # generate
    # ------------------------------------------------------------------

    def generate(
        self,
        prompts: list[str],
        seeds: list[int],
        output_dir: str,
        **kwargs,
    ) -> list[BaselineResult]:
        """Run full denoising, return image at CLIP-peak timestep per prompt.

        Each step's z0_pred is decoded to a PIL image and scored with CLIP.
        The image with the highest score is returned.  The metadata records
        ``peak_nfe`` (step index + 1 at which peak occurred) and ``peak_clip``.

        Args:
            prompts:    Text prompts.
            seeds:      Per-prompt random seeds.
            output_dir: Where to write PNGs and results.jsonl.
            **kwargs:   Ignored (e.g. guidance_scale override not exposed).

        Returns:
            List of BaselineResult, length == len(prompts).
        """
        from torchvision.transforms.functional import to_pil_image

        if self._pipeline is None:
            self._load_pipeline()
        if self._clip_model is None:
            self._load_clip()

        results = []
        for idx, (prompt, seed) in enumerate(zip(prompts, seeds)):
            t0 = time.time()
            state = self._pipeline.prepare(
                prompt, num_steps=self.max_steps, seed=seed
            )

            best_image: Optional[Image.Image] = None
            best_clip: float = -1.0
            best_nfe: int = self.max_steps

            for step_idx in range(self.max_steps):
                step_out = self._pipeline.denoise_step(state, guidance_scale=7.5)
                self._pipeline.advance_state(state, step_out)

                # Decode z0_pred to PIL for CLIP scoring (D-14: latent space)
                img_tensor = self._pipeline.decode(step_out.z0_pred)
                pil_img = to_pil_image(img_tensor[0].cpu().float())

                clip_s = self._clip_score(pil_img, prompt)
                if clip_s > best_clip:
                    best_clip = clip_s
                    best_image = pil_img
                    best_nfe = step_idx + 1  # 1-indexed NFE count

                if state.is_done:
                    break

            results.append(
                BaselineResult(
                    image=best_image,
                    prompt=prompt,
                    seed=seed,
                    nfe=best_nfe,
                    time_s=time.time() - t0,
                    method_name=self.name,
                    metadata={
                        "peak_clip": round(best_clip, 4),
                        "peak_nfe": best_nfe,
                        "max_steps": self.max_steps,
                    },
                )
            )
        return save_results(results, output_dir)
