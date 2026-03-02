"""Random-search baseline: best-of-N selection by CLIP score (Phase 2, Week 5).

Baseline B2 from plan.md Phase 2 Week 5.  Generates N independent images per
prompt (using different random seeds) and returns the one with the highest
image-text CLIP cosine similarity.  This isolates the contribution of
search/selection over a fixed compute budget vs. the agent's adaptive strategy.

NFE = N * steps (default: 8 * 20 = 160), matching the maximum agent budget
when N_max=50 with a generous refine budget.

References:
    - plan.md Phase 2 Week 5 (baseline B2 — random search)
    - research.md §4.1 (experimental baseline descriptions)
    - discovery.md D-15 (NFE = scheduler steps, not network evals)
"""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler
from PIL import Image

from src.baselines.base import BaseBaseline, BaselineResult, load_sd15_pipeline, save_results


class RandomSearchBaseline(BaseBaseline):
    """Best-of-N image generation scored by CLIP similarity (baseline B2).

    Generates ``n_samples`` images for each prompt (with seeds offset by
    multiples of 1000 for independence) and returns the image with the highest
    CLIP ViT-L/14 cosine similarity to the prompt text.

    Args:
        n_samples: Number of candidates to generate per prompt (default 8).
        steps:     Denoising steps per candidate (default 20, matching DDIM-20).
        device:    Torch device string (default "cuda").

    References:
        - plan.md Phase 2 Week 5 (baseline B2)
        - research.md §4.1
    """

    name = "random_search_8"
    nfe = 160  # 8 x DDIM-20

    def __init__(
        self,
        n_samples: int = 8,
        steps: int = 20,
        device: str = "cuda",
    ) -> None:
        self.n = n_samples
        self.steps = steps
        self.device = device
        # Update nfe dynamically if constructor args differ from class defaults
        self.nfe = n_samples * steps
        self.pipe = load_sd15_pipeline(device=device)
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self._clip_model = None
        self._preprocess = None
        self._tokenizer = None

    # ------------------------------------------------------------------
    # CLIP helpers (lazy load to avoid GPU memory cost until first use)
    # ------------------------------------------------------------------

    def _load_clip(self) -> None:
        """Load CLIP ViT-L/14 (openai) via open_clip (lazy)."""
        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="openai"
        )
        self._clip_model = model.to(self.device).eval()
        self._preprocess = preprocess
        self._tokenizer = open_clip.get_tokenizer("ViT-L-14")

    def _clip_score(self, image: Image.Image, text: str) -> float:
        """Return CLIP cosine similarity between an image and a text string.

        Args:
            image: PIL Image to score.
            text:  Prompt string.

        Returns:
            Scalar float in [-1, 1].
        """
        if self._clip_model is None:
            self._load_clip()

        img_tensor = self._preprocess(image).unsqueeze(0).to(self.device)
        text_tokens = self._tokenizer([text]).to(self.device)
        with torch.no_grad():
            img_feat = self._clip_model.encode_image(img_tensor)
            txt_feat = self._clip_model.encode_text(text_tokens)
            img_feat = F.normalize(img_feat, dim=-1)
            txt_feat = F.normalize(txt_feat, dim=-1)
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
        """Generate N candidates per prompt, return the best by CLIP score.

        Seeds are offset by multiples of 1000 across candidates to ensure
        independent draws from the same prompt-starting distribution.

        Args:
            prompts:    Text prompts.
            seeds:      Base seed per prompt; candidate i uses seed + i*1000.
            output_dir: Where to write PNGs and results.jsonl.
            **kwargs:   Ignored.

        Returns:
            List of BaselineResult, length == len(prompts).
        """
        results = []
        for idx, (prompt, seed) in enumerate(zip(prompts, seeds)):
            t0 = time.time()
            best_score: float = -1.0
            best_img: Image.Image | None = None

            for i in range(self.n):
                gen = torch.Generator(self.device).manual_seed(seed + i * 1000)
                img = self.pipe(
                    prompt,
                    num_inference_steps=self.steps,
                    generator=gen,
                ).images[0]
                score = self._clip_score(img, prompt)
                if score > best_score:
                    best_score, best_img = score, img

            results.append(
                BaselineResult(
                    image=best_img,
                    prompt=prompt,
                    seed=seed,
                    nfe=self.nfe,
                    time_s=time.time() - t0,
                    method_name=self.name,
                    metadata={
                        "n_samples": self.n,
                        "steps": self.steps,
                        "best_clip_score": round(best_score, 4),
                    },
                )
            )
        return save_results(results, output_dir)
