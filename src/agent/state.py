"""State feature extraction for the RL agent.

Computes phi(s_i) = [CLIP_img(x0_hat), CLIP_txt(c), emb(t_i), q_i]
using frozen encoders (D-04). All features are detached (no gradients
flow through encoders).

References:
    - research.md §3.2 (state space)
    - discovery.md D-04 (frozen encoders)
    - discovery.md D-16 (early-step unreliability)
"""

from __future__ import annotations

import math
from typing import Optional

import torch

# Lazy imports to avoid loading heavy models at module import time
_clip_model = None
_clip_preprocess = None
_clip_tokenizer = None


def _load_clip(device: str = "cuda", dtype: torch.dtype = torch.float16):
    """Lazy-load CLIP ViT-L-14 (openai pretrained)."""
    global _clip_model, _clip_preprocess, _clip_tokenizer
    if _clip_model is None:
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="openai"
        )
        _clip_model = model.to(device).to(dtype).eval()
        _clip_preprocess = preprocess
        _clip_tokenizer = open_clip.get_tokenizer("ViT-L-14")
    return _clip_model, _clip_preprocess, _clip_tokenizer


class StateExtractor:
    """Extracts state features phi(s_i) for the RL agent.

    Feature vector composition:
        - CLIP image features: 768-dim (from decoded one-step prediction)
        - CLIP text features: 768-dim (from prompt, computed once per episode)
        - Timestep embedding: 128-dim (sinusoidal)
        - Quality vector: ~8-dim (running metrics)
        Total: ~1672-dim (rounded to nearest architecture-friendly size)

    All encoders are frozen (D-04) — no gradients flow through them.
    """

    # Feature dimensions
    CLIP_IMG_DIM = 768
    CLIP_TXT_DIM = 768
    TIMESTEP_DIM = 128
    QUALITY_DIM = 8  # CLIP score, aesthetic delta, ImageReward, DINO sim, step ratio, NFE ratio, + 2 padding

    TOTAL_DIM = CLIP_IMG_DIM + CLIP_TXT_DIM + TIMESTEP_DIM + QUALITY_DIM  # 1672

    def __init__(
        self,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        max_timestep: int = 1000,
    ):
        self.device = device
        self.dtype = dtype
        self.max_timestep = max_timestep

        # Cache for text features (computed once per episode)
        self._cached_text_features: Optional[torch.Tensor] = None
        self._cached_prompt: Optional[str] = None

    def _get_clip(self):
        return _load_clip(self.device, self.dtype)

    @torch.no_grad()
    def encode_image(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Encode a pixel-space image [0,1] to CLIP image features.

        Args:
            image_tensor: (1, 3, H, W) in [0, 1]

        Returns:
            (1, 768) CLIP image embedding (L2-normalized)
        """
        clip_model, preprocess, _ = self._get_clip()

        # Resize and normalize for CLIP (224x224)
        import torchvision.transforms.functional as TF
        img = TF.resize(image_tensor, [224, 224], antialias=True)
        img = TF.normalize(img, mean=[0.48145466, 0.4578275, 0.40821073],
                           std=[0.26862954, 0.26130258, 0.27577711])

        features = clip_model.encode_image(img.to(self.dtype))
        features = features / features.norm(dim=-1, keepdim=True)
        return features.float()  # (1, 768)

    @torch.no_grad()
    def encode_text(self, prompt: str) -> torch.Tensor:
        """Encode text prompt to CLIP text features. Cached per episode.

        Args:
            prompt: Text prompt string.

        Returns:
            (1, 768) CLIP text embedding (L2-normalized)
        """
        if self._cached_prompt == prompt and self._cached_text_features is not None:
            return self._cached_text_features

        clip_model, _, tokenizer = self._get_clip()
        tokens = tokenizer([prompt]).to(self.device)
        features = clip_model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)
        features = features.float()  # (1, 768)

        self._cached_text_features = features
        self._cached_prompt = prompt
        return features

    def encode_timestep(self, timestep: int) -> torch.Tensor:
        """Sinusoidal timestep embedding.

        Args:
            timestep: Current diffusion timestep (0 to max_timestep).

        Returns:
            (1, 128) timestep embedding.
        """
        half_dim = self.TIMESTEP_DIM // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=self.device, dtype=torch.float32) * -emb)
        emb = torch.tensor([timestep], device=self.device, dtype=torch.float32) * emb
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb.unsqueeze(0)  # (1, 128)

    def build_quality_vector(
        self,
        clip_score: float = 0.0,
        aesthetic_delta: float = 0.0,
        image_reward_delta: float = 0.0,
        dino_similarity: float = 0.0,
        step_ratio: float = 0.0,
        nfe_ratio: float = 0.0,
    ) -> torch.Tensor:
        """Build the quality history vector q_i.

        Args:
            clip_score: Current CLIP similarity with prompt.
            aesthetic_delta: Change in aesthetic score from last step.
            image_reward_delta: Change in ImageReward from last step.
            dino_similarity: DINO cosine similarity with previous prediction (D-11: not a delta).
            step_ratio: Current step / N_max (progress indicator).
            nfe_ratio: Current NFE / N_max (compute usage indicator).

        Returns:
            (1, 8) quality vector.
        """
        q = torch.tensor(
            [[clip_score, aesthetic_delta, image_reward_delta, dino_similarity,
              step_ratio, nfe_ratio, 0.0, 0.0]],
            device=self.device, dtype=torch.float32,
        )
        return q

    def extract(
        self,
        decoded_image: torch.Tensor,
        prompt: str,
        timestep: int,
        clip_score: float = 0.0,
        aesthetic_delta: float = 0.0,
        image_reward_delta: float = 0.0,
        dino_similarity: float = 0.0,
        step_ratio: float = 0.0,
        nfe_ratio: float = 0.0,
    ) -> torch.Tensor:
        """Extract the full state feature vector phi(s_i).

        Args:
            decoded_image: (1, 3, H, W) pixel-space image in [0, 1] (from VAE decode of z0_pred).
            prompt: Text prompt.
            timestep: Current diffusion timestep.
            clip_score: Current CLIP score.
            aesthetic_delta: Change in aesthetic score.
            image_reward_delta: Change in ImageReward.
            dino_similarity: DINO similarity with previous step (D-11).
            step_ratio: step_index / N_max.
            nfe_ratio: total_nfe / N_max.

        Returns:
            (1, TOTAL_DIM) state feature vector.
        """
        clip_img = self.encode_image(decoded_image)  # (1, 768)
        clip_txt = self.encode_text(prompt)  # (1, 768)
        t_emb = self.encode_timestep(timestep)  # (1, 128)
        q = self.build_quality_vector(
            clip_score, aesthetic_delta, image_reward_delta,
            dino_similarity, step_ratio, nfe_ratio,
        )  # (1, 8)

        phi = torch.cat([clip_img, clip_txt, t_emb, q], dim=-1)  # (1, 1672)
        return phi

    def reset_cache(self) -> None:
        """Clear cached text features. Call at start of new episode."""
        self._cached_text_features = None
        self._cached_prompt = None
