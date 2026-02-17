"""Reward computation for the RL agent.

Composite reward: R = R_quality + R_efficiency + R_terminal
No outer lambda weights (D-13). All weighting through internal params.

References:
    - research.md §3.4 (reward function)
    - discovery.md D-09 (c_nfe not gamma)
    - discovery.md D-10 (NFE(refine) = 1+k)
    - discovery.md D-11 (R_stability is similarity, not delta)
    - discovery.md D-13 (no lambda_1, lambda_2)
    - discovery.md D-37 (normalize before weighting)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from src.agent.networks import ACTION_CONTINUE, ACTION_REFINE, ACTION_STOP


@dataclass
class RewardConfig:
    """Reward hyperparameters (research.md §3.6)."""

    # Quality delta weights
    alpha_1: float = 1.0   # CLIP delta
    alpha_2: float = 0.5   # Aesthetic delta
    alpha_3: float = 0.2   # Stability (DINO similarity, D-11)
    alpha_4: float = 0.8   # ImageReward delta

    # Normalization scales (D-37)
    clip_norm: float = 0.05
    aesthetic_norm: float = 0.3
    image_reward_norm: float = 0.2

    # Efficiency
    c_nfe: float = 0.01  # D-09: renamed from gamma

    # Terminal reward weights
    beta_1: float = 2.0  # CLIP score
    beta_2: float = 1.0  # Aesthetic
    beta_3: float = 1.5  # ImageReward

    # Refinement
    refine_k: int = 2

    # Normalization toggle (A5 ablation)
    normalize: bool = True


class RewardComputer:
    """Computes per-step rewards for the agent.

    Requires metric models to be available. Lazy-loads them on first use.
    """

    def __init__(self, config: RewardConfig = RewardConfig(), device: str = "cuda"):
        self.config = config
        self.device = device
        self._clip_model = None
        self._clip_preprocess = None
        self._clip_tokenizer = None
        self._ir_model = None
        self._aesthetic_model = None
        self._dino_model = None

    def _load_clip(self):
        if self._clip_model is None:
            import open_clip
            model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-L-14", pretrained="openai"
            )
            self._clip_model = model.to(self.device).eval()
            self._clip_preprocess = preprocess
            self._clip_tokenizer = open_clip.get_tokenizer("ViT-L-14")
        return self._clip_model, self._clip_preprocess, self._clip_tokenizer

    def _load_image_reward(self):
        if self._ir_model is None:
            import ImageReward as RM
            self._ir_model = RM.load("ImageReward-v1.0")
        return self._ir_model

    def _load_dino(self):
        if self._dino_model is None:
            self._dino_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
            self._dino_model = self._dino_model.to(self.device).eval()
        return self._dino_model

    @torch.no_grad()
    def clip_score(self, image: torch.Tensor, prompt: str) -> float:
        """Compute CLIP similarity between image and prompt.

        Args:
            image: (1, 3, H, W) in [0, 1].
            prompt: Text string.

        Returns:
            Cosine similarity (scalar).
        """
        clip_model, preprocess, tokenizer = self._load_clip()

        # Resize and normalize
        import torchvision.transforms.functional as TF
        img = TF.resize(image, [224, 224], antialias=True)
        img = TF.normalize(img, mean=[0.48145466, 0.4578275, 0.40821073],
                           std=[0.26862954, 0.26130258, 0.27577711])

        img_features = clip_model.encode_image(img.to(device=self.device, dtype=torch.float32))
        img_features = img_features / img_features.norm(dim=-1, keepdim=True)

        tokens = tokenizer([prompt]).to(self.device)
        txt_features = clip_model.encode_text(tokens)
        txt_features = txt_features / txt_features.norm(dim=-1, keepdim=True)

        similarity = (img_features * txt_features).sum(dim=-1).item()
        return similarity

    @torch.no_grad()
    def image_reward_score(self, image: torch.Tensor, prompt: str) -> float:
        """Compute ImageReward score.

        Args:
            image: (1, 3, H, W) in [0, 1].
            prompt: Text string.

        Returns:
            ImageReward score (scalar).
        """
        ir_model = self._load_image_reward()
        from PIL import Image as PILImage
        import numpy as np

        img_np = (image[0].permute(1, 2, 0).cpu().float().numpy() * 255).astype(np.uint8)
        pil_img = PILImage.fromarray(img_np)
        score = ir_model.score(prompt, pil_img)
        return float(score)

    @torch.no_grad()
    def dino_similarity(self, image1: torch.Tensor, image2: torch.Tensor) -> float:
        """Compute DINO cosine similarity between two images.

        D-11: This is a similarity score in [0, 1], NOT a delta.

        Args:
            image1: (1, 3, H, W) in [0, 1].
            image2: (1, 3, H, W) in [0, 1].

        Returns:
            Cosine similarity (scalar).
        """
        dino = self._load_dino()

        import torchvision.transforms.functional as TF

        def encode(img):
            img = TF.resize(img, [224, 224], antialias=True)
            img = TF.normalize(img, mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
            return dino(img.to(device=self.device, dtype=torch.float32))

        f1 = encode(image1)
        f2 = encode(image2)
        sim = F.cosine_similarity(f1, f2, dim=-1).item()
        return max(0.0, sim)  # Clamp to [0, 1]

    def compute_quality_reward(
        self,
        prev_image: Optional[torch.Tensor],
        curr_image: torch.Tensor,
        prompt: str,
        prev_clip: Optional[float] = None,
        prev_ir: Optional[float] = None,
    ) -> tuple[float, dict]:
        """Compute R_quality for a single step.

        R_quality = alpha_1 * norm(delta_CLIP) + alpha_2 * norm(delta_Aesthetic)
                  + alpha_3 * R_stability + alpha_4 * norm(delta_ImageReward)

        Args:
            prev_image: Previous step's decoded image (None for first step).
            curr_image: Current step's decoded image.
            prompt: Text prompt.
            prev_clip: Previous CLIP score (for caching).
            prev_ir: Previous ImageReward score (for caching).

        Returns:
            reward: Float quality reward.
            metrics: Dict with per-component values.
        """
        cfg = self.config

        # Current scores
        curr_clip = self.clip_score(curr_image, prompt)
        curr_ir = self.image_reward_score(curr_image, prompt)

        metrics = {"clip_score": curr_clip, "image_reward": curr_ir}

        if prev_image is None:
            # First step — no delta available
            return 0.0, metrics

        # Previous scores (use cached if available)
        if prev_clip is None:
            prev_clip = self.clip_score(prev_image, prompt)
        if prev_ir is None:
            prev_ir = self.image_reward_score(prev_image, prompt)

        # Deltas
        delta_clip = curr_clip - prev_clip
        delta_ir = curr_ir - prev_ir

        # Stability (D-11: similarity, not delta)
        r_stability = self.dino_similarity(prev_image, curr_image)
        metrics["dino_similarity"] = r_stability

        # Normalize (D-37)
        if cfg.normalize:
            delta_clip_norm = delta_clip / cfg.clip_norm
            delta_ir_norm = delta_ir / cfg.image_reward_norm
        else:
            delta_clip_norm = delta_clip
            delta_ir_norm = delta_ir

        # Note: aesthetic delta omitted for now (requires LAION aesthetic model)
        # Can be added when aesthetic model is loaded
        delta_aesthetic_norm = 0.0

        # Weighted sum
        reward = (
            cfg.alpha_1 * delta_clip_norm
            + cfg.alpha_2 * delta_aesthetic_norm
            + cfg.alpha_3 * r_stability
            + cfg.alpha_4 * delta_ir_norm
        )

        metrics.update({
            "delta_clip": delta_clip,
            "delta_clip_norm": delta_clip_norm,
            "delta_ir": delta_ir,
            "delta_ir_norm": delta_ir_norm,
            "r_quality": reward,
        })

        return reward, metrics

    def compute_efficiency_reward(self, action: int) -> float:
        """Compute R_efficiency = -c_nfe * NFE(action).

        D-09: Uses c_nfe (not gamma).
        D-10: NFE(refine) = 1 + k (not scaled by mask area).
        """
        if action == ACTION_CONTINUE:
            nfe = 1
        elif action == ACTION_STOP:
            nfe = 0
        elif action == ACTION_REFINE:
            nfe = 1 + self.config.refine_k  # D-10
        else:
            raise ValueError(f"Unknown action: {action}")

        return -self.config.c_nfe * nfe

    def compute_terminal_reward(
        self, image: torch.Tensor, prompt: str
    ) -> tuple[float, dict]:
        """Compute R_terminal for the final image.

        R_final = beta_1 * CLIP + beta_2 * Aesthetic + beta_3 * ImageReward
        """
        cfg = self.config

        clip = self.clip_score(image, prompt)
        ir = self.image_reward_score(image, prompt)

        reward = cfg.beta_1 * clip + cfg.beta_3 * ir
        # Aesthetic term omitted (same as quality reward — add when model loaded)

        metrics = {
            "terminal_clip": clip,
            "terminal_ir": ir,
            "r_terminal": reward,
        }
        return reward, metrics

    def compute_reward(
        self,
        prev_image: Optional[torch.Tensor],
        curr_image: torch.Tensor,
        prompt: str,
        action: int,
        is_terminal: bool,
        prev_clip: Optional[float] = None,
        prev_ir: Optional[float] = None,
    ) -> tuple[float, dict]:
        """Compute total reward R = R_quality + R_efficiency + R_terminal.

        D-13: No outer lambda weights. Sum directly.
        """
        r_quality, q_metrics = self.compute_quality_reward(
            prev_image, curr_image, prompt, prev_clip, prev_ir,
        )
        r_efficiency = self.compute_efficiency_reward(action)

        r_terminal = 0.0
        t_metrics = {}
        if is_terminal:
            r_terminal, t_metrics = self.compute_terminal_reward(curr_image, prompt)

        total = r_quality + r_efficiency + r_terminal

        metrics = {
            **q_metrics,
            **t_metrics,
            "r_quality": r_quality,
            "r_efficiency": r_efficiency,
            "r_terminal": r_terminal,
            "r_total": total,
        }
        return total, metrics
