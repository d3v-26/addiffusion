"""Evaluation metrics computation for AdDiffusion (Phase 2, Week 6).

Reads a directory of generated PNG images and a prompts JSON file, then
computes a configurable set of image quality metrics and writes a JSON
output file.

Supported metrics:
    - clip_score  : CLIP ViT-L/14 cosine similarity (open_clip, openai pretrained)
    - fid         : Fréchet Inception Distance (cleanfid)
    - cmmd        : Squared MMD on L2-normalised CLIP embeddings (arXiv:2401.09603)
    - imagereward : ImageReward-v1.0 mean score
    - hpsv2       : HPS v2.1 mean score
    - is          : Inception Score (torch_fidelity)
    - aesthetic   : LAION aesthetic predictor (skipped if hub load fails)

Usage:
    uv run python src/evaluation/compute_metrics.py \\
        --images_dir outputs/baselines/ddim20/ \\
        --prompts_file data/drawbench/prompts.json \\
        --output_file outputs/evaluation/metrics/ddim20.json \\
        [--reference_dir data/coco_reference/] \\
        [--metrics fid clip_score cmmd imagereward hpsv2 is]

References:
    - plan.md Phase 2 Week 6 (evaluation pipeline)
    - research.md §4 (experimental setup)
    - discovery.md D-37 (reward normalisation — not applied here; raw metric values reported)
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image as PILImage


# ---------------------------------------------------------------------------
# CMMD helpers
# ---------------------------------------------------------------------------

def _rbf_kernel(x: np.ndarray, y: np.ndarray, sigma: float) -> np.ndarray:
    """RBF kernel k(x, y) = exp(-||x-y||^2 / (2*sigma^2))."""
    x_sq = np.sum(x ** 2, axis=1, keepdims=True)
    y_sq = np.sum(y ** 2, axis=1, keepdims=True)
    xy = x @ y.T
    sq_dists = x_sq + y_sq.T - 2 * xy
    return np.exp(-sq_dists / (2 * sigma ** 2))


def compute_cmmd(
    real_embeds: np.ndarray,
    gen_embeds: np.ndarray,
    sigma: float = 10.0,
) -> float:
    """Squared MMD with RBF kernel on L2-normalised CLIP embeddings.

    Reference: arXiv:2401.09603. sigma=10.0 per official implementation.

    Args:
        real_embeds: (N, d) float32 array of reference image embeddings.
        gen_embeds:  (M, d) float32 array of generated image embeddings.
        sigma:       RBF bandwidth (default 10.0 per paper).

    Returns:
        Squared MMD distance (float, >= 0).
    """
    # L2-normalise
    real_embeds = real_embeds / np.linalg.norm(real_embeds, axis=1, keepdims=True)
    gen_embeds = gen_embeds / np.linalg.norm(gen_embeds, axis=1, keepdims=True)

    xx = _rbf_kernel(real_embeds, real_embeds, sigma)
    yy = _rbf_kernel(gen_embeds, gen_embeds, sigma)
    xy = _rbf_kernel(real_embeds, gen_embeds, sigma)
    return float(xx.mean() + yy.mean() - 2 * xy.mean())


# ---------------------------------------------------------------------------
# MetricsComputer
# ---------------------------------------------------------------------------

class MetricsComputer:
    """Computes image quality metrics over a set of PIL images and prompts.

    All models are lazy-loaded on first use to avoid loading everything when
    only a subset of metrics are requested.

    Args:
        device: Torch device string (e.g. "cuda" or "cpu").
    """

    def __init__(self, device: str = "cuda") -> None:
        self.device = device

        # Lazy model handles
        self._clip_model = None
        self._clip_preprocess = None
        self._clip_tokenizer = None
        self._ir_model = None
        self._aesthetic_model = None

    # ------------------------------------------------------------------
    # Private model loaders
    # ------------------------------------------------------------------

    def _load_clip(self):
        """Load open_clip ViT-L/14 (openai pretrained) on first call."""
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
        """Load ImageReward-v1.0 on first call."""
        if self._ir_model is None:
            import ImageReward as RM
            self._ir_model = RM.load("ImageReward-v1.0")
        return self._ir_model

    def _load_aesthetic(self):
        """Load LAION aesthetic predictor via torch hub on first call.

        Returns None if the hub load fails (network unavailable or model not
        cached), so callers must handle a None return value gracefully.
        """
        if self._aesthetic_model is None:
            try:
                model = torch.hub.load(
                    "christophschuhmann/improved-aesthetic-predictor",
                    "improved_aesthetic_predictor",
                )
                self._aesthetic_model = model.to(self.device).eval()
            except Exception:
                # Hub load failed — aesthetic metric will be skipped.
                return None
        return self._aesthetic_model

    # ------------------------------------------------------------------
    # Per-metric computation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def compute_clip_scores(
        self,
        images: list[PILImage.Image],
        prompts: list[str],
    ) -> tuple[float, float]:
        """CLIP ViT-L/14 cosine similarity between each image and its prompt.

        Args:
            images:  List of PIL images.
            prompts: Matching list of text prompts.

        Returns:
            (mean_score, std_score)
        """
        clip_model, preprocess, tokenizer = self._load_clip()
        scores = []
        for img, prompt in zip(images, prompts):
            img_tensor = preprocess(img).unsqueeze(0).to(self.device)
            img_feat = clip_model.encode_image(img_tensor)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

            tokens = tokenizer([prompt]).to(self.device)
            txt_feat = clip_model.encode_text(tokens)
            txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)

            sim = (img_feat * txt_feat).sum(dim=-1).item()
            scores.append(sim)

        arr = np.array(scores, dtype=np.float64)
        return float(arr.mean()), float(arr.std())

    @torch.no_grad()
    def compute_imagereward(
        self,
        images: list[PILImage.Image],
        prompts: list[str],
    ) -> tuple[float, float]:
        """ImageReward-v1.0 score for each (image, prompt) pair.

        Args:
            images:  List of PIL images.
            prompts: Matching list of text prompts.

        Returns:
            (mean_score, std_score)
        """
        ir_model = self._load_image_reward()
        scores = []
        for img, prompt in zip(images, prompts):
            score = ir_model.score(prompt, img)
            scores.append(float(score))

        arr = np.array(scores, dtype=np.float64)
        return float(arr.mean()), float(arr.std())

    def compute_hpsv2(
        self,
        images: list[PILImage.Image],
        prompts: list[str],
    ) -> tuple[float, float]:
        """HPS v2.1 score for each (image, prompt) pair.

        hpsv2.score() returns a list[float]; we take element [0].

        Args:
            images:  List of PIL images.
            prompts: Matching list of text prompts.

        Returns:
            (mean_score, std_score)
        """
        import hpsv2
        scores = []
        for img, prompt in zip(images, prompts):
            result = hpsv2.score(img, prompt, hps_version="v2.1")
            scores.append(float(result[0]))

        arr = np.array(scores, dtype=np.float64)
        return float(arr.mean()), float(arr.std())

    def compute_fid(
        self,
        images_dir: str,
        reference_dir: Optional[str],
    ) -> Optional[float]:
        """Fréchet Inception Distance via cleanfid.

        If reference_dir is provided, computes FID between images_dir and
        reference_dir.  Otherwise attempts to use precomputed COCO-30k stats
        if they exist; returns None if neither is available.

        Args:
            images_dir:    Directory of generated PNG images.
            reference_dir: Optional directory of reference images.

        Returns:
            FID score (float) or None if computation is not possible.
        """
        from cleanfid import fid

        if reference_dir is not None:
            score = fid.compute_fid(images_dir, reference_dir, mode="clean")
            return float(score)

        # Try precomputed COCO-30k stats
        if fid.test_stats_exists("COCO-30k", "clean"):
            score = fid.compute_fid(
                images_dir,
                dataset_name="COCO-30k",
                dataset_split="custom",
                mode="clean",
            )
            return float(score)

        return None

    @torch.no_grad()
    def compute_cmmd(
        self,
        real_images: list[PILImage.Image],
        gen_images: list[PILImage.Image],
    ) -> float:
        """CMMD: squared MMD on L2-normalised CLIP ViT-L/14 embeddings.

        Reference: arXiv:2401.09603, sigma=10.0.

        Args:
            real_images: Reference PIL images.
            gen_images:  Generated PIL images.

        Returns:
            CMMD distance (float).
        """
        clip_model, preprocess, _ = self._load_clip()

        def _embed_batch(imgs: list[PILImage.Image]) -> np.ndarray:
            embeds = []
            for img in imgs:
                t = preprocess(img).unsqueeze(0).to(self.device)
                feat = clip_model.encode_image(t)
                embeds.append(feat.cpu().float().numpy())
            return np.concatenate(embeds, axis=0)

        real_embeds = _embed_batch(real_images)
        gen_embeds = _embed_batch(gen_images)
        return compute_cmmd(real_embeds, gen_embeds)

    def compute_is(
        self,
        images_dir: str,
    ) -> tuple[Optional[float], Optional[float]]:
        """Inception Score via torch_fidelity.

        Args:
            images_dir: Directory of generated PNG images.

        Returns:
            (is_mean, is_std) or (None, None) if computation fails.
        """
        try:
            import torch_fidelity
            metrics = torch_fidelity.calculate_metrics(
                input1=images_dir,
                isc=True,
                verbose=False,
            )
            return float(metrics["inception_score_mean"]), float(metrics["inception_score_std"])
        except Exception:
            return None, None

    @torch.no_grad()
    def _compute_aesthetic_scores(
        self,
        images: list[PILImage.Image],
    ) -> tuple[Optional[float], Optional[float]]:
        """Aesthetic scores via LAION improved aesthetic predictor.

        Returns (None, None) if the model could not be loaded.

        Args:
            images: List of PIL images.

        Returns:
            (mean_score, std_score) or (None, None).
        """
        aes_model = self._load_aesthetic()
        if aes_model is None:
            return None, None

        import open_clip
        clip_model, clip_preprocess, _ = self._load_clip()

        scores = []
        for img in images:
            t = clip_preprocess(img).unsqueeze(0).to(self.device)
            feat = clip_model.encode_image(t)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            score = aes_model(feat.float()).item()
            scores.append(score)

        arr = np.array(scores, dtype=np.float64)
        return float(arr.mean()), float(arr.std())

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def compute_all(
        self,
        images_dir: str,
        prompts_file: str,
        output_file: str,
        reference_dir: Optional[str] = None,
        metrics: Optional[list[str]] = None,
    ) -> dict:
        """Compute all requested metrics and write results to output_file.

        Args:
            images_dir:    Directory containing generated PNG files (sorted).
            prompts_file:  JSON file containing a list of prompt strings.
            output_file:   Path where the JSON results dict will be written.
            reference_dir: Optional reference image directory for FID/CMMD.
            metrics:       List of metric names to compute.  Defaults to all.

        Returns:
            Dict of computed metrics (same content as output_file).
        """
        all_metrics = {"fid", "clip_score", "cmmd", "imagereward", "hpsv2", "is", "aesthetic"}
        if metrics is None:
            metrics = list(all_metrics)
        requested = set(metrics)

        # Load images
        img_paths = sorted(Path(images_dir).glob("*.png"))
        if not img_paths:
            raise FileNotFoundError(f"No PNG files found in {images_dir}")

        # Load prompts
        with open(prompts_file, "r", encoding="utf-8") as f:
            prompts: list[str] = json.load(f)

        n = min(len(img_paths), len(prompts))
        img_paths = img_paths[:n]
        prompts = prompts[:n]

        print(f"[compute_metrics] Processing {n} images from {images_dir}")

        images = [PILImage.open(p).convert("RGB") for p in img_paths]

        result: dict = {
            "method": Path(images_dir).name,
            "n_images": n,
            "timestamp": datetime.datetime.utcnow().isoformat(),
        }

        # --- CLIP Score ---
        if "clip_score" in requested:
            print("[compute_metrics] Computing CLIP score...")
            clip_mean, clip_std = self.compute_clip_scores(images, prompts)
            result["clip_score_mean"] = clip_mean
            result["clip_score_std"] = clip_std
            print(f"  clip_score: {clip_mean:.4f} ± {clip_std:.4f}")

        # --- ImageReward ---
        if "imagereward" in requested:
            print("[compute_metrics] Computing ImageReward...")
            ir_mean, ir_std = self.compute_imagereward(images, prompts)
            result["imagereward_mean"] = ir_mean
            result["imagereward_std"] = ir_std
            print(f"  imagereward: {ir_mean:.4f} ± {ir_std:.4f}")

        # --- HPS v2 ---
        if "hpsv2" in requested:
            print("[compute_metrics] Computing HPS v2.1...")
            hps_mean, hps_std = self.compute_hpsv2(images, prompts)
            result["hpsv2_mean"] = hps_mean
            result["hpsv2_std"] = hps_std
            print(f"  hpsv2: {hps_mean:.4f} ± {hps_std:.4f}")

        # --- FID ---
        if "fid" in requested:
            print("[compute_metrics] Computing FID...")
            fid_score = self.compute_fid(images_dir, reference_dir)
            result["fid"] = fid_score
            print(f"  fid: {fid_score}")

        # --- CMMD ---
        if "cmmd" in requested:
            if reference_dir is not None:
                print("[compute_metrics] Computing CMMD...")
                ref_paths = sorted(Path(reference_dir).glob("*.png"))
                ref_images = [PILImage.open(p).convert("RGB") for p in ref_paths[:n]]
                cmmd_score = self.compute_cmmd(ref_images, images)
                result["cmmd"] = cmmd_score
                print(f"  cmmd: {cmmd_score:.4f}")
            else:
                result["cmmd"] = None
                print("[compute_metrics] CMMD skipped — no reference_dir provided")

        # --- Aesthetic ---
        if "aesthetic" in requested:
            print("[compute_metrics] Computing Aesthetic score...")
            aes_mean, aes_std = self._compute_aesthetic_scores(images)
            result["aesthetic_mean"] = aes_mean
            result["aesthetic_std"] = aes_std
            if aes_mean is not None:
                print(f"  aesthetic: {aes_mean:.4f} ± {aes_std:.4f}")
            else:
                print("  aesthetic: skipped (model unavailable)")

        # --- Inception Score ---
        if "is" in requested:
            print("[compute_metrics] Computing Inception Score...")
            is_mean, is_std = self.compute_is(images_dir)
            result["is_mean"] = is_mean
            result["is_std"] = is_std
            if is_mean is not None:
                print(f"  IS: {is_mean:.4f} ± {is_std:.4f}")
            else:
                print("  IS: skipped (computation failed)")

        # Write output
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        print(f"[compute_metrics] Results written to {output_file}")
        return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute image quality metrics for a directory of generated images."
    )
    parser.add_argument(
        "--images_dir",
        required=True,
        help="Directory containing generated PNG images.",
    )
    parser.add_argument(
        "--prompts_file",
        required=True,
        help="JSON file containing a list of prompt strings.",
    )
    parser.add_argument(
        "--output_file",
        required=True,
        help="Output JSON file path.",
    )
    parser.add_argument(
        "--reference_dir",
        default=None,
        help="Reference image directory (for FID and CMMD).",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        choices=["fid", "clip_score", "cmmd", "imagereward", "hpsv2", "is", "aesthetic"],
        help="Metrics to compute (default: all).",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device (default: cuda if available, else cpu).",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for CLI usage."""
    args = _parse_args()
    computer = MetricsComputer(device=args.device)
    computer.compute_all(
        images_dir=args.images_dir,
        prompts_file=args.prompts_file,
        output_file=args.output_file,
        reference_dir=args.reference_dir,
        metrics=args.metrics,
    )


if __name__ == "__main__":
    main()
