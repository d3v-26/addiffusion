"""Region-selective refinement for diffusion inference.

Implements the RegionRefine algorithm: identifies under-generated regions via
cross-attention maps, applies Gaussian-blurred soft masks, and re-denoises
masked regions for k iterations.

References:
    - research.md §3.5 (algorithm + mask generation)
    - discovery.md D-10 (NFE = 1+k, not scaled by mask area)
    - discovery.md D-19 (post-refinement re-noise to next schedule point)
    - discovery.md D-26 (soft masks via Gaussian blur)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from src.diffusion.attention import AttentionExtractor
from src.diffusion.pipeline import AdaptiveDiffusionPipeline, PipelineState


def generate_refinement_mask(
    attention_maps: torch.Tensor,
    threshold: float = 0.5,
    blur_sigma: float = 3.0,
    blur_kernel_size: int = 15,
) -> torch.Tensor:
    """Generate a soft refinement mask from cross-attention maps.

    Identifies regions with low max attention across all text tokens, then
    applies Gaussian blur for smooth boundaries (D-26).

    Args:
        attention_maps: (h, w, L) from AttentionExtractor.get_attention_maps()
        threshold: Quantile threshold — regions below this percentile of
            max-attention are marked for refinement.
        blur_sigma: Gaussian blur sigma for mask softening (D-26).
        blur_kernel_size: Kernel size for Gaussian blur. Must be odd.

    Returns:
        Soft mask (1, 1, h, w) in [0, 1]. Values near 1 = refine, near 0 = keep.
    """
    # Max attention across all text tokens per spatial location
    max_attn = attention_maps.max(dim=-1).values  # (h, w)

    # Hard binary mask: regions with low attention (D-07)
    attn_threshold = max_attn.quantile(threshold)
    hard_mask = (max_attn < attn_threshold).float()  # (h, w)

    # Apply Gaussian blur for soft boundaries (D-26)
    # Reshape for conv2d: (1, 1, h, w)
    mask = hard_mask.unsqueeze(0).unsqueeze(0)
    mask = _gaussian_blur(mask, kernel_size=blur_kernel_size, sigma=blur_sigma)

    # Clamp to [0, 1]
    mask = mask.clamp(0, 1)

    return mask


def _gaussian_blur(
    x: torch.Tensor, kernel_size: int = 15, sigma: float = 3.0
) -> torch.Tensor:
    """Apply 2D Gaussian blur to a (B, C, H, W) tensor."""
    # Create 1D Gaussian kernel
    coords = torch.arange(kernel_size, dtype=x.dtype, device=x.device) - kernel_size // 2
    gauss = torch.exp(-coords.pow(2) / (2 * sigma**2))
    gauss = gauss / gauss.sum()

    # Outer product for 2D kernel
    kernel_2d = gauss.outer(gauss)
    kernel_2d = kernel_2d.view(1, 1, kernel_size, kernel_size)

    # Apply with same padding
    pad = kernel_size // 2
    return F.conv2d(x, kernel_2d, padding=pad)


def region_refine(
    pipeline: AdaptiveDiffusionPipeline,
    state: PipelineState,
    mask: torch.Tensor,
    k: int = 2,
    r_noise: float = 0.5,
    guidance_scale: float = 7.5,
    generator: Optional[torch.Generator] = None,
) -> tuple[torch.Tensor, int]:
    """Execute the RegionRefine algorithm.

    Algorithm (research.md §3.5):
        1. Get one-step clean prediction z0_hat
        2. For j = 1 to k:
           a. Re-noise masked regions: z_t' = m * AddNoise(z0_hat, t') + (1-m) * z0_hat
           b. Denoise: z0_hat = Denoise(z_t', c, t')
        3. Return refined z0_hat

    NFE cost: 1 + k (D-10: full forward passes regardless of mask size).

    Args:
        pipeline: The diffusion pipeline.
        state: Current episode state.
        mask: Soft mask (1, 1, h, w) in [0, 1]. 1 = refine, 0 = keep.
        k: Number of refinement iterations.
        r_noise: Noise reduction ratio. t' = t * r_noise.
        guidance_scale: CFG scale for denoising.
        generator: Random generator for reproducibility.

    Returns:
        z0_refined: Refined clean prediction (1, 4, h, w).
        nfe: NFE consumed (1 + k).
    """
    current_t = state.timesteps[state.step_index]
    t_prime = int(current_t.item() * r_noise)  # Reduced noise level

    # Step 1: One-step clean prediction
    noise_pred = pipeline.unet_forward(
        state.z_t, current_t, state.prompt_embeds, state.negative_prompt_embeds,
        guidance_scale=guidance_scale,
    )
    z0_hat = pipeline.predict_clean(state.z_t, noise_pred, current_t)

    # Steps 2a-2b: k refinement iterations
    for j in range(k):
        # Re-noise masked regions to t'
        noise = torch.randn(z0_hat.shape, dtype=z0_hat.dtype, device=z0_hat.device, generator=generator)
        t_prime_tensor = torch.tensor([t_prime], device=pipeline.device, dtype=torch.long)

        alpha_prod_t = pipeline.scheduler.alphas_cumprod[t_prime_tensor.long()]
        while alpha_prod_t.dim() < z0_hat.dim():
            alpha_prod_t = alpha_prod_t.unsqueeze(-1)

        z_noised = alpha_prod_t.sqrt() * z0_hat + (1 - alpha_prod_t).sqrt() * noise

        # Composite: masked regions get noise, unmasked keep clean
        z_composite = mask * z_noised + (1 - mask) * z0_hat

        # Denoise (full forward pass — D-10)
        noise_pred = pipeline.unet_forward(
            z_composite, t_prime, state.prompt_embeds, state.negative_prompt_embeds,
            guidance_scale=guidance_scale,
        )
        z0_hat = pipeline.predict_clean(z_composite, noise_pred, t_prime)

    nfe = 1 + k  # D-10: 1 initial + k refinement passes
    return z0_hat, nfe


def apply_refine_action(
    pipeline: AdaptiveDiffusionPipeline,
    state: PipelineState,
    attention_extractor: AttentionExtractor,
    k: int = 2,
    r_noise: float = 0.5,
    mask_threshold: float = 0.5,
    blur_sigma: float = 3.0,
    guidance_scale: float = 7.5,
    generator: Optional[torch.Generator] = None,
) -> tuple[torch.Tensor, int]:
    """Full refine action: generate mask + run RegionRefine + re-noise to next step.

    This is the complete action handler for a_refine.

    Args:
        pipeline: Diffusion pipeline.
        state: Current episode state.
        attention_extractor: For cross-attention map extraction.
        k: Refinement iterations.
        r_noise: Noise reduction ratio.
        mask_threshold: Quantile for mask generation.
        blur_sigma: Gaussian blur sigma for soft masks.
        guidance_scale: CFG scale.
        generator: For reproducibility.

    Returns:
        z0_refined: The refined clean prediction.
        nfe: Total NFE consumed (1 + k).
    """
    # Get attention maps (should already be populated from the most recent denoise step)
    attn_maps = attention_extractor.get_attention_maps()

    # Generate soft mask
    mask = generate_refinement_mask(
        attn_maps, threshold=mask_threshold, blur_sigma=blur_sigma
    )

    # Run region refinement
    z0_refined, nfe = region_refine(
        pipeline, state, mask,
        k=k, r_noise=r_noise, guidance_scale=guidance_scale, generator=generator,
    )

    # Post-refinement state transition (D-19):
    # Re-noise refined prediction to next schedule point
    if state.step_index + 1 < len(state.timesteps):
        next_t = state.timesteps[state.step_index + 1].item()
        state.z_t = pipeline.add_noise(z0_refined, int(next_t), generator=generator)
    else:
        # Last step — no need to re-noise
        state.z_t = z0_refined

    state.step_index += 1
    state.total_nfe += nfe

    if state.step_index >= len(state.timesteps):
        state.is_done = True

    return z0_refined, nfe
