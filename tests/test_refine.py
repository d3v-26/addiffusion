"""Tests for region refinement.

Run: uv run python tests/test_refine.py
Requires: GPU with SD 1.5 weights available.
"""

import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.diffusion.pipeline import AdaptiveDiffusionPipeline
from src.diffusion.attention import UNetAttentionExtractor
from src.diffusion.refine import generate_refinement_mask, region_refine, apply_refine_action


def test_mask_generation():
    """Test mask generation from synthetic attention maps."""
    print("=" * 60)
    print("TEST: Mask generation")
    print("=" * 60)

    # Create synthetic attention maps (h, w, L)
    h, w, L = 64, 64, 77
    attn_maps = torch.rand(h, w, L)

    # Make a region with low attention (top-left quadrant)
    attn_maps[:32, :32, :] *= 0.1

    mask = generate_refinement_mask(attn_maps, threshold=0.5, blur_sigma=3.0)

    assert mask.shape == (1, 1, h, w), f"Wrong mask shape: {mask.shape}"
    assert mask.min() >= 0 and mask.max() <= 1, f"Mask values out of [0,1]: [{mask.min()}, {mask.max()}]"

    # Top-left should have higher mask values (needs refinement)
    tl_mean = mask[0, 0, :32, :32].mean().item()
    br_mean = mask[0, 0, 32:, 32:].mean().item()
    assert tl_mean > br_mean, f"Top-left ({tl_mean:.3f}) should have higher mask values than bottom-right ({br_mean:.3f})"
    print(f"  Top-left mask mean: {tl_mean:.3f}, Bottom-right: {br_mean:.3f}")
    print("[PASS] Mask correctly identifies low-attention regions")

    # Verify soft mask (should not be purely binary due to Gaussian blur)
    unique_vals = mask.unique()
    assert len(unique_vals) > 2, f"Mask should be soft (>2 unique values), got {len(unique_vals)}"
    print(f"[PASS] Soft mask: {len(unique_vals)} unique values (Gaussian blur applied)")


def test_mask_threshold():
    """Test that threshold controls mask coverage."""
    print("\n" + "=" * 60)
    print("TEST: Mask threshold sensitivity")
    print("=" * 60)

    attn_maps = torch.rand(64, 64, 77)

    mask_low = generate_refinement_mask(attn_maps, threshold=0.2, blur_sigma=3.0)
    mask_mid = generate_refinement_mask(attn_maps, threshold=0.5, blur_sigma=3.0)
    mask_high = generate_refinement_mask(attn_maps, threshold=0.8, blur_sigma=3.0)

    coverage_low = mask_low.mean().item()
    coverage_mid = mask_mid.mean().item()
    coverage_high = mask_high.mean().item()

    print(f"  Threshold 0.2 → coverage {coverage_low:.3f}")
    print(f"  Threshold 0.5 → coverage {coverage_mid:.3f}")
    print(f"  Threshold 0.8 → coverage {coverage_high:.3f}")

    assert coverage_low < coverage_mid < coverage_high, "Higher threshold should produce more coverage"
    print("[PASS] Threshold correctly controls mask coverage")


def test_region_refine(model_id: str = "stable-diffusion-v1-5/stable-diffusion-v1-5"):
    """Test the RegionRefine algorithm on a real model."""
    print("\n" + "=" * 60)
    print("TEST: RegionRefine algorithm")
    print("=" * 60)

    pipe = AdaptiveDiffusionPipeline.from_pretrained(model_id, scheduler_name="ddim")
    state = pipe.prepare("a red apple on a wooden table", num_steps=20, seed=42)

    # Run a few steps first
    for _ in range(5):
        step_out = pipe.denoise_step(state, guidance_scale=7.5)
        pipe.advance_state(state, step_out)

    # Create a synthetic mask (refine bottom half)
    mask = torch.zeros(1, 1, 64, 64, device=pipe.device, dtype=pipe.dtype)
    mask[:, :, 32:, :] = 1.0

    nfe_before = state.total_nfe
    z0_refined, nfe = region_refine(pipe, state, mask, k=2, r_noise=0.5)

    assert z0_refined.shape == (1, 4, 64, 64), f"Wrong output shape: {z0_refined.shape}"
    assert nfe == 3, f"NFE should be 1+k=3, got {nfe} (D-10)"
    print(f"[PASS] RegionRefine: output shape={z0_refined.shape}, NFE={nfe} (1+k=3, correct per D-10)")

    # Decode and save
    image = pipe.decode(z0_refined)
    img = Image.fromarray((image[0].permute(1, 2, 0).cpu().float().numpy() * 255).astype("uint8"))
    out_path = Path("outputs/test_refine_output.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    print(f"[PASS] Refined image saved to {out_path}")


def test_apply_refine_action(model_id: str = "stable-diffusion-v1-5/stable-diffusion-v1-5"):
    """Test the full refine action (mask generation + refine + state transition)."""
    print("\n" + "=" * 60)
    print("TEST: Full refine action (mask + refine + D-19 state transition)")
    print("=" * 60)

    pipe = AdaptiveDiffusionPipeline.from_pretrained(model_id, scheduler_name="ddim")
    extractor = UNetAttentionExtractor(latent_h=64, latent_w=64)
    extractor.hook_with_processor(pipe.unet)

    state = pipe.prepare("five colorful balloons floating in a blue sky", num_steps=20, seed=42)

    # Run a few steps to build up the image
    for i in range(8):
        extractor.clear()
        step_out = pipe.denoise_step(state, guidance_scale=7.5)
        pipe.advance_state(state, step_out)

    step_before = state.step_index
    nfe_before = state.total_nfe
    z_t_before = state.z_t.clone()

    # Apply refine action (uses last step's attention maps)
    extractor.clear()
    # Need one more forward pass to populate attention maps
    step_out = pipe.denoise_step(state, guidance_scale=7.5)
    # Don't advance — we'll refine instead
    # Reset extractor and run again to get fresh attention maps
    # Actually, the denoise_step already populated attention maps

    z0_refined, nfe = apply_refine_action(
        pipe, state, extractor,
        k=2, r_noise=0.5, mask_threshold=0.5, blur_sigma=3.0, guidance_scale=7.5,
    )

    # State should have advanced
    assert state.step_index == step_before + 1, f"Step should advance: {step_before} → {state.step_index}"
    assert state.total_nfe == nfe_before + nfe, f"NFE mismatch: {nfe_before} + {nfe} != {state.total_nfe}"
    assert nfe == 3, f"Refine NFE should be 1+k=3, got {nfe}"

    # z_t should have changed (re-noised to next schedule point, D-19)
    z_t_diff = (state.z_t - z_t_before).abs().mean().item()
    assert z_t_diff > 0.01, f"z_t should change after refine (diff={z_t_diff:.4f})"
    print(f"[PASS] State advanced: step {step_before}→{state.step_index}, NFE +{nfe}, z_t changed (diff={z_t_diff:.4f})")

    # Can still continue denoising after refine
    remaining = len(state.timesteps) - state.step_index
    print(f"  Remaining steps: {remaining}")
    for _ in range(min(3, remaining)):
        extractor.clear()
        step_out = pipe.denoise_step(state, guidance_scale=7.5)
        pipe.advance_state(state, step_out)

    print(f"[PASS] Continued denoising after refine. Total NFE={state.total_nfe}")

    extractor.clear()
    extractor.remove_hooks()


if __name__ == "__main__":
    model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    if len(sys.argv) > 1:
        model_id = sys.argv[1]

    test_mask_generation()
    test_mask_threshold()
    test_region_refine(model_id)
    test_apply_refine_action(model_id)

    print("\n" + "=" * 60)
    print("ALL REFINEMENT TESTS PASSED")
    print("=" * 60)
