"""Tests for cross-attention map extraction.

Run: uv run python tests/test_attention.py
Requires: GPU with SD 1.5 weights available.
"""

import sys
from pathlib import Path

import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.diffusion.pipeline import AdaptiveDiffusionPipeline
from src.diffusion.attention import UNetAttentionExtractor, create_attention_extractor


def test_attention_extraction(model_id: str = "stable-diffusion-v1-5/stable-diffusion-v1-5"):
    """Test that cross-attention maps are extracted correctly."""
    print("=" * 60)
    print("TEST: Cross-attention map extraction")
    print("=" * 60)

    pipe = AdaptiveDiffusionPipeline.from_pretrained(model_id, scheduler_name="ddim")

    # Create extractor and hook into UNet
    extractor = create_attention_extractor("unet", latent_h=64, latent_w=64)
    assert isinstance(extractor, UNetAttentionExtractor)

    extractor.hook_with_processor(pipe.unet)
    print("[PASS] Hooks registered on UNet attention processors")

    # Run a single denoising step to generate attention maps
    prompt = "a red cat sitting on a blue chair"
    state = pipe.prepare(prompt, num_steps=20, seed=42)

    extractor.clear()
    step_out = pipe.denoise_step(state, guidance_scale=7.5)

    # Get aggregated attention maps
    attn_maps = extractor.get_attention_maps()
    print(f"  Attention map shape: {attn_maps.shape}")

    # Should be (h, w, L) where L is the max token length (77 for SD 1.5 CLIP)
    assert attn_maps.shape[0] == 64, f"Height should be 64, got {attn_maps.shape[0]}"
    assert attn_maps.shape[1] == 64, f"Width should be 64, got {attn_maps.shape[1]}"
    assert attn_maps.shape[2] == 77, f"Seq len should be 77, got {attn_maps.shape[2]}"
    print(f"[PASS] Attention maps shape: {attn_maps.shape} (h=64, w=64, L=77)")

    # Verify values are valid probabilities
    assert attn_maps.min() >= 0, f"Negative attention values: min={attn_maps.min()}"
    print(f"  Value range: [{attn_maps.min():.4f}, {attn_maps.max():.4f}]")
    print("[PASS] Attention values are non-negative")

    # Visualize: max attention across tokens (used for mask generation)
    max_attn = attn_maps.max(dim=-1).values  # (h, w)
    assert max_attn.shape == (64, 64)
    print(f"[PASS] Max attention per spatial location: shape={max_attn.shape}")

    # Save visualization
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Max attention across all tokens
    axes[0].imshow(max_attn.cpu().numpy(), cmap="hot")
    axes[0].set_title("Max attention (all tokens)")
    axes[0].axis("off")

    # Attention for a specific token (e.g., "cat" — token index depends on tokenizer)
    tokens = pipe.tokenizer.encode(prompt)
    token_strs = [pipe.tokenizer.decode([t]) for t in tokens]
    # Find "cat" token
    cat_idx = None
    for i, ts in enumerate(token_strs):
        if "cat" in ts.lower():
            cat_idx = i
            break
    if cat_idx is not None:
        axes[1].imshow(attn_maps[:, :, cat_idx].cpu().numpy(), cmap="hot")
        axes[1].set_title(f'Attention for "{token_strs[cat_idx].strip()}" (idx={cat_idx})')
    else:
        axes[1].imshow(attn_maps[:, :, 2].cpu().numpy(), cmap="hot")
        axes[1].set_title(f"Attention for token idx=2")
    axes[1].axis("off")

    # Low-attention mask (regions that would be refined)
    threshold = 0.5
    low_attn_mask = (max_attn < max_attn.quantile(threshold)).float()
    axes[2].imshow(low_attn_mask.cpu().numpy(), cmap="gray")
    axes[2].set_title(f"Low-attention mask (bottom {threshold*100:.0f}%)")
    axes[2].axis("off")

    plt.suptitle(f'Prompt: "{prompt}"')
    plt.tight_layout()
    plt.savefig(out_dir / "test_attention_maps.png", dpi=100)
    plt.close()
    print(f"[PASS] Visualization saved to {out_dir / 'test_attention_maps.png'}")

    # Cleanup
    extractor.clear()
    extractor.remove_hooks()


def test_attention_multiple_steps(model_id: str = "stable-diffusion-v1-5/stable-diffusion-v1-5"):
    """Test that attention maps update across denoising steps."""
    print("\n" + "=" * 60)
    print("TEST: Attention maps across multiple steps")
    print("=" * 60)

    pipe = AdaptiveDiffusionPipeline.from_pretrained(model_id, scheduler_name="ddim")
    extractor = UNetAttentionExtractor(latent_h=64, latent_w=64)
    extractor.hook_with_processor(pipe.unet)

    state = pipe.prepare("a dog playing in a park", num_steps=20, seed=42)

    maps_at_steps = []
    for i in range(5):
        extractor.clear()
        step_out = pipe.denoise_step(state, guidance_scale=7.5)
        attn_map = extractor.get_attention_maps()
        maps_at_steps.append(attn_map.clone())
        pipe.advance_state(state, step_out)

    # Attention maps should change across steps
    diff_01 = (maps_at_steps[0] - maps_at_steps[1]).abs().mean().item()
    diff_04 = (maps_at_steps[0] - maps_at_steps[4]).abs().mean().item()
    print(f"  Mean abs diff step 0→1: {diff_01:.6f}")
    print(f"  Mean abs diff step 0→4: {diff_04:.6f}")
    assert diff_01 > 0, "Attention maps should differ between steps"
    print("[PASS] Attention maps evolve across denoising steps")

    extractor.clear()
    extractor.remove_hooks()


def test_factory():
    """Test the factory function."""
    print("\n" + "=" * 60)
    print("TEST: Attention extractor factory")
    print("=" * 60)

    unet_ext = create_attention_extractor("unet", 64, 64)
    assert isinstance(unet_ext, UNetAttentionExtractor)
    print("[PASS] UNet extractor created")

    try:
        create_attention_extractor("invalid")
        assert False, "Should have raised ValueError"
    except ValueError:
        print("[PASS] Invalid architecture raises ValueError")

    # DiT extractor exists but is not implemented
    from src.diffusion.attention import DiTAttentionExtractor
    dit_ext = create_attention_extractor("dit", 64, 64)
    assert isinstance(dit_ext, DiTAttentionExtractor)
    try:
        dit_ext.hook(None)
        assert False, "Should have raised NotImplementedError"
    except NotImplementedError:
        print("[PASS] DiT extractor raises NotImplementedError (Phase 4 placeholder)")


if __name__ == "__main__":
    model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    if len(sys.argv) > 1:
        model_id = sys.argv[1]

    test_factory()
    test_attention_extraction(model_id)
    test_attention_multiple_steps(model_id)

    print("\n" + "=" * 60)
    print("ALL ATTENTION TESTS PASSED")
    print("=" * 60)
