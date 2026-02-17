"""Tests for state feature extraction.

Run: uv run python tests/test_state.py
Requires: GPU with CLIP ViT-L-14 weights available.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.state import StateExtractor


def test_dimensions():
    """Test that feature dimensions match the specification."""
    print("=" * 60)
    print("TEST: State feature dimensions")
    print("=" * 60)

    ext = StateExtractor(device="cuda", dtype=torch.float16)

    # Synthetic decoded image
    image = torch.rand(1, 3, 512, 512, device="cuda")

    phi = ext.extract(
        decoded_image=image,
        prompt="a photo of a cat",
        timestep=500,
        clip_score=0.25,
        step_ratio=0.5,
        nfe_ratio=0.4,
    )

    assert phi.shape == (1, StateExtractor.TOTAL_DIM), f"Expected (1, {StateExtractor.TOTAL_DIM}), got {phi.shape}"
    assert phi.dtype == torch.float32, f"Features should be float32, got {phi.dtype}"
    assert not phi.isnan().any(), "Features contain NaN"
    assert not phi.isinf().any(), "Features contain Inf"
    print(f"[PASS] phi(s) shape: {phi.shape} (expected {StateExtractor.TOTAL_DIM})")
    print(f"  Value range: [{phi.min():.4f}, {phi.max():.4f}]")


def test_clip_image_encoding():
    """Test CLIP image encoding."""
    print("\n" + "=" * 60)
    print("TEST: CLIP image encoding")
    print("=" * 60)

    ext = StateExtractor(device="cuda", dtype=torch.float16)
    image = torch.rand(1, 3, 512, 512, device="cuda")

    features = ext.encode_image(image)
    assert features.shape == (1, 768), f"Expected (1, 768), got {features.shape}"

    # Should be approximately unit norm (L2-normalized)
    norm = features.norm(dim=-1).item()
    assert abs(norm - 1.0) < 0.01, f"CLIP features should be unit norm, got {norm}"
    print(f"[PASS] CLIP image features: shape={features.shape}, norm={norm:.4f}")


def test_clip_text_caching():
    """Test that text features are cached per prompt."""
    print("\n" + "=" * 60)
    print("TEST: CLIP text feature caching")
    print("=" * 60)

    ext = StateExtractor(device="cuda", dtype=torch.float16)

    f1 = ext.encode_text("a photo of a cat")
    f2 = ext.encode_text("a photo of a cat")  # should be cached
    assert torch.equal(f1, f2), "Cached features should be identical"
    print("[PASS] Same prompt returns cached features")

    f3 = ext.encode_text("a photo of a dog")
    diff = (f1 - f3).abs().mean().item()
    assert diff > 0.01, f"Different prompts should have different features (diff={diff})"
    print(f"[PASS] Different prompts produce different features (diff={diff:.4f})")

    ext.reset_cache()
    assert ext._cached_text_features is None
    print("[PASS] Cache reset works")


def test_timestep_embedding():
    """Test sinusoidal timestep embedding."""
    print("\n" + "=" * 60)
    print("TEST: Timestep embedding")
    print("=" * 60)

    ext = StateExtractor(device="cuda")

    emb_0 = ext.encode_timestep(0)
    emb_500 = ext.encode_timestep(500)
    emb_999 = ext.encode_timestep(999)

    assert emb_0.shape == (1, 128), f"Expected (1, 128), got {emb_0.shape}"
    print(f"[PASS] Timestep embedding shape: {emb_0.shape}")

    # Different timesteps should produce different embeddings
    diff_0_500 = (emb_0 - emb_500).abs().mean().item()
    diff_0_999 = (emb_0 - emb_999).abs().mean().item()
    assert diff_0_500 > 0.01, "Timestep 0 and 500 should differ"
    assert diff_0_999 > 0.01, "Timestep 0 and 999 should differ"
    print(f"[PASS] Embeddings differ: t=0 vs t=500 diff={diff_0_500:.4f}, t=0 vs t=999 diff={diff_0_999:.4f}")


def test_quality_vector():
    """Test quality vector construction."""
    print("\n" + "=" * 60)
    print("TEST: Quality vector")
    print("=" * 60)

    ext = StateExtractor(device="cuda")

    q = ext.build_quality_vector(
        clip_score=0.3, aesthetic_delta=0.01, image_reward_delta=-0.05,
        dino_similarity=0.95, step_ratio=0.4, nfe_ratio=0.3,
    )
    assert q.shape == (1, 8), f"Expected (1, 8), got {q.shape}"
    assert q[0, 0].item() == 0.3  # clip_score
    assert q[0, 3].item() == 0.95  # dino_similarity (not a delta — D-11)
    print(f"[PASS] Quality vector: shape={q.shape}, values={q.tolist()}")


def test_full_pipeline_integration(model_id: str = "stable-diffusion-v1-5/stable-diffusion-v1-5"):
    """Test state extraction integrated with the diffusion pipeline."""
    print("\n" + "=" * 60)
    print("TEST: Full pipeline integration")
    print("=" * 60)

    from src.diffusion.pipeline import AdaptiveDiffusionPipeline

    pipe = AdaptiveDiffusionPipeline.from_pretrained(model_id, scheduler_name="ddim")
    ext = StateExtractor(device="cuda", dtype=torch.float16)

    prompt = "a golden retriever playing in the park"
    state = pipe.prepare(prompt, num_steps=20, seed=42)

    # Run a few steps and extract state features
    for i in range(5):
        step_out = pipe.denoise_step(state, guidance_scale=7.5)
        pipe.advance_state(state, step_out)

        # Decode one-step prediction for CLIP features
        x0_hat = pipe.decode(step_out.z0_pred)

        phi = ext.extract(
            decoded_image=x0_hat,
            prompt=prompt,
            timestep=step_out.timestep,
            step_ratio=state.step_index / len(state.timesteps),
            nfe_ratio=state.total_nfe / len(state.timesteps),
        )

        print(f"  Step {i}: phi shape={phi.shape}, norm={phi.norm():.2f}, timestep={step_out.timestep}")
        assert phi.shape == (1, StateExtractor.TOTAL_DIM)

    print(f"[PASS] State features extracted for 5 steps with diffusion pipeline")


if __name__ == "__main__":
    model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    if len(sys.argv) > 1:
        model_id = sys.argv[1]

    test_timestep_embedding()
    test_quality_vector()
    test_clip_image_encoding()
    test_clip_text_caching()
    test_dimensions()
    test_full_pipeline_integration(model_id)

    print("\n" + "=" * 60)
    print("ALL STATE EXTRACTION TESTS PASSED")
    print("=" * 60)
