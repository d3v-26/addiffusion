"""Tests for the step-by-step diffusion pipeline wrapper.

Run: uv run python tests/test_pipeline.py
Requires: GPU with SD 1.5 weights available.
"""

import sys
import time
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.diffusion.pipeline import AdaptiveDiffusionPipeline, PipelineState, StepOutput


def test_pipeline_basic(model_id: str = "stable-diffusion-v1-5/stable-diffusion-v1-5"):
    """Test basic pipeline: prepare, step, decode."""
    print("=" * 60)
    print("TEST: Pipeline basic step-by-step denoising")
    print("=" * 60)

    pipe = AdaptiveDiffusionPipeline.from_pretrained(model_id, scheduler_name="ddim")

    # Check latent dims (D-14: latent space, not pixel space)
    h, w = pipe.get_latent_dims(512, 512)
    assert h == 64 and w == 64, f"Expected 64x64 latent, got {h}x{w}"
    print(f"[PASS] Latent dims: {h}x{w} (correct for 512x512 pixel space)")

    # Prepare episode
    prompt = "a photo of a cat sitting on a windowsill"
    num_steps = 20
    state = pipe.prepare(prompt, num_steps=num_steps, seed=42)

    assert state.z_t.shape == (1, 4, 64, 64), f"Wrong z_t shape: {state.z_t.shape}"
    assert len(state.timesteps) == num_steps, f"Expected {num_steps} timesteps, got {len(state.timesteps)}"
    assert state.step_index == 0
    assert state.total_nfe == 0
    assert not state.is_done
    print(f"[PASS] State initialized: z_t shape={state.z_t.shape}, timesteps={len(state.timesteps)}")

    # Run all steps
    t_start = time.time()
    for i in range(num_steps):
        step_out = pipe.denoise_step(state, guidance_scale=7.5)

        assert step_out.z_t.shape == (1, 4, 64, 64), f"Step {i}: wrong z_t shape"
        assert step_out.z0_pred.shape == (1, 4, 64, 64), f"Step {i}: wrong z0_pred shape"
        assert step_out.nfe == 1, f"Step {i}: NFE should be 1, got {step_out.nfe}"
        assert step_out.step_index == i

        pipe.advance_state(state, step_out)

    elapsed = time.time() - t_start
    assert state.is_done, "Should be done after all steps"
    assert state.total_nfe == num_steps, f"Expected NFE={num_steps}, got {state.total_nfe}"
    print(f"[PASS] Completed {num_steps} steps in {elapsed:.1f}s (NFE={state.total_nfe})")

    # Decode final image
    last_z0 = state.history[-1].z0_pred
    image_tensor = pipe.decode(last_z0)
    assert image_tensor.shape[0] == 1 and image_tensor.shape[1] == 3, f"Bad image shape: {image_tensor.shape}"
    assert image_tensor.min() >= 0 and image_tensor.max() <= 1, "Image not in [0,1]"

    # Save for visual inspection
    img = Image.fromarray((image_tensor[0].permute(1, 2, 0).cpu().float().numpy() * 255).astype("uint8"))
    out_path = Path("outputs/test_pipeline_basic.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    print(f"[PASS] Decoded image: {image_tensor.shape}, saved to {out_path}")


def test_early_stop(model_id: str = "stable-diffusion-v1-5/stable-diffusion-v1-5"):
    """Test early stopping: run 10/50 steps, decode intermediate prediction."""
    print("\n" + "=" * 60)
    print("TEST: Early stopping at step 10/50")
    print("=" * 60)

    pipe = AdaptiveDiffusionPipeline.from_pretrained(model_id, scheduler_name="ddim")
    state = pipe.prepare("a beautiful sunset over the ocean", num_steps=50, seed=42)

    stop_step = 10
    for i in range(stop_step):
        step_out = pipe.denoise_step(state, guidance_scale=7.5)
        pipe.advance_state(state, step_out)

    assert not state.is_done, "Should not be done after 10/50 steps"
    assert state.total_nfe == stop_step

    # Decode the one-step prediction at stop point
    z0_at_stop = state.history[-1].z0_pred
    image_tensor = pipe.decode(z0_at_stop)

    img = Image.fromarray((image_tensor[0].permute(1, 2, 0).cpu().float().numpy() * 255).astype("uint8"))
    out_path = Path("outputs/test_pipeline_early_stop.png")
    img.save(out_path)
    print(f"[PASS] Early stop at step {stop_step}, NFE={state.total_nfe}, saved to {out_path}")


def test_add_noise(model_id: str = "stable-diffusion-v1-5/stable-diffusion-v1-5"):
    """Test re-noising for post-refinement state transition (D-19)."""
    print("\n" + "=" * 60)
    print("TEST: Add noise (re-noising for D-19)")
    print("=" * 60)

    pipe = AdaptiveDiffusionPipeline.from_pretrained(model_id, scheduler_name="ddim")
    state = pipe.prepare("a red ball on a green table", num_steps=50, seed=42)

    # Run a few steps
    for _ in range(5):
        step_out = pipe.denoise_step(state, guidance_scale=7.5)
        pipe.advance_state(state, step_out)

    z0_pred = state.history[-1].z0_pred

    # Re-noise to next timestep (simulates post-refine transition)
    next_t = state.timesteps[state.step_index].item()
    z_renoised = pipe.add_noise(z0_pred, next_t)

    assert z_renoised.shape == z0_pred.shape, f"Shape mismatch: {z_renoised.shape} vs {z0_pred.shape}"
    # Re-noised should be different from clean prediction
    diff = (z_renoised - z0_pred).abs().mean().item()
    assert diff > 0.01, f"Re-noised latent too similar to clean (diff={diff:.4f})"
    print(f"[PASS] Re-noised latent: mean abs diff from clean = {diff:.4f}")


def test_schedulers(model_id: str = "stable-diffusion-v1-5/stable-diffusion-v1-5"):
    """Test that multiple schedulers work."""
    print("\n" + "=" * 60)
    print("TEST: Multiple schedulers")
    print("=" * 60)

    for sched_name in ["ddim", "dpm_solver", "euler"]:
        pipe = AdaptiveDiffusionPipeline.from_pretrained(
            model_id, scheduler_name=sched_name
        )
        state = pipe.prepare("a house", num_steps=10, seed=42)
        for _ in range(10):
            step_out = pipe.denoise_step(state)
            pipe.advance_state(state, step_out)

        assert state.is_done
        assert state.total_nfe == 10
        print(f"  [PASS] {sched_name}: 10 steps completed, NFE={state.total_nfe}")

    print("[PASS] All schedulers work")


def test_deterministic(model_id: str = "stable-diffusion-v1-5/stable-diffusion-v1-5"):
    """Test that same seed produces same output."""
    print("\n" + "=" * 60)
    print("TEST: Deterministic with same seed")
    print("=" * 60)

    pipe = AdaptiveDiffusionPipeline.from_pretrained(model_id, scheduler_name="ddim")

    results = []
    for run in range(2):
        state = pipe.prepare("a dog", num_steps=10, seed=42)
        for _ in range(10):
            step_out = pipe.denoise_step(state)
            pipe.advance_state(state, step_out)
        results.append(state.history[-1].z0_pred)

    diff = (results[0] - results[1]).abs().max().item()
    assert diff < 1e-3, f"Non-deterministic: max diff = {diff}"
    print(f"[PASS] Deterministic: max diff between runs = {diff:.6f}")


if __name__ == "__main__":
    model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    if len(sys.argv) > 1:
        model_id = sys.argv[1]

    print(f"Using model: {model_id}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    print()

    test_pipeline_basic(model_id)
    test_early_stop(model_id)
    test_add_noise(model_id)
    test_schedulers(model_id)
    test_deterministic(model_id)

    print("\n" + "=" * 60)
    print("ALL PIPELINE TESTS PASSED")
    print("=" * 60)
