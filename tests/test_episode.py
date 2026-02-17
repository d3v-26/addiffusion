"""Tests for episode runner.

Run: uv run python tests/test_episode.py
Requires: GPU with SD 1.5 and CLIP weights.
"""

import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.episode import EpisodeRunner, ACTION_CONTINUE, ACTION_STOP, ACTION_REFINE
from src.agent.networks import PolicyNetwork, ValueNetwork
from src.agent.state import StateExtractor
from src.diffusion.pipeline import AdaptiveDiffusionPipeline
from src.diffusion.attention import UNetAttentionExtractor


def _setup(model_id: str):
    """Create all components needed for an episode."""
    pipe = AdaptiveDiffusionPipeline.from_pretrained(model_id, scheduler_name="ddim")
    state_ext = StateExtractor(device="cuda", dtype=torch.float16)
    attn_ext = UNetAttentionExtractor(latent_h=64, latent_w=64)
    attn_ext.hook_with_processor(pipe.unet)

    policy = PolicyNetwork().to("cuda").float()
    value_net = ValueNetwork().to("cuda").float()

    runner = EpisodeRunner(
        pipeline=pipe,
        state_extractor=state_ext,
        attention_extractor=attn_ext,
        warmup_steps=3,
        guidance_scale=7.5,
        refine_k=2,
    )
    return runner, policy, value_net, attn_ext


def test_full_episode(model_id: str = "stable-diffusion-v1-5/stable-diffusion-v1-5"):
    """Test a complete episode with random policy."""
    print("=" * 60)
    print("TEST: Full episode with random policy")
    print("=" * 60)

    runner, policy, value_net, attn_ext = _setup(model_id)

    result = runner.run_episode(
        prompt="a cat sitting on a table",
        policy=policy,
        value_net=value_net,
        num_steps=15,  # Short for testing
        seed=42,
    )

    assert result.final_image is not None, "Should produce a final image"
    assert result.final_image.shape[1] == 3, f"Image should have 3 channels, got {result.final_image.shape}"
    assert len(result.transitions) > 0, "Should have transitions"
    assert result.total_nfe > 0, "Should have consumed NFE"

    print(f"  Transitions: {len(result.transitions)}")
    print(f"  Total NFE: {result.total_nfe}")
    print(f"  Actions: {result.action_sequence}")
    print(f"  Final image shape: {result.final_image.shape}")

    # First 3 actions should be warmup continues (D-16)
    for i in range(min(3, len(result.action_sequence))):
        assert result.action_sequence[i] == "continue", f"Warmup step {i} should be continue, got {result.action_sequence[i]}"
    print("[PASS] First 3 actions are warmup continues (D-16)")

    # Save final image
    img = Image.fromarray((result.final_image[0].permute(1, 2, 0).cpu().float().numpy() * 255).astype("uint8"))
    out_path = Path("outputs/test_episode_full.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    print(f"[PASS] Episode complete, saved to {out_path}")

    attn_ext.clear()
    attn_ext.remove_hooks()


def test_deterministic_episode(model_id: str = "stable-diffusion-v1-5/stable-diffusion-v1-5"):
    """Test deterministic mode (argmax actions)."""
    print("\n" + "=" * 60)
    print("TEST: Deterministic episode")
    print("=" * 60)

    runner, policy, value_net, attn_ext = _setup(model_id)

    r1 = runner.run_episode("a dog", policy=policy, value_net=value_net,
                            num_steps=10, seed=42, deterministic=True)
    r2 = runner.run_episode("a dog", policy=policy, value_net=value_net,
                            num_steps=10, seed=42, deterministic=True)

    assert r1.action_sequence == r2.action_sequence, "Deterministic should give same actions"
    assert r1.total_nfe == r2.total_nfe, "Deterministic should give same NFE"
    print(f"[PASS] Deterministic episodes match: actions={r1.action_sequence}, NFE={r1.total_nfe}")

    attn_ext.clear()
    attn_ext.remove_hooks()


def test_transition_data(model_id: str = "stable-diffusion-v1-5/stable-diffusion-v1-5"):
    """Test that transitions contain valid data for PPO training."""
    print("\n" + "=" * 60)
    print("TEST: Transition data quality")
    print("=" * 60)

    runner, policy, value_net, attn_ext = _setup(model_id)

    result = runner.run_episode(
        prompt="a blue car on a road",
        policy=policy,
        value_net=value_net,
        num_steps=10,
        seed=42,
    )

    for i, t in enumerate(result.transitions):
        assert t.state_features.shape == (1, StateExtractor.TOTAL_DIM), f"Step {i}: bad state shape {t.state_features.shape}"
        assert t.action in [ACTION_CONTINUE, ACTION_STOP, ACTION_REFINE], f"Step {i}: invalid action {t.action}"
        assert isinstance(t.log_prob, float), f"Step {i}: log_prob should be float"
        assert isinstance(t.value, float), f"Step {i}: value should be float"
        assert isinstance(t.done, bool), f"Step {i}: done should be bool"
        assert t.timestep > 0 or i == len(result.transitions) - 1, f"Step {i}: timestep should be > 0"

    # Last transition should be done
    assert result.transitions[-1].done, "Last transition should be done"

    print(f"[PASS] All {len(result.transitions)} transitions have valid data")
    print(f"  State features: {result.transitions[0].state_features.shape}")
    print(f"  Actions: {[t.action for t in result.transitions]}")
    print(f"  Log probs (non-warmup): {[f'{t.log_prob:.3f}' for t in result.transitions[3:]]}")

    attn_ext.clear()
    attn_ext.remove_hooks()


if __name__ == "__main__":
    model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    if len(sys.argv) > 1:
        model_id = sys.argv[1]

    test_full_episode(model_id)
    test_deterministic_episode(model_id)
    test_transition_data(model_id)

    print("\n" + "=" * 60)
    print("ALL EPISODE TESTS PASSED")
    print("=" * 60)
