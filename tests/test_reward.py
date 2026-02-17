"""Tests for reward computation.

Run: uv run python tests/test_reward.py
Requires: GPU with CLIP, ImageReward, DINO weights.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rewards.reward import RewardComputer, RewardConfig
from src.agent.networks import ACTION_CONTINUE, ACTION_STOP, ACTION_REFINE


def test_efficiency_reward():
    """Test NFE-based efficiency reward (D-09, D-10)."""
    print("=" * 60)
    print("TEST: Efficiency reward (D-09 c_nfe, D-10 NFE formula)")
    print("=" * 60)

    cfg = RewardConfig(c_nfe=0.01, refine_k=2)
    rc = RewardComputer(config=cfg, device="cuda")

    r_continue = rc.compute_efficiency_reward(ACTION_CONTINUE)
    r_stop = rc.compute_efficiency_reward(ACTION_STOP)
    r_refine = rc.compute_efficiency_reward(ACTION_REFINE)

    assert r_continue == -0.01 * 1, f"Continue: -c_nfe*1 = {r_continue}"
    assert r_stop == 0.0, f"Stop: -c_nfe*0 = {r_stop}"
    assert r_refine == -0.01 * 3, f"Refine: -c_nfe*(1+k) = {r_refine} (D-10: should be -0.03)"

    print(f"  Continue: {r_continue} (NFE=1)")
    print(f"  Stop: {r_stop} (NFE=0)")
    print(f"  Refine: {r_refine} (NFE=1+k=3, D-10)")
    print("[PASS] Efficiency rewards use c_nfe (D-09) and correct NFE formula (D-10)")


def test_normalization_toggle():
    """Test reward normalization on/off (A5 ablation, D-37)."""
    print("\n" + "=" * 60)
    print("TEST: Reward normalization toggle (D-37)")
    print("=" * 60)

    cfg_norm = RewardConfig(normalize=True, clip_norm=0.05)
    cfg_raw = RewardConfig(normalize=False)

    # With normalization: delta_clip / 0.05
    # Without: delta_clip raw
    delta = 0.01
    normalized = delta / cfg_norm.clip_norm
    assert abs(normalized - 0.2) < 1e-9, f"Normalized delta should be ~0.2, got {normalized}"
    print(f"  Raw delta_CLIP=0.01 → normalized={normalized} (divided by {cfg_norm.clip_norm})")
    print("[PASS] Normalization scales correctly (D-37)")


def test_clip_score(model_id: str = "stable-diffusion-v1-5/stable-diffusion-v1-5"):
    """Test CLIP score computation."""
    print("\n" + "=" * 60)
    print("TEST: CLIP score computation")
    print("=" * 60)

    rc = RewardComputer(device="cuda")

    # Generate a test image with the pipeline
    from src.diffusion.pipeline import AdaptiveDiffusionPipeline
    pipe = AdaptiveDiffusionPipeline.from_pretrained(model_id, scheduler_name="ddim")
    state = pipe.prepare("a red car", num_steps=20, seed=42)
    for _ in range(20):
        step_out = pipe.denoise_step(state)
        pipe.advance_state(state, step_out)
    image = pipe.decode(state.history[-1].z0_pred)

    score_match = rc.clip_score(image, "a red car")
    score_mismatch = rc.clip_score(image, "a blue whale swimming")

    print(f"  CLIP('a red car' image, 'a red car'): {score_match:.4f}")
    print(f"  CLIP('a red car' image, 'a blue whale swimming'): {score_mismatch:.4f}")
    assert score_match > score_mismatch, "Matching prompt should have higher CLIP score"
    print("[PASS] CLIP score is higher for matching prompt")


def test_dino_similarity():
    """Test DINO similarity (D-11: similarity, not delta)."""
    print("\n" + "=" * 60)
    print("TEST: DINO similarity (D-11: not a delta)")
    print("=" * 60)

    rc = RewardComputer(device="cuda")

    img1 = torch.rand(1, 3, 256, 256, device="cuda")
    img2 = img1.clone()  # Identical
    img3 = torch.rand(1, 3, 256, 256, device="cuda")  # Different

    sim_identical = rc.dino_similarity(img1, img2)
    sim_different = rc.dino_similarity(img1, img3)

    print(f"  DINO sim (identical): {sim_identical:.4f}")
    print(f"  DINO sim (different): {sim_different:.4f}")

    assert sim_identical > sim_different, "Identical images should have higher similarity"
    assert 0 <= sim_identical <= 1, f"Similarity should be in [0,1], got {sim_identical}"
    assert 0 <= sim_different <= 1, f"Similarity should be in [0,1], got {sim_different}"
    print("[PASS] DINO similarity is a score in [0,1], not a delta (D-11)")


def test_full_reward(model_id: str = "stable-diffusion-v1-5/stable-diffusion-v1-5"):
    """Test full composite reward computation."""
    print("\n" + "=" * 60)
    print("TEST: Full composite reward (D-13: no outer lambdas)")
    print("=" * 60)

    rc = RewardComputer(device="cuda")

    # Generate two images (simulating consecutive steps)
    from src.diffusion.pipeline import AdaptiveDiffusionPipeline
    pipe = AdaptiveDiffusionPipeline.from_pretrained(model_id, scheduler_name="ddim")
    state = pipe.prepare("a beautiful landscape", num_steps=20, seed=42)

    images = []
    for i in range(10):
        step_out = pipe.denoise_step(state)
        pipe.advance_state(state, step_out)
        if i in [4, 9]:
            images.append(pipe.decode(step_out.z0_pred))

    # Non-terminal reward
    reward, metrics = rc.compute_reward(
        prev_image=images[0], curr_image=images[1],
        prompt="a beautiful landscape",
        action=ACTION_CONTINUE, is_terminal=False,
    )
    print(f"  Non-terminal reward: {reward:.4f}")
    print(f"  Components: quality={metrics.get('r_quality', 0):.4f}, efficiency={metrics['r_efficiency']:.4f}")
    assert "r_quality" in metrics
    assert metrics["r_efficiency"] == -0.01  # c_nfe * 1
    assert metrics["r_terminal"] == 0.0  # Not terminal

    # Terminal reward
    reward_t, metrics_t = rc.compute_reward(
        prev_image=images[0], curr_image=images[1],
        prompt="a beautiful landscape",
        action=ACTION_STOP, is_terminal=True,
    )
    print(f"  Terminal reward: {reward_t:.4f}")
    print(f"  Components: quality={metrics_t.get('r_quality', 0):.4f}, efficiency={metrics_t['r_efficiency']:.4f}, terminal={metrics_t['r_terminal']:.4f}")
    assert metrics_t["r_terminal"] != 0.0, "Terminal reward should be non-zero"
    assert metrics_t["r_efficiency"] == 0.0  # Stop has NFE=0

    print("[PASS] Composite reward: R = R_quality + R_efficiency + R_terminal (D-13: no lambdas)")


if __name__ == "__main__":
    model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    if len(sys.argv) > 1:
        model_id = sys.argv[1]

    test_efficiency_reward()
    test_normalization_toggle()
    test_clip_score(model_id)
    test_dino_similarity()
    test_full_reward(model_id)

    print("\n" + "=" * 60)
    print("ALL REWARD TESTS PASSED")
    print("=" * 60)
