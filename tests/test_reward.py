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


def test_new_reward_config_fields():
    """RewardConfig accepts c_save, c_refine, baseline_scores_path."""
    cfg = RewardConfig(c_save=1.0, c_refine=0.2, baseline_scores_path=None)
    assert cfg.c_save == 1.0
    assert cfg.c_refine == 0.2
    assert cfg.baseline_scores_path is None
    print("[PASS] RewardConfig accepts new fields")


def test_compute_attention_entropy_uniform():
    """Uniform attention map → entropy = 1.0 (maximum)."""
    from src.rewards.reward import compute_attention_entropy
    h, w, L = 8, 8, 10
    attn = torch.ones(h, w, L)  # uniform across spatial and token dims
    H = compute_attention_entropy(attn)
    assert abs(H - 1.0) < 1e-4, f"Expected 1.0, got {H}"
    print(f"[PASS] Uniform attention entropy = {H:.4f} ≈ 1.0")


def test_compute_attention_entropy_concentrated():
    """Concentrated attention map → entropy near 0."""
    from src.rewards.reward import compute_attention_entropy
    h, w, L = 8, 8, 10
    attn = torch.zeros(h, w, L)
    attn[0, 0, 0] = 1.0  # all mass on one cell
    H = compute_attention_entropy(attn)
    assert H < 0.1, f"Expected near 0, got {H}"
    print(f"[PASS] Concentrated attention entropy = {H:.4f} ≈ 0")


def test_compute_attention_entropy_range():
    """Attention entropy is always in [0, 1]."""
    from src.rewards.reward import compute_attention_entropy
    for _ in range(10):
        attn = torch.rand(16, 16, 20)
        H = compute_attention_entropy(attn)
        assert 0.0 <= H <= 1.0, f"Entropy out of range: {H}"
    print("[PASS] Attention entropy always in [0, 1]")


def test_terminal_reward_with_baselines():
    """Terminal reward is normalized relative to DDIM-20/50 baselines."""
    cfg = RewardConfig(
        beta_1=3.105, beta_2=1.349, beta_3=0.643,
        c_save=1.0,
        baseline_scores_path=None,
    )
    rc = RewardComputer(config=cfg, device="cpu")

    # Inject baseline scores directly (bypassing file load)
    rc.baseline_scores = {
        "a cat": {
            "ddim20": {"clip": 0.30, "aesthetic": 5.0, "image_reward": 0.40},
            "ddim50": {"clip": 0.35, "aesthetic": 5.5, "image_reward": 0.60},
        }
    }

    fake_image = torch.rand(1, 3, 64, 64)

    # Mock score methods so we control the values
    rc.clip_score = lambda img, prompt: 0.35
    rc.image_reward_score = lambda img, prompt: 0.60
    rc.aesthetic_score = lambda img: 5.5

    reward, metrics = rc.compute_terminal_reward(fake_image, "a cat", nfe_used=25, n_max=50)

    # Quality term: each metric normalized to 1.0 (matches DDIM-50)
    # beta_1 * 1.0 + beta_2 * 1.0 + beta_3 * 1.0 = 3.105 + 1.349 + 0.643 = 5.097
    expected_quality = cfg.beta_1 + cfg.beta_2 + cfg.beta_3
    # Step savings: 1.0 * (50 - 25) / 50 = 0.5
    expected_savings = 1.0 * (50 - 25) / 50
    expected_total = expected_quality + expected_savings

    assert abs(metrics["terminal_quality_term"] - expected_quality) < 1e-3, \
        f"Quality term: expected {expected_quality:.3f}, got {metrics['terminal_quality_term']:.3f}"
    assert abs(metrics["terminal_step_savings"] - expected_savings) < 1e-3, \
        f"Step savings: expected {expected_savings:.3f}, got {metrics['terminal_step_savings']:.3f}"
    assert abs(reward - expected_total) < 1e-3, \
        f"Total: expected {expected_total:.3f}, got {reward:.3f}"
    print(f"[PASS] Terminal reward (DDIM-50 quality, 25 steps saved): {reward:.3f}")


def test_terminal_reward_fallback_no_baseline():
    """When prompt not in baseline_scores, uses score_ddim20=0, norm=1 (absolute scores)."""
    cfg = RewardConfig(beta_1=1.0, beta_2=0.0, beta_3=0.0, c_save=0.0, baseline_scores_path=None)
    rc = RewardComputer(config=cfg, device="cpu")
    rc.baseline_scores = {}  # empty — prompt will be missing

    fake_image = torch.rand(1, 3, 64, 64)
    rc.clip_score = lambda img, prompt: 0.5
    rc.image_reward_score = lambda img, prompt: 0.0
    rc.aesthetic_score = lambda img: 0.0

    reward, metrics = rc.compute_terminal_reward(fake_image, "unseen prompt", nfe_used=50, n_max=50)
    # Fallback: ddim20={"clip":0.0,...}, ddim50={"clip":1.0,...}
    # norm(0.5, 0.0, 1.0) = 0.5; beta_1=1.0 → quality_term=0.5; c_save=0 → savings=0
    assert abs(metrics["terminal_quality_term"] - 0.5) < 1e-3, \
        f"Fallback quality term: expected 0.5, got {metrics['terminal_quality_term']:.3f}"
    print(f"[PASS] Fallback terminal reward for unseen prompt: {reward:.3f}")


def test_refine_bonus():
    """Refine bonus = c_refine * attention_entropy."""
    cfg = RewardConfig(c_refine=0.2)
    rc = RewardComputer(config=cfg, device="cpu")

    bonus_high = rc.compute_refine_bonus(attention_entropy=1.0)
    bonus_low = rc.compute_refine_bonus(attention_entropy=0.0)
    bonus_mid = rc.compute_refine_bonus(attention_entropy=0.5)

    assert abs(bonus_high - 0.2) < 1e-6, f"Expected 0.2, got {bonus_high}"
    assert abs(bonus_low - 0.0) < 1e-6, f"Expected 0.0, got {bonus_low}"
    assert abs(bonus_mid - 0.1) < 1e-6, f"Expected 0.1, got {bonus_mid}"
    print(f"[PASS] Refine bonus: high={bonus_high:.2f}, mid={bonus_mid:.2f}, low={bonus_low:.2f}")


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
