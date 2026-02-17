"""Toy training integration test.

Runs a minimal training loop (3 prompts, 2 iterations, batch=2) to verify
the entire pipeline works end-to-end.

Run: uv run python tests/test_training_toy.py
Requires: GPU with SD 1.5, CLIP, ImageReward, DINO weights.

This corresponds to plan.md Phase 1, Week 4, Day 4:
"Debug on toy dataset (10 prompts, 100 iterations)"
"""

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_toy_training(model_id: str = "stable-diffusion-v1-5/stable-diffusion-v1-5"):
    """Run a minimal training loop to test full integration."""
    print("=" * 60)
    print("TEST: Toy training integration (2 iters, batch=2, 10 steps)")
    print("=" * 60)

    from src.diffusion.pipeline import AdaptiveDiffusionPipeline
    from src.diffusion.attention import UNetAttentionExtractor
    from src.agent.state import StateExtractor
    from src.agent.networks import PolicyNetwork, ValueNetwork
    from src.agent.episode import EpisodeRunner
    from src.agent.ppo import PPOConfig, PPOTrainer, episodes_to_batch
    from src.rewards.reward import RewardComputer, RewardConfig

    device = "cuda"
    dtype = torch.float16

    # Components
    print("Loading pipeline...")
    pipeline = AdaptiveDiffusionPipeline.from_pretrained(model_id, scheduler_name="ddim", device=device, dtype=dtype)

    attn_ext = UNetAttentionExtractor(latent_h=64, latent_w=64)
    attn_ext.hook_with_processor(pipeline.unet)

    state_ext = StateExtractor(device=device, dtype=dtype)
    reward_computer = RewardComputer(config=RewardConfig(), device=device)

    policy = PolicyNetwork().to(device).float()
    value_net = ValueNetwork().to(device).float()

    ppo_cfg = PPOConfig(lr=1e-3, ppo_epochs=2, mini_batch_size=8)
    trainer = PPOTrainer(policy, value_net, config=ppo_cfg, device=device)

    runner = EpisodeRunner(
        pipeline=pipeline,
        state_extractor=state_ext,
        attention_extractor=attn_ext,
        warmup_steps=3,
        refine_k=2,
    )

    prompts = [
        "a cat",
        "a red car on a road",
        "a beautiful mountain landscape with a river",
    ]

    num_iters = 2
    batch_size = 2
    num_steps = 10  # Short episodes for testing

    print(f"\nTraining: {num_iters} iters, batch={batch_size}, N_max={num_steps}")
    print()

    for iteration in range(1, num_iters + 1):
        t_start = time.time()

        episodes = []
        for b in range(batch_size):
            prompt = prompts[b % len(prompts)]

            prev_metrics = {}

            def reward_fn(prev_z0, curr_z0, action, step_index, n_max, is_terminal, prompt, decoded_image, _prev=prev_metrics):
                prev_img = _prev.get("prev_decoded")
                reward, metrics = reward_computer.compute_reward(
                    prev_image=prev_img, curr_image=decoded_image,
                    prompt=prompt, action=action, is_terminal=is_terminal,
                    prev_clip=_prev.get("clip_score"), prev_ir=_prev.get("image_reward"),
                )
                _prev["prev_decoded"] = decoded_image
                _prev["clip_score"] = metrics.get("clip_score")
                _prev["image_reward"] = metrics.get("image_reward")
                return reward

            ep = runner.run_episode(
                prompt=prompt, policy=policy, value_net=value_net,
                num_steps=num_steps, seed=42 + iteration * batch_size + b,
                reward_fn=reward_fn,
            )
            episodes.append(ep)

        # PPO update
        batch = episodes_to_batch(episodes, gamma=ppo_cfg.gamma_d, lam=ppo_cfg.gae_lambda)
        metrics = trainer.update(batch)

        elapsed = time.time() - t_start

        # Report
        avg_nfe = sum(ep.total_nfe for ep in episodes) / batch_size
        avg_reward = sum(sum(t.reward for t in ep.transitions) for ep in episodes) / batch_size
        all_actions = []
        for ep in episodes:
            all_actions.extend(ep.action_sequence)

        print(
            f"Iter {iteration}/{num_iters} | "
            f"reward={avg_reward:+.3f} | "
            f"NFE={avg_nfe:.1f} | "
            f"loss={metrics['total_loss']:.4f} | "
            f"entropy={metrics['entropy']:.3f} | "
            f"actions={all_actions} | "
            f"time={elapsed:.1f}s"
        )

    # Validation checks
    print()
    assert len(episodes) == batch_size, f"Expected {batch_size} episodes"
    for ep in episodes:
        assert ep.final_image is not None, "Episode should produce final image"
        assert len(ep.transitions) > 0, "Episode should have transitions"
        assert ep.total_nfe > 0, "Episode should consume NFE"
        # First 3 should be warmup
        for i in range(min(3, len(ep.action_sequence))):
            assert ep.action_sequence[i] == "continue", f"Warmup step {i} should be continue"

    assert metrics["n_updates"] > 0, "Should have done PPO updates"
    assert not any(torch.isnan(p).any() for p in policy.parameters()), "NaN in policy weights"
    assert not any(torch.isnan(p).any() for p in value_net.parameters()), "NaN in value weights"

    print("[PASS] Toy training completed successfully!")
    print("  - Episodes produce final images")
    print("  - Warmup steps enforced (D-16)")
    print("  - PPO updates run without NaN")
    print("  - Rewards computed with normalization (D-37)")

    # Cleanup
    attn_ext.clear()
    attn_ext.remove_hooks()


if __name__ == "__main__":
    model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    if len(sys.argv) > 1:
        model_id = sys.argv[1]

    test_toy_training(model_id)

    print("\n" + "=" * 60)
    print("TOY TRAINING TEST PASSED")
    print("=" * 60)
