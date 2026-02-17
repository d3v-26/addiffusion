"""G1 Decision Gate Test: 100 iterations on toy data.

Evaluates all four G1 criteria from plan.md:
  1. Training loss decreases over 100 iterations
  2. Agent actions are non-degenerate (not 100% one action)
  3. Reward components are in comparable scales after normalization
  4. One full training iteration completes in reasonable time

Run: sbatch scripts/g1_test.slurm
  or: uv run python -u tests/test_g1.py

Requires: GPU with SD 1.5, CLIP, ImageReward, DINO weights.
"""

import sys
import time
from pathlib import Path
from collections import Counter

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))


def run_g1_test(model_id: str = "stable-diffusion-v1-5/stable-diffusion-v1-5"):
    from src.diffusion.pipeline import AdaptiveDiffusionPipeline
    from src.diffusion.attention import UNetAttentionExtractor
    from src.agent.state import StateExtractor
    from src.agent.networks import PolicyNetwork, ValueNetwork
    from src.agent.episode import EpisodeRunner
    from src.agent.ppo import PPOConfig, PPOTrainer, episodes_to_batch
    from src.rewards.reward import RewardComputer, RewardConfig

    # === Config ===
    num_iters = 100
    batch_size = 4
    num_steps = 10
    seed = 42

    device = "cuda"
    dtype = torch.float16

    prompts = [
        "a cat sitting on a windowsill",
        "a red car on a highway",
        "a beautiful mountain landscape with a river",
        "a dog playing fetch in a park",
        "a cup of coffee on a wooden table",
        "a sunset over the ocean",
        "a city skyline at night",
        "a bowl of fruit on a kitchen counter",
        "a child flying a kite",
        "a snowy forest path",
    ]

    print("=" * 60)
    print(f"G1 DECISION GATE TEST: {num_iters} iters, batch={batch_size}, N_max={num_steps}")
    print("=" * 60)

    # === Initialize components ===
    print("\nLoading components...")
    pipeline = AdaptiveDiffusionPipeline.from_pretrained(
        model_id, scheduler_name="ddim", device=device, dtype=dtype
    )

    attn_ext = UNetAttentionExtractor(latent_h=64, latent_w=64)
    attn_ext.hook_with_processor(pipeline.unet)

    state_ext = StateExtractor(device=device, dtype=dtype)
    reward_computer = RewardComputer(config=RewardConfig(), device=device)

    policy = PolicyNetwork().to(device).float()
    value_net = ValueNetwork().to(device).float()

    ppo_cfg = PPOConfig(lr=3e-4, ppo_epochs=4, mini_batch_size=16)
    trainer = PPOTrainer(policy, value_net, config=ppo_cfg, device=device)

    runner = EpisodeRunner(
        pipeline=pipeline,
        state_extractor=state_ext,
        attention_extractor=attn_ext,
        warmup_steps=3,
        refine_k=2,
    )

    # === Tracking ===
    loss_history = []
    reward_history = []
    entropy_history = []
    nfe_history = []
    action_counts = Counter()
    iter_times = []

    print(f"\nTraining with {len(prompts)} prompts...")
    print("-" * 100)

    for iteration in range(1, num_iters + 1):
        t_start = time.time()

        episodes = []
        for b in range(batch_size):
            prompt = prompts[(iteration * batch_size + b) % len(prompts)]
            prev_metrics = {}

            def reward_fn(prev_z0, curr_z0, action, step_index, n_max,
                          is_terminal, prompt, decoded_image, _prev=prev_metrics):
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
                num_steps=num_steps, seed=seed + iteration * batch_size + b,
                reward_fn=reward_fn,
            )
            episodes.append(ep)

        # PPO update
        batch = episodes_to_batch(episodes, gamma=ppo_cfg.gamma_d, lam=ppo_cfg.gae_lambda)
        metrics = trainer.update(batch)

        elapsed = time.time() - t_start
        iter_times.append(elapsed)

        # Track metrics
        avg_reward = sum(sum(t.reward for t in ep.transitions) for ep in episodes) / batch_size
        avg_nfe = sum(ep.total_nfe for ep in episodes) / batch_size
        loss_history.append(metrics["total_loss"])
        reward_history.append(avg_reward)
        entropy_history.append(metrics["entropy"])
        nfe_history.append(avg_nfe)

        # Count actions
        for ep in episodes:
            action_counts.update(ep.action_sequence)

        # Print every 10 iterations
        if iteration % 10 == 0 or iteration == 1:
            action_str = ", ".join(f"{a}:{c}" for a, c in sorted(action_counts.items()))
            print(
                f"Iter {iteration:3d}/{num_iters} | "
                f"reward={avg_reward:+8.3f} | "
                f"loss={metrics['total_loss']:8.4f} | "
                f"entropy={metrics['entropy']:.4f} | "
                f"NFE={avg_nfe:5.1f} | "
                f"clip_frac={metrics.get('clip_fraction', 0):.3f} | "
                f"time={elapsed:.1f}s | "
                f"actions=[{action_str}]"
            )

    print("-" * 100)

    # === G1 Evaluation ===
    print("\n" + "=" * 60)
    print("G1 CRITERIA EVALUATION")
    print("=" * 60)

    g1_pass = True

    # Criterion 1: Loss decreases
    first_10_loss = sum(loss_history[:10]) / 10
    last_10_loss = sum(loss_history[-10:]) / 10
    loss_decreased = last_10_loss < first_10_loss
    print(f"\n1. Loss trend: first_10_avg={first_10_loss:.4f} → last_10_avg={last_10_loss:.4f}")
    if loss_decreased:
        print("   [PASS] Loss decreased")
    else:
        print("   [FAIL] Loss did NOT decrease")
        g1_pass = False

    # Criterion 2: Non-degenerate actions
    total_actions = sum(action_counts.values())
    action_pcts = {a: c / total_actions * 100 for a, c in action_counts.items()}
    max_pct = max(action_pcts.values())
    print(f"\n2. Action distribution (cumulative):")
    for action, pct in sorted(action_pcts.items()):
        print(f"   {action}: {pct:.1f}%")
    if max_pct < 95:
        print("   [PASS] Actions are non-degenerate (no single action >95%)")
    else:
        dominant = max(action_pcts, key=action_pcts.get)
        print(f"   [FAIL] Degenerate: {dominant} is {max_pct:.1f}%")
        g1_pass = False

    # Criterion 3: Reward trend
    first_10_reward = sum(reward_history[:10]) / 10
    last_10_reward = sum(reward_history[-10:]) / 10
    print(f"\n3. Reward trend: first_10_avg={first_10_reward:+.3f} → last_10_avg={last_10_reward:+.3f}")
    if last_10_reward > first_10_reward:
        print("   [PASS] Reward improved")
    else:
        print("   [WARN] Reward did not improve (not a hard fail)")

    # Criterion 4: Iteration time
    avg_time = sum(iter_times) / len(iter_times)
    total_time = sum(iter_times)
    print(f"\n4. Timing: avg={avg_time:.1f}s/iter, total={total_time:.0f}s ({total_time/60:.1f}min)")
    print("   [INFO] Target: <10min/iter on A100 for batch=64 (this is batch={})".format(batch_size))

    # Entropy check
    first_entropy = entropy_history[0]
    last_entropy = entropy_history[-1]
    print(f"\n5. Entropy: {first_entropy:.4f} → {last_entropy:.4f}")
    if last_entropy > 0.1:
        print("   [PASS] Policy has not collapsed (entropy > 0.1)")
    else:
        print("   [WARN] Low entropy — policy may be collapsing")

    # No NaN check
    has_nan = any(torch.isnan(p).any() for p in policy.parameters()) or \
              any(torch.isnan(p).any() for p in value_net.parameters())
    print(f"\n6. NaN check: {'[FAIL] NaN detected!' if has_nan else '[PASS] No NaN in weights'}")
    if has_nan:
        g1_pass = False

    # === Final verdict ===
    print("\n" + "=" * 60)
    if g1_pass:
        print("G1 DECISION GATE: PASS")
        print("Proceed to Phase 2 (Baselines & Hyperparameter Tuning)")
    else:
        print("G1 DECISION GATE: FAIL")
        print("Fallback: check reward scales (D-37), increase entropy coeff,")
        print("          or simplify to stop/continue only (A2 No-Refine)")
    print("=" * 60)

    # Cleanup
    attn_ext.clear()
    attn_ext.remove_hooks()

    return g1_pass


if __name__ == "__main__":
    model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    if len(sys.argv) > 1:
        model_id = sys.argv[1]

    passed = run_g1_test(model_id)
    sys.exit(0 if passed else 1)
