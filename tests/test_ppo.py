"""Tests for PPO training components.

Run: uv run python tests/test_ppo.py
No GPU required for unit tests. GPU required for integration test.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.networks import PolicyNetwork, ValueNetwork, ACTION_CONTINUE, ACTION_STOP
from src.agent.ppo import PPOConfig, PPOTrainer, compute_gae, episodes_to_batch, TrajectoryBatch
from src.agent.episode import EpisodeResult, Transition
from src.agent.state import StateExtractor


def test_gae():
    """Test Generalized Advantage Estimation."""
    print("=" * 60)
    print("TEST: GAE computation")
    print("=" * 60)

    rewards = [1.0, 1.0, 1.0, 1.0, 1.0]
    values = [0.5, 0.5, 0.5, 0.5, 0.5]
    dones = [False, False, False, False, True]

    advantages, returns = compute_gae(rewards, values, dones, gamma=0.99, lam=0.95)

    assert len(advantages) == 5
    assert len(returns) == 5
    # Returns should be advantages + values
    for i in range(5):
        assert abs(returns[i] - (advantages[i] + values[i])) < 1e-6
    print(f"  Advantages: {[f'{a:.4f}' for a in advantages]}")
    print(f"  Returns: {[f'{r:.4f}' for r in returns]}")

    # Last advantage should be just delta (no future)
    # delta = r + gamma * 0 (done) - V = 1.0 - 0.5 = 0.5
    assert abs(advantages[-1] - 0.5) < 1e-6, f"Last advantage should be 0.5, got {advantages[-1]}"
    print("[PASS] GAE correctly computed with terminal state")


def test_gae_discount():
    """Test that gamma_d discounting works correctly."""
    print("\n" + "=" * 60)
    print("TEST: GAE discounting (D-09: gamma_d)")
    print("=" * 60)

    rewards = [0.0, 0.0, 0.0, 0.0, 10.0]
    values = [0.0, 0.0, 0.0, 0.0, 0.0]
    dones = [False, False, False, False, True]

    adv_high, _ = compute_gae(rewards, values, dones, gamma=0.99, lam=0.95)
    adv_low, _ = compute_gae(rewards, values, dones, gamma=0.5, lam=0.95)

    # Higher discount should propagate reward further back
    assert adv_high[0] > adv_low[0], "Higher gamma should propagate more reward to early steps"
    print(f"  gamma=0.99: first adv={adv_high[0]:.4f}")
    print(f"  gamma=0.50: first adv={adv_low[0]:.4f}")
    print("[PASS] Discount factor (gamma_d) affects advantage propagation")


def test_episodes_to_batch():
    """Test converting episodes to a training batch."""
    print("\n" + "=" * 60)
    print("TEST: Episodes to batch conversion")
    print("=" * 60)

    state_dim = StateExtractor.TOTAL_DIM

    # Create synthetic episodes
    episodes = []
    for _ in range(3):
        ep = EpisodeResult(prompt="test")
        for i in range(5):
            t = Transition(
                state_features=torch.randn(1, state_dim),
                action=ACTION_CONTINUE if i < 4 else ACTION_STOP,
                log_prob=-0.5,
                value=0.1 * i,
                reward=0.1,
                done=(i == 4),
                nfe=1,
                timestep=1000 - i * 200,
            )
            ep.transitions.append(t)
        episodes.append(ep)

    batch = episodes_to_batch(episodes, gamma=0.99, lam=0.95)

    assert batch.states.shape == (15, state_dim), f"Expected (15, {state_dim}), got {batch.states.shape}"
    assert batch.actions.shape == (15,)
    assert batch.old_log_probs.shape == (15,)
    assert batch.returns.shape == (15,)
    assert batch.advantages.shape == (15,)

    # Advantages should be normalized
    assert abs(batch.advantages.mean()) < 0.1, f"Advantages should be ~zero-mean after normalization"
    print(f"[PASS] Batch: states={batch.states.shape}, actions={batch.actions.shape}")
    print(f"  Advantages: mean={batch.advantages.mean():.4f}, std={batch.advantages.std():.4f}")


def test_ppo_update():
    """Test a single PPO update step."""
    print("\n" + "=" * 60)
    print("TEST: PPO update step")
    print("=" * 60)

    state_dim = StateExtractor.TOTAL_DIM
    policy = PolicyNetwork(input_dim=state_dim).float()
    value_net = ValueNetwork(input_dim=state_dim).float()

    cfg = PPOConfig(lr=1e-3, ppo_epochs=2, mini_batch_size=8)
    trainer = PPOTrainer(policy, value_net, config=cfg, device="cpu")

    # Create synthetic batch
    n = 32
    batch = TrajectoryBatch(
        states=torch.randn(n, state_dim),
        actions=torch.randint(0, 3, (n,)),
        old_log_probs=torch.randn(n) * 0.5,
        returns=torch.randn(n),
        advantages=torch.randn(n),
        values=torch.randn(n),
    )

    # Run update
    metrics = trainer.update(batch)

    assert "policy_loss" in metrics
    assert "value_loss" in metrics
    assert "entropy" in metrics
    assert "total_loss" in metrics
    assert "clip_fraction" in metrics
    assert metrics["n_updates"] > 0

    print(f"  Policy loss: {metrics['policy_loss']:.4f}")
    print(f"  Value loss: {metrics['value_loss']:.4f}")
    print(f"  Entropy: {metrics['entropy']:.4f}")
    print(f"  Clip fraction: {metrics['clip_fraction']:.4f}")
    print(f"  N updates: {metrics['n_updates']}")
    print("[PASS] PPO update completed successfully")


def test_ppo_learning():
    """Test that PPO can learn on a simple problem (policy should shift)."""
    print("\n" + "=" * 60)
    print("TEST: PPO learning signal (policy shifts)")
    print("=" * 60)

    state_dim = 32  # Small for speed
    policy = PolicyNetwork(input_dim=state_dim, hidden_dim=64).float()
    value_net = ValueNetwork(input_dim=state_dim, hidden_dim=64).float()

    cfg = PPOConfig(lr=1e-3, ppo_epochs=4, mini_batch_size=16)
    trainer = PPOTrainer(policy, value_net, config=cfg, device="cpu")

    # Create batch where action=0 (continue) always gets high advantage
    n = 64
    states = torch.randn(n, state_dim)
    actions = torch.zeros(n, dtype=torch.long)  # Always continue
    advantages = torch.ones(n) * 5.0  # High advantage for continue

    # Get initial log probs
    with torch.no_grad():
        init_probs = policy.get_distribution(states).probs.mean(dim=0)
        init_continue_prob = init_probs[0].item()

    batch = TrajectoryBatch(
        states=states,
        actions=actions,
        old_log_probs=policy.get_distribution(states).log_prob(actions).detach(),
        returns=advantages + value_net(states).squeeze(-1).detach(),
        advantages=advantages,
        values=value_net(states).squeeze(-1).detach(),
    )

    # Train for several iterations
    for _ in range(10):
        trainer.update(batch)

    # Check that continue probability increased
    with torch.no_grad():
        final_probs = policy.get_distribution(states).probs.mean(dim=0)
        final_continue_prob = final_probs[0].item()

    print(f"  P(continue) before: {init_continue_prob:.4f}")
    print(f"  P(continue) after:  {final_continue_prob:.4f}")
    assert final_continue_prob > init_continue_prob, "PPO should increase probability of high-advantage action"
    print("[PASS] PPO shifts policy toward high-advantage actions")


if __name__ == "__main__":
    test_gae()
    test_gae_discount()
    test_episodes_to_batch()
    test_ppo_update()
    test_ppo_learning()

    print("\n" + "=" * 60)
    print("ALL PPO TESTS PASSED")
    print("=" * 60)
