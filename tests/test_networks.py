"""Tests for policy and value networks.

Run: uv run python tests/test_networks.py
No GPU required (CPU-only tests).
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.networks import (
    PolicyNetwork, ValueNetwork,
    ACTION_CONTINUE, ACTION_STOP, ACTION_REFINE, NUM_ACTIONS, ACTION_NAMES,
)
from src.agent.state import StateExtractor


def test_policy_network():
    """Test policy network dimensions and outputs."""
    print("=" * 60)
    print("TEST: Policy network")
    print("=" * 60)

    input_dim = StateExtractor.TOTAL_DIM
    policy = PolicyNetwork(input_dim=input_dim)

    # Count parameters
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"  Parameters: {n_params:,}")
    assert n_params < 1_500_000, f"Policy too large: {n_params} params"

    # Forward pass
    batch_size = 8
    phi = torch.randn(batch_size, input_dim)
    logits = policy(phi)

    assert logits.shape == (batch_size, NUM_ACTIONS), f"Expected ({batch_size}, {NUM_ACTIONS}), got {logits.shape}"
    print(f"[PASS] Logits shape: {logits.shape}")

    # Distribution
    dist = policy.get_distribution(phi)
    probs = dist.probs
    assert probs.shape == (batch_size, NUM_ACTIONS)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(batch_size), atol=1e-5), "Probabilities should sum to 1"
    print(f"[PASS] Probabilities sum to 1, shape={probs.shape}")

    # Sample action
    action, log_prob, entropy = policy.get_action(phi)
    assert action.shape == (batch_size,)
    assert log_prob.shape == (batch_size,)
    assert entropy.shape == (batch_size,)
    assert (action >= 0).all() and (action < NUM_ACTIONS).all(), f"Actions out of range: {action}"
    print(f"[PASS] Sampled actions: {action.tolist()}")
    print(f"  Log probs: {log_prob.tolist()}")
    print(f"  Entropy: {entropy.mean():.4f}")

    # Deterministic action
    action_det, _, _ = policy.get_action(phi, deterministic=True)
    action_det2, _, _ = policy.get_action(phi, deterministic=True)
    assert torch.equal(action_det, action_det2), "Deterministic actions should be identical"
    print(f"[PASS] Deterministic actions: {action_det.tolist()}")

    # Evaluate action
    log_prob_eval, entropy_eval = policy.evaluate_action(phi, action)
    assert log_prob_eval.shape == (batch_size,)
    print(f"[PASS] evaluate_action works: log_prob shape={log_prob_eval.shape}")


def test_value_network():
    """Test value network dimensions and outputs."""
    print("\n" + "=" * 60)
    print("TEST: Value network")
    print("=" * 60)

    input_dim = StateExtractor.TOTAL_DIM
    value_net = ValueNetwork(input_dim=input_dim)

    n_params = sum(p.numel() for p in value_net.parameters())
    print(f"  Parameters: {n_params:,}")

    batch_size = 8
    phi = torch.randn(batch_size, input_dim)
    values = value_net(phi)

    assert values.shape == (batch_size, 1), f"Expected ({batch_size}, 1), got {values.shape}"
    print(f"[PASS] Value output shape: {values.shape}")
    print(f"  Values: {values.squeeze().tolist()}")


def test_gradient_flow():
    """Test that gradients flow correctly through both networks."""
    print("\n" + "=" * 60)
    print("TEST: Gradient flow")
    print("=" * 60)

    input_dim = StateExtractor.TOTAL_DIM
    policy = PolicyNetwork(input_dim=input_dim)
    value_net = ValueNetwork(input_dim=input_dim)

    phi = torch.randn(4, input_dim, requires_grad=False)

    # Policy gradient
    action, log_prob, entropy = policy.get_action(phi)
    loss_policy = -log_prob.mean()  # Simple REINFORCE loss
    loss_policy.backward()
    grad_norms = [p.grad.norm().item() for p in policy.parameters() if p.grad is not None]
    assert all(g > 0 for g in grad_norms), "Some policy gradients are zero"
    print(f"[PASS] Policy gradients flow: {len(grad_norms)} params with non-zero grad")

    # Value gradient
    values = value_net(phi)
    target = torch.ones_like(values)
    loss_value = (values - target).pow(2).mean()
    loss_value.backward()
    grad_norms_v = [p.grad.norm().item() for p in value_net.parameters() if p.grad is not None]
    assert all(g > 0 for g in grad_norms_v), "Some value gradients are zero"
    print(f"[PASS] Value gradients flow: {len(grad_norms_v)} params with non-zero grad")


def test_action_space():
    """Test that action constants are consistent."""
    print("\n" + "=" * 60)
    print("TEST: Action space constants")
    print("=" * 60)

    assert NUM_ACTIONS == 3, f"|A| should be 3 (D-38), got {NUM_ACTIONS}"
    assert ACTION_CONTINUE == 0
    assert ACTION_STOP == 1
    assert ACTION_REFINE == 2
    assert len(ACTION_NAMES) == NUM_ACTIONS
    print(f"[PASS] |A| = {NUM_ACTIONS}: {ACTION_NAMES}")


if __name__ == "__main__":
    test_action_space()
    test_policy_network()
    test_value_network()
    test_gradient_flow()

    print("\n" + "=" * 60)
    print("ALL NETWORK TESTS PASSED")
    print("=" * 60)
