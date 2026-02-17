"""PPO training loop for the AdDiffusion agent.

Implements Proximal Policy Optimization with:
- GAE advantage estimation (lambda=0.95)
- Clipped objective (epsilon=0.2)
- Entropy bonus (c_2=0.01)
- Value loss (c_1=0.5)
- Gradient clipping (max_norm=0.5)

References:
    - research.md §3.6 (PPO objective, hyperparameters)
    - discovery.md D-21 (use index i, not t, for trajectory steps)
    - discovery.md D-29 (sample inefficiency mitigation)
    - discovery.md D-32 (curriculum learning)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn

from src.agent.episode import EpisodeResult, Transition
from src.agent.networks import PolicyNetwork, ValueNetwork


@dataclass
class PPOConfig:
    """PPO training hyperparameters (research.md §3.6)."""

    lr: float = 3e-4
    gamma_d: float = 0.99         # Discount factor (D-09: not gamma)
    gae_lambda: float = 0.95      # GAE lambda
    clip_epsilon: float = 0.2     # PPO clip range
    entropy_coeff: float = 0.01   # c_2: entropy bonus
    value_coeff: float = 0.5      # c_1: value loss weight
    max_grad_norm: float = 0.5    # Gradient clipping
    ppo_epochs: int = 4           # K gradient epochs per data collection
    mini_batch_size: int = 64     # Mini-batch size for PPO updates


@dataclass
class TrajectoryBatch:
    """Batch of trajectory data for PPO update."""

    states: torch.Tensor        # (total_steps, state_dim)
    actions: torch.Tensor       # (total_steps,) int
    old_log_probs: torch.Tensor # (total_steps,)
    returns: torch.Tensor       # (total_steps,)
    advantages: torch.Tensor    # (total_steps,)
    values: torch.Tensor        # (total_steps,)


def compute_gae(
    rewards: list[float],
    values: list[float],
    dones: list[bool],
    gamma: float = 0.99,
    lam: float = 0.95,
    last_value: float = 0.0,
) -> tuple[list[float], list[float]]:
    """Compute Generalized Advantage Estimation.

    Args:
        rewards: Per-step rewards.
        values: Per-step value estimates V(s_i).
        dones: Per-step done flags.
        gamma: Discount factor (gamma_d, D-09).
        lam: GAE lambda.
        last_value: Bootstrap value for incomplete episodes.

    Returns:
        advantages: Per-step advantage estimates.
        returns: Per-step return targets (advantages + values).
    """
    n = len(rewards)
    advantages = [0.0] * n
    last_gae = 0.0

    for i in reversed(range(n)):
        if i == n - 1:
            next_value = last_value
            next_non_terminal = 1.0 - float(dones[i])
        else:
            next_value = values[i + 1]
            next_non_terminal = 1.0 - float(dones[i])

        delta = rewards[i] + gamma * next_value * next_non_terminal - values[i]
        advantages[i] = delta + gamma * lam * next_non_terminal * last_gae
        last_gae = advantages[i]

    returns = [adv + val for adv, val in zip(advantages, values)]
    return advantages, returns


def episodes_to_batch(
    episodes: list[EpisodeResult],
    gamma: float = 0.99,
    lam: float = 0.95,
) -> TrajectoryBatch:
    """Convert a list of episode results into a flat batch for PPO.

    Args:
        episodes: List of completed episodes.
        gamma: Discount factor.
        lam: GAE lambda.

    Returns:
        TrajectoryBatch with flattened trajectory data.
    """
    all_states = []
    all_actions = []
    all_log_probs = []
    all_returns = []
    all_advantages = []
    all_values = []

    for ep in episodes:
        if not ep.transitions:
            continue

        rewards = [t.reward for t in ep.transitions]
        values = [t.value for t in ep.transitions]
        dones = [t.done for t in ep.transitions]

        advantages, returns = compute_gae(rewards, values, dones, gamma, lam)

        for i, t in enumerate(ep.transitions):
            all_states.append(t.state_features.squeeze(0))  # (d,)
            all_actions.append(t.action)
            all_log_probs.append(t.log_prob)
            all_returns.append(returns[i])
            all_advantages.append(advantages[i])
            all_values.append(t.value)

    if not all_states:
        raise ValueError("No transitions in any episode")

    batch = TrajectoryBatch(
        states=torch.stack(all_states),
        actions=torch.tensor(all_actions, dtype=torch.long),
        old_log_probs=torch.tensor(all_log_probs, dtype=torch.float32),
        returns=torch.tensor(all_returns, dtype=torch.float32),
        advantages=torch.tensor(all_advantages, dtype=torch.float32),
        values=torch.tensor(all_values, dtype=torch.float32),
    )

    # Normalize advantages
    if batch.advantages.std() > 1e-8:
        batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)

    return batch


class PPOTrainer:
    """PPO trainer for policy and value networks."""

    def __init__(
        self,
        policy: PolicyNetwork,
        value_net: ValueNetwork,
        config: PPOConfig = PPOConfig(),
        device: str = "cuda",
    ):
        self.policy = policy
        self.value_net = value_net
        self.config = config
        self.device = device

        self.optimizer = torch.optim.Adam(
            list(policy.parameters()) + list(value_net.parameters()),
            lr=config.lr,
        )

    def update(self, batch: TrajectoryBatch) -> dict:
        """Run K epochs of PPO updates on a trajectory batch.

        Returns:
            Dictionary of training metrics.
        """
        cfg = self.config

        # Move to device
        states = batch.states.to(self.device)
        actions = batch.actions.to(self.device)
        old_log_probs = batch.old_log_probs.to(self.device)
        returns = batch.returns.to(self.device)
        advantages = batch.advantages.to(self.device)

        total_steps = states.shape[0]
        metrics = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "total_loss": 0.0,
            "clip_fraction": 0.0,
            "approx_kl": 0.0,
            "n_updates": 0,
        }

        for epoch in range(cfg.ppo_epochs):
            # Generate random mini-batch indices
            indices = torch.randperm(total_steps, device=self.device)

            for start in range(0, total_steps, cfg.mini_batch_size):
                end = min(start + cfg.mini_batch_size, total_steps)
                mb_idx = indices[start:end]

                mb_states = states[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_returns = returns[mb_idx]
                mb_advantages = advantages[mb_idx]

                # Current policy evaluation
                new_log_probs, entropy = self.policy.evaluate_action(mb_states, mb_actions)

                # PPO clipped objective (research.md §3.6)
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - cfg.clip_epsilon, 1 + cfg.clip_epsilon) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                new_values = self.value_net(mb_states).squeeze(-1)
                value_loss = (new_values - mb_returns).pow(2).mean()

                # Total loss: L = L_policy + c1 * L_value - c2 * H[pi]
                loss = policy_loss + cfg.value_coeff * value_loss - cfg.entropy_coeff * entropy.mean()

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.policy.parameters()) + list(self.value_net.parameters()),
                    cfg.max_grad_norm,
                )
                self.optimizer.step()

                # Track metrics
                with torch.no_grad():
                    clip_frac = ((ratio - 1).abs() > cfg.clip_epsilon).float().mean().item()
                    approx_kl = (mb_old_log_probs - new_log_probs).mean().item()

                metrics["policy_loss"] += policy_loss.item()
                metrics["value_loss"] += value_loss.item()
                metrics["entropy"] += entropy.mean().item()
                metrics["total_loss"] += loss.item()
                metrics["clip_fraction"] += clip_frac
                metrics["approx_kl"] += approx_kl
                metrics["n_updates"] += 1

        # Average metrics
        n = max(metrics["n_updates"], 1)
        for key in ["policy_loss", "value_loss", "entropy", "total_loss", "clip_fraction", "approx_kl"]:
            metrics[key] /= n

        return metrics
