"""Policy and value networks for the RL agent.

Architecture (research.md §3.3):
    Policy:  Linear(d, 512) → ReLU → Linear(512, 256) → ReLU → Linear(256, 3) → Softmax
    Value:   Linear(d, 512) → ReLU → Linear(512, 256) → ReLU → Linear(256, 1)

|A| = 3: {continue, stop, refine} (D-03, D-38).

References:
    - research.md §3.3
    - discovery.md D-03 (deterministic mask → 3 actions)
    - discovery.md D-38 (|A| should be explicit)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical

from src.agent.state import StateExtractor

# Action indices
ACTION_CONTINUE = 0
ACTION_STOP = 1
ACTION_REFINE = 2
NUM_ACTIONS = 3
ACTION_NAMES = ["continue", "stop", "refine"]


class PolicyNetwork(nn.Module):
    """Policy network: maps state features to action probabilities.

    pi(a|s) = Softmax(f(phi(s))) ∈ Δ³
    """

    def __init__(self, input_dim: int = StateExtractor.TOTAL_DIM, hidden_dim: int = 512, num_actions: int = NUM_ACTIONS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_actions),
        )
        self.num_actions = num_actions
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        """Compute action logits.

        Args:
            phi: (batch, input_dim) state features.

        Returns:
            logits: (batch, num_actions) unnormalized log-probabilities.
        """
        return self.net(phi)

    def get_distribution(self, phi: torch.Tensor) -> Categorical:
        """Get the action distribution for given state features."""
        logits = self.forward(phi)
        return Categorical(logits=logits)

    def get_action(
        self, phi: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample an action and return action, log_prob, entropy.

        Args:
            phi: (batch, input_dim) state features.
            deterministic: If True, return argmax action.

        Returns:
            action: (batch,) sampled action indices.
            log_prob: (batch,) log probability of selected action.
            entropy: (batch,) entropy of the distribution.
        """
        dist = self.get_distribution(phi)
        if deterministic:
            action = dist.probs.argmax(dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy

    def evaluate_action(
        self, phi: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate log_prob and entropy for a given state-action pair.

        Used during PPO update to compute the importance ratio.

        Args:
            phi: (batch, input_dim) state features.
            action: (batch,) action indices.

        Returns:
            log_prob: (batch,) log probability of the action.
            entropy: (batch,) entropy of the distribution.
        """
        dist = self.get_distribution(phi)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob, entropy


class ValueNetwork(nn.Module):
    """Value network: maps state features to a scalar value estimate.

    V(s) = g(phi(s)) ∈ R
    """

    def __init__(self, input_dim: int = StateExtractor.TOTAL_DIM, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)

    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        """Compute state value.

        Args:
            phi: (batch, input_dim) state features.

        Returns:
            value: (batch, 1) estimated state value.
        """
        return self.net(phi)
