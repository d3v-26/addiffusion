# Reward Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the always-continue, never-refine policy collapse by replacing the terminal reward with a dual-baseline normalized formula and adding an attention-entropy refine bonus.

**Architecture:** `RewardComputer` loads a precomputed `data/baseline_scores.json` at init (DDIM-20 + DDIM-50 scores per training prompt) and uses them to normalize the terminal reward so that stopping early with DDIM-20 quality yields a positive signal proportional to steps saved. A new `compute_attention_entropy()` utility (pure tensor math, no models) is added to `src/rewards/reward.py` and used in `episode.py` to compute the refine bonus. The PPO trainer gains an `override_entropy_coeff` parameter to support the annealing schedule driven from `train.py`.

**Tech Stack:** Python 3.10, PyTorch 2.7, diffusers, open_clip, ImageReward, uv, SLURM

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `configs/default.yaml` | Modify | Add `c_save`, `c_refine`, `baseline_scores_path`, entropy schedule params; remove `entropy_coeff` |
| `src/rewards/reward.py` | Modify | Add `compute_attention_entropy()`, update `RewardConfig`, rewrite `compute_terminal_reward`, add `compute_refine_bonus`, update `compute_reward` signature |
| `src/agent/episode.py` | Modify | Compute attention entropy before action; track `running_nfe`; pass both to `reward_fn` |
| `src/agent/ppo.py` | Modify | Add `override_entropy_coeff` param to `PPOTrainer.update()` |
| `src/train.py` | Modify | Entropy annealing schedule; update `RewardConfig` construction; update `reward_fn` closure |
| `tests/test_reward.py` | Modify | Add CPU-only tests for new terminal formula, refine bonus, entropy computation, fallback |
| `tests/test_ppo.py` | Modify | Add test for `override_entropy_coeff` |
| `scripts/precompute_baseline_scores.py` | Create | Run DDIM-20 + DDIM-50 on training prompts, write `data/baseline_scores.json` |
| `scripts/precompute_baseline_scores.slurm` | Create | SLURM wrapper for precomputation job |

---

## Task 1: Update `configs/default.yaml`

**Files:**
- Modify: `configs/default.yaml`

- [ ] **Step 1: Replace `entropy_coeff` with schedule params and add reward fields**

In `configs/default.yaml`, make these changes:

Under `ppo:`, replace:
```yaml
  entropy_coeff: 0.01  # c_2
```
with:
```yaml
  entropy_coeff_start: 0.05   # high entropy for first 500 iters to avoid collapse
  entropy_coeff_end: 0.01     # c_2 from research.md §3.6
  entropy_anneal_steps: 500
```

Under `reward:`, add after `normalize: true`:
```yaml
  baseline_scores_path: "data/baseline_scores.json"
  c_save: 1.0    # step-savings weight: saving all N_max steps ≈ going from DDIM-20 to DDIM-50 quality
  c_refine: 0.2  # attention-entropy refine bonus weight
```

- [ ] **Step 2: Commit**

```bash
git add configs/default.yaml
git commit -m "config: add reward redesign params and entropy annealing schedule"
```

---

## Task 2: Add `compute_attention_entropy` + update `RewardConfig` in `src/rewards/reward.py`

**Files:**
- Modify: `src/rewards/reward.py`
- Test: `tests/test_reward.py`

- [ ] **Step 1: Write failing tests for new config fields and attention entropy**

Add to `tests/test_reward.py` (before the `if __name__ == "__main__":` block):

```python
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
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /Users/nik/Desktop/smile-lab/addiffusion
uv run python -c "
import sys; sys.path.insert(0, '.')
import torch
from tests.test_reward import test_new_reward_config_fields, test_compute_attention_entropy_uniform
try:
    test_new_reward_config_fields()
    print('UNEXPECTED PASS')
except Exception as e:
    print(f'EXPECTED FAIL: {e}')
"
```

Expected: `AttributeError: 'RewardConfig' object has no attribute 'c_save'` or similar.

- [ ] **Step 3: Add `compute_attention_entropy` function and new `RewardConfig` fields**

At the top of `src/rewards/reward.py`, add `import math` after the existing imports.

Replace the `RewardConfig` dataclass with:

```python
@dataclass
class RewardConfig:
    """Reward hyperparameters (research.md §3.6)."""

    # Quality delta weights
    alpha_1: float = 1.0   # CLIP delta
    alpha_2: float = 0.5   # Aesthetic delta
    alpha_3: float = 0.2   # Stability (DINO similarity, D-11)
    alpha_4: float = 0.8   # ImageReward delta

    # Normalization scales (D-37)
    clip_norm: float = 0.05
    aesthetic_norm: float = 0.3
    image_reward_norm: float = 0.2

    # Efficiency
    c_nfe: float = 0.01  # D-09: renamed from gamma

    # Terminal reward weights
    beta_1: float = 2.0  # CLIP score
    beta_2: float = 1.0  # Aesthetic
    beta_3: float = 1.5  # ImageReward

    # Step-savings bonus (new)
    c_save: float = 1.0  # weight for (N_max - NFE_used) / N_max term

    # Refine bonus (new)
    c_refine: float = 0.2  # weight for attention entropy bonus on refine action

    # Baseline scores file (new) — None disables normalization (fallback to absolute)
    baseline_scores_path: Optional[str] = None

    # Refinement
    refine_k: int = 2

    # Normalization toggle (A5 ablation)
    normalize: bool = True
```

After the `RewardConfig` dataclass, add this standalone function:

```python
def compute_attention_entropy(attention_maps: torch.Tensor) -> float:
    """Compute normalized Shannon entropy of aggregated cross-attention map.

    Measures how diffuse (uncertain) the attention is across spatial positions.
    High entropy = attention spread across many pixels = under-generated regions.
    Low entropy = attention concentrated = image structure is clear.

    Args:
        attention_maps: (h, w, L) cross-attention maps from UNetAttentionExtractor,
                        where L is the number of text tokens.

    Returns:
        Normalized entropy in [0, 1]. 1 = maximally uniform (diffuse), 0 = concentrated.
    """
    h, w, _L = attention_maps.shape
    # Sum over text tokens → spatial attention distribution (h, w)
    A = attention_maps.sum(dim=-1).float()
    A = A / (A.sum() + 1e-8)  # normalize to probability distribution over pixels
    # Shannon entropy H = -sum(p * log(p))
    H = -(A * (A + 1e-8).log()).sum()
    # Normalize by maximum possible entropy (uniform distribution over h*w pixels)
    H_max = math.log(h * w)
    return float((H / H_max).clamp(0.0, 1.0))
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
uv run python -c "
import sys; sys.path.insert(0, '.')
import torch
from tests.test_reward import (
    test_new_reward_config_fields,
    test_compute_attention_entropy_uniform,
    test_compute_attention_entropy_concentrated,
    test_compute_attention_entropy_range,
)
test_new_reward_config_fields()
test_compute_attention_entropy_uniform()
test_compute_attention_entropy_concentrated()
test_compute_attention_entropy_range()
print('ALL PASS')
"
```

Expected: `ALL PASS`

- [ ] **Step 5: Commit**

```bash
git add src/rewards/reward.py tests/test_reward.py
git commit -m "feat: add compute_attention_entropy and RewardConfig c_save/c_refine fields"
```

---

## Task 3: Rewrite `compute_terminal_reward` and add `compute_refine_bonus`

**Files:**
- Modify: `src/rewards/reward.py`
- Test: `tests/test_reward.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_reward.py`:

```python
def test_terminal_reward_with_baselines():
    """Terminal reward is normalized relative to DDIM-20/50 baselines."""
    cfg = RewardConfig(
        beta_1=3.105, beta_2=1.349, beta_3=0.643,
        c_save=1.0,
        baseline_scores_path=None,  # will inject baseline_scores directly
    )
    rc = RewardComputer(config=cfg, device="cpu")

    # Inject baseline scores directly (bypassing file load)
    rc.baseline_scores = {
        "a cat": {
            "ddim20": {"clip": 0.30, "aesthetic": 5.0, "image_reward": 0.40},
            "ddim50": {"clip": 0.35, "aesthetic": 5.5, "image_reward": 0.60},
        }
    }

    # Agent matches DDIM-50 quality exactly, saves 25 steps out of 50
    fake_image = torch.rand(1, 3, 64, 64)

    # Mock score methods
    rc._mock_scores = {"clip": 0.35, "aesthetic": 5.5, "image_reward": 0.60}

    def mock_clip(img, prompt): return rc._mock_scores["clip"]
    def mock_ir(img, prompt): return rc._mock_scores["image_reward"]
    def mock_aes(img): return rc._mock_scores["aesthetic"]
    rc.clip_score = mock_clip
    rc.image_reward_score = mock_ir
    rc.aesthetic_score = mock_aes

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
    # Fallback: (0.5 - 0) / max(1.0 - 0, eps) = 0.5; beta_1=1.0 → quality_term=0.5
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
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
uv run python -c "
import sys; sys.path.insert(0, '.')
import torch
from tests.test_reward import test_refine_bonus
try:
    test_refine_bonus()
    print('UNEXPECTED PASS')
except AttributeError as e:
    print(f'EXPECTED FAIL: {e}')
"
```

Expected: `AttributeError: 'RewardComputer' object has no attribute 'compute_refine_bonus'`

- [ ] **Step 3: Update `RewardComputer.__init__` to load baseline scores**

In `src/rewards/reward.py`, update `RewardComputer.__init__`:

```python
def __init__(self, config: RewardConfig = RewardConfig(), device: str = "cuda"):
    self.config = config
    self.device = device
    self._clip_model = None
    self._clip_preprocess = None
    self._clip_tokenizer = None
    self._ir_model = None
    self._aesthetic_model = None
    self._dino_model = None

    # Load baseline scores (DDIM-20 and DDIM-50) for terminal reward normalization
    self.baseline_scores: dict = {}
    if config.baseline_scores_path is not None:
        import json as _json
        try:
            with open(config.baseline_scores_path) as f:
                self.baseline_scores = _json.load(f)
            print(f"[RewardComputer] Loaded baseline scores for {len(self.baseline_scores)} prompts")
        except FileNotFoundError:
            print(f"[RewardComputer] WARNING: baseline_scores_path={config.baseline_scores_path} not found. "
                  f"Terminal reward will use fallback (unnormalized absolute scores).")
```

- [ ] **Step 4: Replace `compute_terminal_reward` with the new formula**

Replace the existing `compute_terminal_reward` method in `src/rewards/reward.py`:

```python
def compute_terminal_reward(
    self,
    image: torch.Tensor,
    prompt: str,
    nfe_used: int = 50,
    n_max: int = 50,
) -> tuple[float, dict]:
    """Compute R_terminal for the final image.

    R_terminal = sum(beta_k * (score_agent_k - ddim20_k) / max(ddim50_k - ddim20_k, eps))
               + c_save * (n_max - nfe_used) / n_max

    Quality term is 0 when agent matches DDIM-20, 1 when it matches DDIM-50.
    Step savings term rewards early stopping proportional to steps saved.
    Falls back to unnormalized absolute scores when prompt not in baseline_scores.
    """
    cfg = self.config
    eps = 1e-6

    clip = self.clip_score(image, prompt)
    ir = self.image_reward_score(image, prompt)
    aesthetic = self.aesthetic_score(image)

    # Lookup baseline scores — fallback: ddim20=0, ddim50=1 → unnormalized absolute
    baseline = self.baseline_scores.get(prompt, {})
    ddim20 = baseline.get("ddim20", {"clip": 0.0, "aesthetic": 0.0, "image_reward": 0.0})
    ddim50 = baseline.get("ddim50", {"clip": 1.0, "aesthetic": 1.0, "image_reward": 1.0})

    def norm(score: float, low: float, high: float) -> float:
        return (score - low) / max(high - low, eps)

    quality_term = (
        cfg.beta_1 * norm(clip, ddim20["clip"], ddim50["clip"])
        + cfg.beta_2 * norm(aesthetic, ddim20["aesthetic"], ddim50["aesthetic"])
        + cfg.beta_3 * norm(ir, ddim20["image_reward"], ddim50["image_reward"])
    )
    step_savings = cfg.c_save * (n_max - nfe_used) / max(n_max, 1)
    reward = quality_term + step_savings

    metrics = {
        "terminal_clip": clip,
        "terminal_aesthetic": aesthetic,
        "terminal_ir": ir,
        "terminal_quality_term": quality_term,
        "terminal_step_savings": step_savings,
        "r_terminal": reward,
    }
    return reward, metrics
```

- [ ] **Step 5: Add `compute_refine_bonus` method**

Add after `compute_terminal_reward`:

```python
def compute_refine_bonus(self, attention_entropy: float) -> float:
    """Compute per-step refine bonus based on attention entropy.

    R_refine_bonus = c_refine * H  where H ∈ [0, 1] is normalized attention entropy.
    High entropy = diffuse attention = under-generated regions = refine genuinely useful.
    Only applied when action == ACTION_REFINE.
    """
    return self.config.c_refine * attention_entropy
```

- [ ] **Step 6: Update `compute_reward` to use new terminal and refine bonus**

Replace the existing `compute_reward` method:

```python
def compute_reward(
    self,
    prev_image: Optional[torch.Tensor],
    curr_image: torch.Tensor,
    prompt: str,
    action: int,
    is_terminal: bool,
    nfe_used: int = 50,
    n_max: int = 50,
    attention_entropy: float = 0.0,
    prev_clip: Optional[float] = None,
    prev_ir: Optional[float] = None,
    prev_aesthetic: Optional[float] = None,
) -> tuple[float, dict]:
    """Compute total reward R = R_quality + R_efficiency + R_terminal + R_refine_bonus.

    D-13: No outer lambda weights. Sum directly.
    """
    r_quality, q_metrics = self.compute_quality_reward(
        prev_image, curr_image, prompt, prev_clip, prev_ir, prev_aesthetic,
    )
    r_efficiency = self.compute_efficiency_reward(action)

    r_terminal = 0.0
    t_metrics = {}
    if is_terminal:
        r_terminal, t_metrics = self.compute_terminal_reward(
            curr_image, prompt, nfe_used=nfe_used, n_max=n_max,
        )

    r_refine_bonus = 0.0
    if action == ACTION_REFINE:
        r_refine_bonus = self.compute_refine_bonus(attention_entropy)

    total = r_quality + r_efficiency + r_terminal + r_refine_bonus

    metrics = {
        **q_metrics,
        **t_metrics,
        "r_quality": r_quality,
        "r_efficiency": r_efficiency,
        "r_terminal": r_terminal,
        "r_refine_bonus": r_refine_bonus,
        "r_total": total,
    }
    return total, metrics
```

- [ ] **Step 7: Run all new tests**

```bash
uv run python -c "
import sys; sys.path.insert(0, '.')
import torch
from tests.test_reward import (
    test_new_reward_config_fields,
    test_compute_attention_entropy_uniform,
    test_compute_attention_entropy_concentrated,
    test_compute_attention_entropy_range,
    test_terminal_reward_with_baselines,
    test_terminal_reward_fallback_no_baseline,
    test_refine_bonus,
    test_efficiency_reward,
    test_normalization_toggle,
)
test_new_reward_config_fields()
test_compute_attention_entropy_uniform()
test_compute_attention_entropy_concentrated()
test_compute_attention_entropy_range()
test_terminal_reward_with_baselines()
test_terminal_reward_fallback_no_baseline()
test_refine_bonus()
test_efficiency_reward()
test_normalization_toggle()
print('ALL PASS')
"
```

Expected: `ALL PASS`

- [ ] **Step 8: Commit**

```bash
git add src/rewards/reward.py tests/test_reward.py
git commit -m "feat: new terminal reward (dual-baseline normalized) and refine bonus"
```

---

## Task 4: Update `src/agent/episode.py` to pass `nfe_used` and `attention_entropy`

**Files:**
- Modify: `src/agent/episode.py`

- [ ] **Step 1: Add imports and `running_nfe` tracker**

At the top of `src/agent/episode.py`, add this import after the existing imports:

```python
from src.rewards.reward import compute_attention_entropy
```

In `run_episode`, add `running_nfe = 0` immediately before the `while not pipe_state.is_done:` loop:

```python
prev_z0_pred = None
prev_decoded = None
running_nfe = 0  # tracks cumulative NFE for terminal reward computation

while not pipe_state.is_done:
```

- [ ] **Step 2: Compute attention entropy before the action block**

Inside the loop, after `step_out = self.pipeline.denoise_step(...)` and before the `if is_warmup:` block, add:

```python
# Compute attention entropy from this step's cross-attention maps (before action modifies them)
try:
    _attn_maps = self.attention_extractor.get_attention_maps()  # (h, w, L)
    _attn_entropy = compute_attention_entropy(_attn_maps)
except RuntimeError:
    _attn_entropy = 0.0
```

- [ ] **Step 3: Increment `running_nfe` after each action**

After the action execution block (after the `elif action == ACTION_REFINE:` block, before the reward computation), add:

```python
running_nfe += nfe
```

- [ ] **Step 4: Pass `nfe_used` and `attention_entropy` to `reward_fn`**

Update the `reward_fn` call to include the two new keyword arguments:

```python
if reward_fn is not None:
    reward = reward_fn(
        prev_z0=prev_z0_pred,
        curr_z0=z0_for_reward,
        action=action,
        step_index=step_idx,
        n_max=n_max,
        is_terminal=pipe_state.is_done,
        prompt=prompt,
        decoded_image=decoded_image,
        nfe_used=running_nfe,
        attention_entropy=_attn_entropy,
    )
```

- [ ] **Step 5: Verify the existing `test_episode.py` still passes (CPU, no GPU needed for structure tests)**

```bash
uv run python -c "
import sys; sys.path.insert(0, '.')
# Just verify the module imports cleanly — GPU tests need HPC
from src.agent.episode import EpisodeRunner
from src.rewards.reward import compute_attention_entropy
print('Imports OK')
"
```

Expected: `Imports OK`

- [ ] **Step 6: Commit**

```bash
git add src/agent/episode.py
git commit -m "feat: pass nfe_used and attention_entropy to reward_fn in episode loop"
```

---

## Task 5: Add `override_entropy_coeff` to `PPOTrainer.update()`

**Files:**
- Modify: `src/agent/ppo.py`
- Test: `tests/test_ppo.py`

- [ ] **Step 1: Write failing test**

Read `tests/test_ppo.py` first to understand its structure, then add:

```python
def test_ppo_override_entropy_coeff():
    """PPOTrainer.update() respects override_entropy_coeff over config value."""
    import torch
    from src.agent.networks import PolicyNetwork, ValueNetwork
    from src.agent.ppo import PPOConfig, PPOTrainer, TrajectoryBatch

    policy = PolicyNetwork(input_dim=16, hidden_dim=32, num_actions=3)
    value_net = ValueNetwork(input_dim=16, hidden_dim=32)
    cfg = PPOConfig(entropy_coeff=0.01, ppo_epochs=1, mini_batch_size=4)
    trainer = PPOTrainer(policy, value_net, config=cfg, device="cpu")

    batch = TrajectoryBatch(
        states=torch.randn(8, 16),
        actions=torch.randint(0, 3, (8,)),
        old_log_probs=torch.randn(8),
        returns=torch.randn(8),
        advantages=torch.randn(8),
        values=torch.randn(8),
    )

    # Should not raise — override_entropy_coeff is accepted
    metrics = trainer.update(batch, override_entropy_coeff=0.05)
    assert "entropy" in metrics
    print(f"[PASS] override_entropy_coeff=0.05 accepted, entropy={metrics['entropy']:.4f}")
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
uv run python -c "
import sys; sys.path.insert(0, '.')
from tests.test_ppo import test_ppo_override_entropy_coeff
try:
    test_ppo_override_entropy_coeff()
    print('UNEXPECTED PASS')
except TypeError as e:
    print(f'EXPECTED FAIL: {e}')
"
```

Expected: `TypeError: update() got an unexpected keyword argument 'override_entropy_coeff'`

- [ ] **Step 3: Update `PPOTrainer.update()` signature**

In `src/agent/ppo.py`, change the `update` method signature from:

```python
def update(self, batch: TrajectoryBatch) -> dict:
```

to:

```python
def update(self, batch: TrajectoryBatch, override_entropy_coeff: Optional[float] = None) -> dict:
```

Add at the top of the method body (before `cfg = self.config`):

```python
entropy_coeff = override_entropy_coeff if override_entropy_coeff is not None else self.config.entropy_coeff
```

Then replace all uses of `cfg.entropy_coeff` in the method body with `entropy_coeff`. There is one occurrence in the loss line:

```python
loss = policy_loss + cfg.value_coeff * value_loss - cfg.entropy_coeff * entropy.mean()
```

Change to:

```python
loss = policy_loss + cfg.value_coeff * value_loss - entropy_coeff * entropy.mean()
```

Also add `from typing import Optional` if not already imported at the top of `ppo.py`. Check the existing imports first — if `Optional` is already there, skip this.

- [ ] **Step 4: Run test to confirm it passes**

```bash
uv run python -c "
import sys; sys.path.insert(0, '.')
from tests.test_ppo import test_ppo_override_entropy_coeff
test_ppo_override_entropy_coeff()
"
```

Expected: `[PASS] override_entropy_coeff=0.05 accepted`

- [ ] **Step 5: Commit**

```bash
git add src/agent/ppo.py tests/test_ppo.py
git commit -m "feat: add override_entropy_coeff param to PPOTrainer.update()"
```

---

## Task 6: Update `src/train.py`

**Files:**
- Modify: `src/train.py`

- [ ] **Step 1: Update `RewardConfig` construction to use new fields**

In `train()`, replace the `RewardConfig(...)` construction block:

```python
reward_cfg = RewardConfig(
    alpha_1=cfg["reward"]["alpha_1"],
    alpha_2=cfg["reward"]["alpha_2"],
    alpha_3=cfg["reward"]["alpha_3"],
    alpha_4=cfg["reward"]["alpha_4"],
    c_nfe=cfg["reward"]["c_nfe"],
    beta_1=cfg["reward"]["beta_1"],
    beta_2=cfg["reward"]["beta_2"],
    beta_3=cfg["reward"]["beta_3"],
    clip_norm=cfg["reward"]["clip_norm"],
    aesthetic_norm=cfg["reward"]["aesthetic_norm"],
    image_reward_norm=cfg["reward"]["image_reward_norm"],
    normalize=cfg["reward"]["normalize"],
    refine_k=cfg["refinement"]["k"],
    c_save=cfg["reward"].get("c_save", 1.0),
    c_refine=cfg["reward"].get("c_refine", 0.2),
    baseline_scores_path=cfg["reward"].get("baseline_scores_path", None),
)
```

- [ ] **Step 2: Update `PPOConfig` construction to handle removed `entropy_coeff` key**

In `train()`, replace the `PPOConfig(...)` construction:

```python
ppo_cfg = PPOConfig(
    lr=float(cfg["ppo"]["lr"]),
    gamma_d=cfg["ppo"]["gamma_d"],
    gae_lambda=cfg["ppo"]["gae_lambda"],
    clip_epsilon=cfg["ppo"]["clip_epsilon"],
    entropy_coeff=cfg["ppo"].get("entropy_coeff", cfg["ppo"].get("entropy_coeff_end", 0.01)),
    value_coeff=cfg["ppo"]["value_coeff"],
    max_grad_norm=cfg["ppo"]["max_grad_norm"],
    ppo_epochs=cfg["ppo"]["ppo_epochs"],
    mini_batch_size=cfg["ppo"]["mini_batch_size"],
)
```

- [ ] **Step 3: Add entropy annealing schedule variables**

After the `trainer = PPOTrainer(...)` line, add:

```python
# Entropy annealing schedule: high entropy early to prevent always-continue collapse
_entropy_start = cfg["ppo"].get("entropy_coeff_start", 0.05)
_entropy_end = cfg["ppo"].get("entropy_coeff_end", 0.01)
_entropy_anneal_steps = cfg["ppo"].get("entropy_anneal_steps", 500)
```

- [ ] **Step 4: Pass scheduled entropy to `trainer.update()` in the training loop**

In the training loop, replace:

```python
ppo_metrics = trainer.update(batch)
```

with:

```python
_entropy_coeff = _entropy_start if iteration < _entropy_anneal_steps else _entropy_end
ppo_metrics = trainer.update(batch, override_entropy_coeff=_entropy_coeff)
```

- [ ] **Step 5: Update `reward_fn` closure to pass `nfe_used` and `attention_entropy`**

In the training loop, replace the `reward_fn` closure:

```python
def reward_fn(prev_z0, curr_z0, action, step_index, n_max, is_terminal, prompt, decoded_image,
              nfe_used=50, attention_entropy=0.0, _prev=prev_metrics):
    prev_img = _prev.get("prev_decoded")
    reward, metrics = reward_computer.compute_reward(
        prev_image=prev_img,
        curr_image=decoded_image,
        prompt=prompt,
        action=action,
        is_terminal=is_terminal,
        nfe_used=nfe_used,
        n_max=n_max,
        attention_entropy=attention_entropy,
        prev_clip=_prev.get("clip_score"),
        prev_ir=_prev.get("image_reward"),
        prev_aesthetic=_prev.get("aesthetic_score"),
    )
    _prev["prev_decoded"] = decoded_image
    _prev["clip_score"] = metrics.get("clip_score")
    _prev["image_reward"] = metrics.get("image_reward")
    _prev["aesthetic_score"] = metrics.get("aesthetic_score")
    return reward
```

- [ ] **Step 6: Verify module imports cleanly**

```bash
uv run python -c "
import sys; sys.path.insert(0, '.')
import ast, pathlib
src = pathlib.Path('src/train.py').read_text()
ast.parse(src)
print('train.py parses OK')
"
```

Expected: `train.py parses OK`

- [ ] **Step 7: Commit**

```bash
git add src/train.py
git commit -m "feat: entropy annealing schedule and updated reward_fn in train.py"
```

---

## Task 7: Create `scripts/precompute_baseline_scores.py`

**Files:**
- Create: `scripts/precompute_baseline_scores.py`

- [ ] **Step 1: Write the script**

Create `scripts/precompute_baseline_scores.py`:

```python
"""Precompute DDIM-20 and DDIM-50 baseline scores for training prompts.

Runs both baselines on all (or a capped subset of) training prompts and writes
data/baseline_scores.json, which is used by RewardComputer for per-prompt
terminal reward normalization.

Usage:
    uv run python scripts/precompute_baseline_scores.py \\
        --config /blue/ruogu.fang/pateld3/addiffusion/configs/default.yaml \\
        --prompts_file /blue/ruogu.fang/pateld3/addiffusion/data/coco/annotations/captions_val2014.json \\
        --output /blue/ruogu.fang/pateld3/addiffusion/data/baseline_scores.json \\
        --max_prompts 10000 \\
        --seed 42

References:
    - docs/superpowers/specs/2026-04-08-reward-redesign-design.md
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import yaml

from src.diffusion.pipeline import AdaptiveDiffusionPipeline
from src.rewards.reward import RewardComputer, RewardConfig


def load_prompts(path: str, max_prompts: int = None, seed: int = 42) -> list[str]:
    """Load prompts sorted by length (matches curriculum simple_until heuristic)."""
    with open(path) as f:
        data = json.load(f)

    if isinstance(data, list):
        prompts = [p if isinstance(p, str) else p.get("caption", p.get("prompt", "")) for p in data]
    elif "annotations" in data:
        prompts = [ann["caption"] for ann in data["annotations"]]
    else:
        raise ValueError(f"Unknown prompt format in {path}")

    # Deduplicate preserving order
    seen = set()
    unique = []
    for p in prompts:
        if p and p not in seen:
            seen.add(p)
            unique.append(p)

    # Sort by length (matches curriculum: shorter = simpler = trained earlier)
    unique.sort(key=len)

    if max_prompts:
        unique = unique[:max_prompts]

    return unique


def run_ddim(pipeline: AdaptiveDiffusionPipeline, prompt: str, num_steps: int, seed: int, guidance_scale: float) -> torch.Tensor:
    """Run DDIM with `num_steps` steps and return the decoded image (1, 3, H, W)."""
    state = pipeline.prepare(prompt, num_steps=num_steps, seed=seed, guidance_scale=guidance_scale)
    while not state.is_done:
        step_out = pipeline.denoise_step(state, guidance_scale=guidance_scale)
        pipeline.advance_state(state, step_out)
    return pipeline.decode(state.history[-1].z0_pred)


def main():
    parser = argparse.ArgumentParser(description="Precompute DDIM-20 and DDIM-50 baseline scores.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--prompts_file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_prompts", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[precompute] Device: {device}")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    dtype = torch.float16 if cfg["model"]["dtype"] == "float16" else torch.float32
    guidance_scale = cfg["model"]["guidance_scale"]

    print("[precompute] Loading pipeline...")
    pipeline = AdaptiveDiffusionPipeline.from_pretrained(
        cfg["model"]["model_id"],
        scheduler_name="ddim",
        device=device,
        dtype=dtype,
    )

    print("[precompute] Loading reward models...")
    rc = RewardComputer(config=RewardConfig(), device=device)

    prompts = load_prompts(args.prompts_file, max_prompts=args.max_prompts, seed=args.seed)
    print(f"[precompute] {len(prompts)} prompts to process")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume from existing output if present
    if output_path.exists():
        with open(output_path) as f:
            scores = json.load(f)
        print(f"[precompute] Resuming — {len(scores)} prompts already done")
    else:
        scores = {}

    t_start = time.time()
    for i, prompt in enumerate(prompts):
        if prompt in scores:
            continue

        try:
            img20 = run_ddim(pipeline, prompt, num_steps=20, seed=args.seed, guidance_scale=guidance_scale)
            img50 = run_ddim(pipeline, prompt, num_steps=50, seed=args.seed, guidance_scale=guidance_scale)

            scores[prompt] = {
                "ddim20": {
                    "clip": rc.clip_score(img20, prompt),
                    "aesthetic": rc.aesthetic_score(img20),
                    "image_reward": rc.image_reward_score(img20, prompt),
                },
                "ddim50": {
                    "clip": rc.clip_score(img50, prompt),
                    "aesthetic": rc.aesthetic_score(img50),
                    "image_reward": rc.image_reward_score(img50, prompt),
                },
            }
        except Exception as e:
            print(f"[precompute] WARNING: skipped prompt {i} due to {e}")
            continue

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            remaining = (len(prompts) - i - 1) / max(rate, 1e-6)
            print(f"  [{i+1}/{len(prompts)}] {rate:.1f} prompts/s | ETA {remaining/3600:.1f}h")
            # Save checkpoint every 100 prompts
            with open(output_path, "w") as f:
                json.dump(scores, f)

    # Final save
    with open(output_path, "w") as f:
        json.dump(scores, f, indent=2)

    total_time = time.time() - t_start
    print(f"\n[precompute] Done. {len(scores)} prompts in {total_time/3600:.2f}h")
    print(f"[precompute] Output: {output_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the script parses without errors**

```bash
uv run python -c "
import ast, pathlib
src = pathlib.Path('scripts/precompute_baseline_scores.py').read_text()
ast.parse(src)
print('Script parses OK')
"
```

Expected: `Script parses OK`

- [ ] **Step 3: Commit**

```bash
git add scripts/precompute_baseline_scores.py
git commit -m "feat: precompute_baseline_scores.py — DDIM-20+50 baseline scores for reward normalization"
```

---

## Task 8: Create `scripts/precompute_baseline_scores.slurm`

**Files:**
- Create: `scripts/precompute_baseline_scores.slurm`

- [ ] **Step 1: Write the SLURM script**

Create `scripts/precompute_baseline_scores.slurm`:

```bash
#!/bin/bash
#SBATCH --job-name=precompute_baselines
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=/blue/ruogu.fang/pateld3/addiffusion/logs/precompute_baselines_%j.out
#SBATCH --error=/blue/ruogu.fang/pateld3/addiffusion/logs/precompute_baselines_%j.err

# Precompute DDIM-20 and DDIM-50 baseline scores for all training prompts.
# Must be run before training. Output: data/baseline_scores.json
#
# Usage: sbatch scripts/precompute_baseline_scores.slurm

set -e

module load cuda/12.8.0
module load gcc/11.4.0

cd "/blue/ruogu.fang/pateld3/addiffusion/"

export PYTHONUNBUFFERED=1
export HF_HOME=/blue/ruogu.fang/pateld3/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export TORCH_HOME=/blue/ruogu.fang/pateld3/.cache/torch

mkdir -p /blue/ruogu.fang/pateld3/addiffusion/logs
mkdir -p /blue/ruogu.fang/pateld3/addiffusion/data

echo "=== Precompute Baseline Scores ==="
echo "Job ID  : ${SLURM_JOB_ID}"
echo "Node    : ${SLURM_NODELIST}"
echo "GPU     : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Started : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "==================================="

uv run python -u /blue/ruogu.fang/pateld3/addiffusion/scripts/precompute_baseline_scores.py \
    --config /blue/ruogu.fang/pateld3/addiffusion/configs/default.yaml \
    --prompts_file /blue/ruogu.fang/pateld3/addiffusion/data/coco/annotations/captions_val2014.json \
    --output /blue/ruogu.fang/pateld3/addiffusion/data/baseline_scores.json \
    --max_prompts 10000 \
    --seed 42

echo "=== Done: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
```

- [ ] **Step 2: Commit**

```bash
git add scripts/precompute_baseline_scores.slurm
git commit -m "feat: precompute_baseline_scores.slurm — SLURM job for baseline score precomputation"
```

---

## Self-Review Checklist

**Spec coverage:**
- ✅ Terminal reward: dual-baseline normalized formula → Task 3
- ✅ Refine bonus: attention entropy → Tasks 2, 3, 4
- ✅ Entropy annealing schedule → Tasks 1, 5, 6
- ✅ Precomputation script + SLURM → Tasks 7, 8
- ✅ Config updates → Task 1
- ✅ `episode.py` passes `nfe_used` + `attention_entropy` → Task 4
- ✅ `ppo.py` `override_entropy_coeff` → Task 5
- ✅ `train.py` wired end-to-end → Task 6
- ✅ Fallback for unseen prompts → Task 3 (ddim20=0, norm=1)

**Type consistency:**
- `compute_attention_entropy(attention_maps: torch.Tensor) -> float` defined in Task 2, imported in Task 4 ✅
- `compute_refine_bonus(attention_entropy: float) -> float` defined in Task 3 ✅
- `compute_terminal_reward(image, prompt, nfe_used, n_max)` defined in Task 3, called in Task 3 (compute_reward) ✅
- `trainer.update(batch, override_entropy_coeff=...)` defined in Task 5, called in Task 6 ✅
- `reward_fn(..., nfe_used=..., attention_entropy=...)` closure updated in Task 6, called by episode.py Task 4 ✅

**Execution order on HPC after implementation:**
1. `sbatch scripts/precompute_baseline_scores.slurm` (prerequisite — wait for completion)
2. `sbatch scripts/train.slurm` (retrain from scratch)
3. `sbatch scripts/run_agent_inference.slurm` (DrawBench evaluation)
4. `sbatch scripts/eval_metrics.slurm agent` (G2 gate metrics)
