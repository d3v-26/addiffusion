# Reward Redesign — Design Spec
**Date:** 2026-04-08  
**Status:** Approved  
**Addresses:** Always-continue policy collapse, never-refine behavior

---

## Problem

The trained agent (`agent_v1_final.pt`) always selects `continue` for all 50 steps across all 200 DrawBench prompts (avg NFE = 50.0). Two root causes:

1. **Always-continue**: Per-step quality deltas (CLIP/Aesthetic/ImageReward) are monotonically positive during DDIM denoising. The efficiency penalty (`c_nfe = 0.01355`) is too small to overcome cumulative quality gains. The terminal reward is highest at step 50 (cleanest image), so the agent rationally waits.

2. **Never-refine**: Refine costs 3× more NFE than continue (`c_nfe * 3` vs `c_nfe * 1`), but global quality metrics (CLIP, Aesthetic, ImageReward) are not sensitive to localized regional improvements. No explicit incentive exists for the agent to prefer refine over continue.

---

## Chosen Approach: Relative Terminal + Attention-Entropy Refine Bonus

Selected over two alternatives:
- **Approach A** (step-savings bonus only) — doesn't fix refine incentive
- **Approach C** (GRPO group-relative) — 4× more GPU rollouts, too expensive

---

## Reward Function Changes

### 1. Terminal Reward (replaces existing)

```python
R_terminal = sum(
    beta_k * (score_agent_k - score_ddim20_k[prompt])
    for k in [CLIP, Aesthetic, ImageReward]
) + c_save * (N_max - NFE_used) / N_max
```

- `score_ddim20_k[prompt]`: precomputed DDIM-20 baseline scores, loaded from `data/ddim20_baseline_scores.json`
- When agent matches DDIM-20 quality → quality term = 0, only step savings matter
- When agent beats DDIM-20 → positive bonus regardless of NFE used
- When agent stops early and matches DDIM-20 → earns full `c_save` bonus
- Fallback for unseen prompts: `score_ddim20_k = 0` (terminal reward becomes purely absolute)
- `c_save = 2.0` (initial; comparable to ~40 steps of typical per-step quality reward)

### 2. Refine Bonus (new, per-step, action-conditional)

```python
if action == ACTION_REFINE:
    # Cross-attention maps, averaged over all UNet decoder layers and heads → (h, w)
    A = mean_cross_attention_map(attention_maps)   # shape (64, 64)
    A = A / (A.sum() + 1e-8)                       # normalize to probability distribution
    H = -sum(A * log(A + 1e-8)) / log(h * w)      # normalized Shannon entropy, in [0, 1]
    R_refine_bonus = c_refine * H
```

- Uses cross-attention maps only (not self-attention) — these encode where the model attends to generate each token
- Averaged over all decoder layers and heads, same maps already used by `UNetAttentionExtractor` for mask computation — zero extra compute
- High entropy = diffuse/uncertain attention = under-generated region = refine genuinely useful
- Low entropy = sharp, focused attention = image is clean, refine is waste
- `c_refine = 0.2` (initial)

### 3. Unchanged

- `R_quality` — per-step delta CLIP/Aesthetic/ImageReward/DINO stability
- `R_efficiency` — `-c_nfe * NFE(action)`
- All alpha/beta weights from Optuna sweep

### Total Reward

```
R = R_quality + R_efficiency + R_terminal (if terminal) + R_refine_bonus (if action=refine)
```

---

## Entropy Schedule (PPO)

Entropy collapse (policy converging to always-continue despite `c_2=0.01`) is a confirmed failure mode in PPO for multi-action sequential tasks. Fix: anneal entropy coefficient from high to low.

```
entropy_coeff(iter) = 0.05   if iter < 500
                    = 0.01   if iter >= 500
```

Implemented by adding an optional `entropy_coeff` override parameter to `PPOTrainer.update(override_entropy_coeff=None)`. When provided, it replaces `self.config.entropy_coeff` for that update call. `src/train.py` computes the scheduled value and passes it each iteration. `PPOConfig.entropy_coeff` is kept as the default fallback.

---

## Precomputation

**New script:** `scripts/precompute_ddim20_scores.py`

- Runs DDIM-20 on every prompt in `data/coco/annotations/captions_val2014.json`
- Computes CLIP, Aesthetic, ImageReward for each output
- Writes `data/ddim20_baseline_scores.json`: `{prompt_str: {clip, aesthetic, image_reward}}`
- Must be run before retraining

**New SLURM job:** `scripts/precompute_ddim20_scores.slurm`

- Partition: `hpg-b200`, 1 GPU, ~2–3h wall time for full COCO val2014

---

## Config Changes (`configs/default.yaml`)

```yaml
ppo:
  entropy_coeff_start: 0.05
  entropy_coeff_end: 0.01
  entropy_anneal_steps: 500
  # entropy_coeff removed (now a schedule)

reward:
  ddim20_scores_path: "data/ddim20_baseline_scores.json"
  c_save: 2.0
  c_refine: 0.2
  # all existing alpha/beta/c_nfe values unchanged
```

---

## Files Changed

| File | Change |
|------|--------|
| `src/rewards/reward.py` | New terminal reward formula, refine bonus, load scores JSON |
| `src/train.py` | Entropy schedule: pass annealed value to PPO update per iteration |
| `configs/default.yaml` | New `c_save`, `c_refine`, `ddim20_scores_path`, entropy schedule params |
| `scripts/precompute_ddim20_scores.py` | New: run DDIM-20, compute scores, write JSON |
| `scripts/precompute_ddim20_scores.slurm` | New: SLURM wrapper for precomputation |

## Files Unchanged

- `src/agent/episode.py` — reward_fn interface unchanged
- `src/agent/networks.py` — architecture unchanged
- `src/agent/ppo.py` — PPO update logic unchanged (entropy_coeff passed as arg, not stored)
- `src/agent/state.py` — unchanged
- `scripts/run_agent_inference.py` — unchanged (reward_fn=None at inference)
- `scripts/run_agent_inference.slurm` — unchanged

---

## Retraining

Full 2000 iterations from scratch. `agent_v1_final.pt` was trained with a broken reward structure and is not worth continuing from.

Order of operations:
1. Implement code changes
2. Run `precompute_ddim20_scores.slurm` (prerequisite)
3. Retrain with `train.slurm`
4. Run `run_agent_inference.slurm` on DrawBench
5. Check action distribution — expect mix of continue/stop/refine
6. Run `eval_metrics.slurm` for G2 gate comparison

---

## Success Criteria

- Action distribution: not all-continue (any stop/refine usage is a win)
- Mean NFE < 50 across DrawBench prompts
- G2 gate: agent CLIP/ImageReward >= DDIM-20 at equal or lower NFE

---

## Hyperparameters to Watch

| Param | Initial Value | Risk if Too Low | Risk if Too High |
|-------|--------------|-----------------|------------------|
| `c_save` | 2.0 | Agent still prefers continue to step 50 | Agent stops at step 3 (before quality is adequate) |
| `c_refine` | 0.2 | Refine never selected | Refine selected even on clean images |
| `entropy_coeff_start` | 0.05 | Entropy collapse early in training | Slow convergence |

If agent action distribution still collapses to always-continue after retraining, the next escalation is increasing `c_save` to 4.0–6.0 or running a narrow Optuna sweep over `{c_save, c_refine}`.
