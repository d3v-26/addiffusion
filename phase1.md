# Phase 1: Core Agent Development — Completion Report

> Covers Weeks 2–4 of plan.md. All code traces to research.md §3 and discovery.md fixes.

---

## Summary

Phase 1 implements the full agent infrastructure: a step-by-step diffusion pipeline wrapper, cross-attention extraction, region refinement, state feature extraction, policy/value networks, reward computation, PPO training, and an end-to-end training script. Every critical discovery.md issue (D-09, D-10, D-11, D-13, D-14, D-16, D-19, D-26, D-37) is addressed in the implementation.

---

## Steps Performed

### Week 2: Diffusion Wrappers & State Features

| Step | Deliverable | Discovery Fixes Applied |
|------|-------------|------------------------|
| 1. Created project directory structure | `src/{agent,diffusion,rewards,evaluation,baselines,utils}/`, `configs/`, `scripts/`, `tests/` with `__init__.py` files | — |
| 2. Implemented diffusion pipeline wrapper | `src/diffusion/pipeline.py` — wraps StableDiffusionPipeline for step-by-step denoising with one-step clean prediction extraction | D-14 (latent space), D-15 (CFG documented), D-18 (episode structure) |
| 3. Implemented cross-attention extraction | `src/diffusion/attention.py` — pluggable interface with `UNetAttentionExtractor` (SD 1.5/SDXL) and `DiTAttentionExtractor` placeholder (Flux.1) | D-27 (pluggable UNet/DiT interface) |
| 4. Implemented state feature extraction | `src/agent/state.py` — computes φ(s) = [CLIP_img, CLIP_txt, timestep_emb, quality_vec] (~1672-dim) using frozen encoders | D-04 (frozen encoders), D-11 (DINO similarity not delta), D-16 (early-step unreliability noted) |

### Week 3: Agent Architecture & Actions

| Step | Deliverable | Discovery Fixes Applied |
|------|-------------|------------------------|
| 5. Implemented policy and value networks | `src/agent/networks.py` — Policy: Linear(d,512)→ReLU→Linear(512,256)→ReLU→Linear(256,3)→Softmax. Value: same with scalar output. | D-03 (|A|=3), D-38 (|A| explicit) |
| 6. Implemented RegionRefine with soft masks | `src/diffusion/refine.py` — mask generation from attention maps, Gaussian blur, k refinement iterations, post-refine re-noising | D-10 (NFE=1+k), D-19 (re-noise to next schedule point), D-26 (soft masks with σ=3) |
| 7. Implemented episode loop | `src/agent/episode.py` — full episode runner with 3-step mandatory warmup, action execution, transition collection | D-16 (warmup), D-18 (episode/scheduler interaction) |

### Week 4: Reward & PPO Training

| Step | Deliverable | Discovery Fixes Applied |
|------|-------------|------------------------|
| 8. Implemented reward computation | `src/rewards/reward.py` — R = R_quality + R_efficiency + R_terminal with per-component normalization | D-09 (c_nfe not γ), D-10 (NFE formula), D-11 (R_stability is similarity), D-13 (no λ₁,λ₂), D-37 (normalize before weighting) |
| 9. Implemented PPO training loop | `src/agent/ppo.py` — GAE, clipped objective, entropy bonus, gradient clipping | D-21 (index i not t), D-32 (curriculum support) |
| 10. Created training script and config | `src/train.py` + `configs/default.yaml` — main entry point with Hydra-style overrides, curriculum schedule, checkpointing | All hyperparameters from research.md §3.6 |
| 11. Created SLURM template | `scripts/train.slurm` — A100 job submission script | — |

---

## File Inventory

### Source Code

| File | Lines | What It Does | Research Reference |
|------|-------|--------------|-------------------|
| `src/diffusion/pipeline.py` | ~250 | Step-by-step diffusion inference: `prepare()` → `denoise_step()` → `advance_state()` → `decode()`. Supports DDIM, DPM-Solver, Euler, UniPC, PNDM schedulers. Extracts one-step clean predictions. All latent space. | §3.1, §3.2 |
| `src/diffusion/attention.py` | ~220 | Cross-attention map extraction via `StoringAttnProcessor`. Pluggable interface: `UNetAttentionExtractor` for SD 1.5/SDXL, `DiTAttentionExtractor` stub for Flux.1. Returns maps at (h, w, L). | §3.5 |
| `src/diffusion/refine.py` | ~170 | `RegionRefine` algorithm: generates soft mask from attention (Gaussian blur σ=3), re-noises masked regions, runs k=2 denoising iterations. Handles post-refine state transition. | §3.5 |
| `src/agent/state.py` | ~160 | State features φ(s_i): CLIP image (768d) + CLIP text (768d, cached) + sinusoidal timestep (128d) + quality vector (8d) = 1672d total. Frozen encoders. | §3.2 |
| `src/agent/networks.py` | ~120 | Policy (3-action softmax) and Value (scalar) networks. Two-layer MLP with orthogonal init. Provides `get_action()`, `evaluate_action()`, distribution access. | §3.3 |
| `src/agent/episode.py` | ~190 | Runs complete episodes: 3-step warmup → agent decisions → transitions. Tracks action sequences, NFE, rewards. Supports deterministic mode for evaluation. | §3.2 |
| `src/agent/ppo.py` | ~200 | PPO trainer: GAE advantage estimation (λ=0.95), clipped objective (ε=0.2), entropy bonus (c₂=0.01), value loss (c₁=0.5), gradient clipping (0.5). Mini-batch updates over K=4 epochs. | §3.6 |
| `src/rewards/reward.py` | ~230 | Composite reward: R_quality (normalized CLIP/ImageReward deltas + DINO stability) + R_efficiency (-c_nfe × NFE) + R_terminal (weighted final metrics). Lazy-loads CLIP, ImageReward, DINO. | §3.4 |
| `src/train.py` | ~200 | Main training script: loads config, initializes all components, runs training loop with curriculum (simple→mixed→full prompts), logging, checkpointing. | §3.6, EXP-T01 |
| `configs/default.yaml` | ~70 | All hyperparameters from research.md §3.6 in structured YAML. | §3.6 |
| `scripts/train.slurm` | ~30 | SLURM job template for A100 GPU training on HPC. | plan.md §Phase 0 |

### Test Scripts

| File | GPU Required | What It Tests | How to Run |
|------|-------------|---------------|------------|
| `tests/test_networks.py` | No | Policy/value dimensions, gradient flow, action space constants (|A|=3), deterministic mode | `uv run python tests/test_networks.py` |
| `tests/test_ppo.py` | No | GAE computation, discount propagation, episode→batch conversion, PPO update mechanics, learning signal verification | `uv run python tests/test_ppo.py` |
| `tests/test_pipeline.py` | Yes | Step-by-step denoising, early stopping, re-noising (D-19), multi-scheduler support, deterministic reproducibility | `uv run python tests/test_pipeline.py` |
| `tests/test_attention.py` | Yes | Attention map extraction shape (64×64×77), value validity, evolution across steps, per-token visualization, factory function | `uv run python tests/test_attention.py` |
| `tests/test_refine.py` | Yes | Mask generation from synthetic/real attention, threshold sensitivity, soft mask verification (D-26), NFE=1+k (D-10), full refine action with state transition (D-19) | `uv run python tests/test_refine.py` |
| `tests/test_state.py` | Yes | Feature dimensions (1672d), CLIP image encoding (768d, L2-norm), text caching, sinusoidal timestep embedding, quality vector, pipeline integration | `uv run python tests/test_state.py` |
| `tests/test_reward.py` | Yes | Efficiency reward (c_nfe, D-09), NFE formula (D-10), normalization toggle (D-37), CLIP score computation, DINO similarity (D-11: not a delta), composite reward (D-13: no lambdas) | `uv run python tests/test_reward.py` |
| `tests/test_episode.py` | Yes | Full episode with random policy, warmup enforcement (D-16), deterministic episodes, transition data quality for PPO | `uv run python tests/test_episode.py` |
| `tests/test_training_toy.py` | Yes | End-to-end integration: 2 iterations, batch=2, 10-step episodes. Verifies pipeline→episode→reward→PPO→update works without NaN/crashes | `uv run python tests/test_training_toy.py` |
| `tests/run_all.py` | — | Orchestrates all tests. Flags: `--cpu-only` (no GPU), `--no-integration` (skip slow test), `--model <id>` (custom weights) | `uv run python tests/run_all.py` |

---

## Discovery.md Issues Addressed

| ID | Issue | Where Fixed |
|----|-------|-------------|
| D-03 | Deterministic mask → |A|=3 | `networks.py`: NUM_ACTIONS=3 |
| D-04 | Frozen encoder state representation | `state.py`: all encoders @torch.no_grad |
| D-09 | γ notation collision | `reward.py`: uses `c_nfe`, config uses `gamma_d` |
| D-10 | NFE(refine) = 1+k not 1+k|m|/hw | `refine.py` line 98, `reward.py` line 168 |
| D-11 | ΔConsistency → R_stability (similarity) | `reward.py`: `dino_similarity()` returns [0,1] |
| D-13 | No outer λ₁, λ₂ | `reward.py`: R = R_quality + R_efficiency + R_terminal directly |
| D-14 | Latent space, not pixel space | `pipeline.py`: all z_t at h×w, decode only for metrics |
| D-16 | Early predictions unreliable | `episode.py`: 3 mandatory warmup continues |
| D-18 | Episode/scheduler interaction | `episode.py`: N_max matches scheduler, stop returns decoded z0_pred |
| D-19 | Post-refine state transition | `refine.py`: re-noise z0_refined to next schedule point |
| D-21 | PPO subscript collision | `ppo.py`: uses index i throughout |
| D-26 | Soft masks via Gaussian blur | `refine.py`: σ=3 blur before compositing |
| D-27 | Pluggable attention interface | `attention.py`: ABC with UNet/DiT subclasses |
| D-32 | Curriculum learning | `train.py`: simple→mixed→full schedule |
| D-37 | Normalize reward components | `reward.py`: divide by scale before weighting |
| D-38 | |A| explicitly stated | `networks.py`: NUM_ACTIONS=3, ACTION_NAMES list |

---

## Decision Gate G1 Criteria (plan.md)

Phase 1 delivers the infrastructure to evaluate G1 at the end of Week 4:

| Criterion | How to Test | Status |
|-----------|-------------|--------|
| Training loss decreases over 100 iterations on toy data | `uv run python tests/test_training_toy.py` (expand to 100 iters) | Ready to test |
| Agent actions are non-degenerate (not 100% one action) | Training logs report action distribution per iteration | Implemented |
| Reward components are in comparable scales after normalization | `uv run python tests/test_reward.py` verifies normalization | Implemented |
| One full training iteration completes in < 10 min on A100 | `sbatch scripts/train.slurm` and time first iteration | Ready to test |

**If G1 fails:** Debug rewards → simplify to stop/continue only → offline RL (per plan.md contingency).

---

## Connection to Phase 2

Phase 2 (Weeks 5–7) builds on Phase 1 in three areas:

### Week 5: Baseline Implementations
- Uses `src/diffusion/pipeline.py` to run fixed-step baselines (DDIM-20/50, DPM-Solver, Euler, etc.)
- Uses the scheduler registry in `pipeline.py` to switch schedulers
- New files: `src/baselines/fixed_step.py`, `src/baselines/sag.py`, `src/baselines/attend_excite.py`, `src/baselines/oracle.py`
- Baselines generate images that are compared against the agent (EXP-B01 through EXP-B04)

### Week 6: Evaluation Pipeline
- New files: `src/evaluation/compute_metrics.py`, `src/evaluation/vlm_judge.py`, `src/evaluation/significance.py`
- Reuses `src/rewards/reward.py` metric functions (CLIP score, ImageReward) for evaluation
- Adds FID, CMMD, IS, HPS v2, Aesthetic scoring
- Implements VLM-as-Judge protocol (research.md §4.6)

### Week 7: Hyperparameter Tuning
- Uses `src/train.py` with config overrides for sequential tuning and Optuna sweeps
- Tunes reward weights (α₁–α₄, c_nfe, β₁–β₃), learning rate, and entropy coefficient
- Evaluates tuned configs on DrawBench subset using the evaluation pipeline
- Produces the final hyperparameter configuration for Phase 3 full training

---

*Document Version: 1.0*
*Created: February 2026*
*Sources: plan.md (v1.0), research.md (v1.0), discovery.md (v1.0)*