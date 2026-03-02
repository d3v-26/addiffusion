# Phase 2: Baselines, Evaluation Pipeline & Hyperparameter Tuning — Completion Report

> Covers Weeks 5–7 of plan.md. Builds directly on Phase 1's agent infrastructure.
> Decision gate: **G2** — best-tuned config outperforms DDIM-20 on ≥1 metric at comparable NFE.

---

## Summary

Phase 2 establishes the comparison framework and tuning infrastructure needed to evaluate the Phase 1 agent and configure Phase 3 full training. Three things are built:

1. **Baselines (Week 5):** Eight comparison methods ranging from simple fixed-step schedulers to distillation models, all wrapped in a uniform `BaseBaseline → BaselineResult` interface.
2. **Evaluation pipeline (Week 6):** A complete metrics stack (FID, CLIP, CMMD, IS, ImageReward, HPS v2, Aesthetic) plus a GPT-4o pairwise VLM judge and statistical significance testing.
3. **Hyperparameter tuning (Week 7):** Sequential LR sweep, three-profile reward comparison, and a 20-trial Optuna sweep over all reward weights — using NFS-safe JournalStorage for SLURM parallelism.

All baselines reuse `src/diffusion/pipeline.py` from Phase 1. The evaluation pipeline reuses the CLIP/ImageReward model loading from `src/rewards/reward.py`. The tuning scripts drive the Phase 1 training loop (`src/train.py`) with config overrides.

---

## Connection to Phase 1

Phase 2 depends on the following Phase 1 deliverables:

| Phase 1 File | Used By Phase 2 As |
|---|---|
| `src/diffusion/pipeline.py` | `OracleStopBaseline` steps through its loop; all fixed-step baselines use the scheduler registry indirectly via `load_sd15_pipeline()` |
| `src/rewards/reward.py` (`RewardComputer`) | `OracleStopBaseline` and `RandomSearchBaseline` score via CLIP; tune scripts use `RewardConfig` for sweeping |
| `src/agent/networks.py` (`PolicyNetwork`, `ValueNetwork`) | Tune scripts instantiate networks for training |
| `src/agent/state.py` (`StateExtractor`) | Tune scripts use `StateExtractor.TOTAL_DIM = 1672` as the network input dimension |
| `src/agent/episode.py` (`EpisodeRunner`) | Tune scripts collect episodes in the inner training loop |
| `src/agent/ppo.py` (`PPOTrainer`, `PPOConfig`) | Tune scripts configure and call `PPOTrainer.update()` |
| `src/train.py` | Tune scripts mirror its training loop structure |
| `configs/default.yaml` | Baseline reward params (β₁–β₃) pulled from here; Hydra config extended with `tuned_reward.yaml` |

**Key dependency:** `StateExtractor.TOTAL_DIM = 1672` (768 CLIP-image + 768 CLIP-text + 128 sinusoidal timestep + 8 quality vector). All network constructors in Phase 2 tuning scripts use this value, not the approximate `d~1280` stated in the architecture overview.

---

## Steps Performed

### Week 5: Baseline Implementations

| Step | Deliverable | Notes |
|------|-------------|-------|
| 1. Foundation | `src/baselines/base.py` — `BaseBaseline` ABC, `BaselineResult` dataclass, `load_sd15_pipeline()`, `save_results()` | Uniform interface: every baseline returns `list[BaselineResult]`, writes PNGs + `results.jsonl` + `metadata.json` |
| 2. Fixed-step schedulers | `src/baselines/fixed_step.py` — DDIM20/50, DPMSolver20/50, Euler20, UniPC20, PNDM20 | `DPMSolver`: algorithm_type default used (not overridden). `PNDM`: skip_prk_steps left at default. All schedulers via `from_config()` |
| 3. Oracle upper bound | `src/baselines/oracle.py` — `OracleStopBaseline` | Runs all steps via `AdaptiveDiffusionPipeline`, scores each `z0_pred` with CLIP ViT-L/14, returns image at peak NFE. `nfe = -1` class constant; actual peak recorded per-image |
| 4. Random search | `src/baselines/random_search.py` — `RandomSearchBaseline` | Best-of-8 by CLIP score. Seeds offset by `+i*1000` per candidate for independence. NFE = 8 × 20 = 160 |
| 5. Self-Attention Guidance | `src/baselines/sag.py` — `SAGBaseline` | `StableDiffusionSAGPipeline`, 20 steps, `sag_scale=0.75` (paper default) |
| 6. Attend-and-Excite | `src/baselines/attend_excite.py` — `AttendExciteBaseline` | 50 steps (minimum for A&E effectiveness). `token_indices` = all non-BOS/EOS positions. `max_iter_to_alter=25` |
| 7. LCM-LoRA | `src/baselines/lcm.py` — `LCMBaseline` | `load_lora_weights()` + `fuse_lora()` + `LCMScheduler`. `guidance_scale=0` (CFG fully off — per official LCM docs) |
| 8. SDXL-Turbo | `src/baselines/sdxl_turbo.py` — `SDXLTurboBaseline` | `AutoPipelineForText2Image`, `variant="fp16"`, `guidance_scale=0.0` (mandatory). Requires A100-80GB (~24 GB VRAM) |
| 9. Exports | `src/baselines/__init__.py` | Exports all 8 baseline classes + `ALL_FIXED_STEP_BASELINES` |
| 10. Tests | `tests/test_baselines_fixed/search/attention/distill.py` | One-prompt smoke tests per class: image shape (512×512), `nfe > 0`, `time_s > 0`, `results.jsonl` written |

### Week 6: Evaluation Pipeline

| Step | Deliverable | Notes |
|------|-------------|-------|
| 11. Metrics | `src/evaluation/compute_metrics.py` — `MetricsComputer` | Lazy-loaded models. FID via `cleanfid`, CLIP via `open_clip` ViT-L/14, CMMD manual (RBF kernel MMD, σ=10.0, L2-normalised embeds), IS via `torch_fidelity`, ImageReward, HPS v2, Aesthetic (hub-loaded, gracefully skipped on failure) |
| 12. VLM judge | `src/evaluation/vlm_judge.py` — `VLMJudge` | GPT-4o (`gpt-4o-2024-11-20`), `client.chat.completions.create()`, base64 PNG in `image_url` content blocks with `"detail": "high"`. 50% A/B order swap for positional-bias mitigation. Incremental JSONL caching (resume support). Bradley-Terry win-probability fitting |
| 13. Significance | `src/evaluation/significance.py` | Paired bootstrap (scipy), Wilson score CI, `compare_all_methods()` loads per-method JSON files and runs tests vs reference method, `print_latex_table()` for paper |
| 14. Tests | `tests/test_compute_metrics/vlm_judge/significance.py` | CPU-safe: CMMD kernel shape/symmetry/self-similarity, mock OpenAI API for VLM judge, bootstrap calibration tests |
| 15. Baseline runner | `scripts/run_baselines.py` + `scripts/run_baselines.slurm` | Dynamic baseline loading via `importlib`. Registry of all 13 baselines by short name. Per-baseline error isolation |
| 16. Metrics runner | `scripts/eval_metrics.slurm` | SLURM wrapper calling `compute_metrics.py` for a given method's output directory |

### Week 7: Hyperparameter Tuning

| Step | Deliverable | Notes |
|------|-------------|-------|
| 17. LR sweep | `scripts/tune_lr.py` + `scripts/tune_lr.slurm` | {1e-4, 3e-4, 1e-3} × 500 iters. Saves `results.json` (best LR, per-LR final-50-iter stats) and `curves.png`. Default 3e-4 expected to win |
| 18. Reward profiles | `scripts/tune_reward.py` | 3 profiles × 500 iters. Best profile written to `configs/tuned_reward.yaml` (Hydra-compatible, `reward:` key) |
| 19. Optuna sweep | `scripts/optuna_sweep.py` + `scripts/optuna_sweep.slurm` | 20 trials × 200 iters. `JournalStorage(JournalFileBackend(...))` — NFS-safe, no SQLite. Objective: `mean_reward_final50 - 0.5 × std_reward`. Top-3 configs written to `configs/optuna_top3.yaml` |

---

## File Inventory

### Source Code

| File | What It Does | Used By |
|------|--------------|---------|
| `src/baselines/base.py` | Abstract base class + result container + shared utilities | All baselines |
| `src/baselines/fixed_step.py` | 7 fixed-step scheduler baselines | EXP-B01 comparisons |
| `src/baselines/oracle.py` | CLIP-optimal early-stopping upper bound (B4) | H1 ceiling analysis |
| `src/baselines/random_search.py` | Best-of-8 search baseline (B2) | NFE-controlled comparison |
| `src/baselines/sag.py` | Self-Attention Guidance (B6) | Quality-boosted comparison |
| `src/baselines/attend_excite.py` | Attend-and-Excite (B6) | Compositional quality comparison |
| `src/baselines/lcm.py` | LCM-LoRA SD 1.5, 4-step (B5) | Low-NFE distillation comparison |
| `src/baselines/sdxl_turbo.py` | SDXL-Turbo, 4-step (B5) | Cross-architecture distillation comparison |
| `src/evaluation/compute_metrics.py` | All quality/efficiency metrics; CLI interface | Phase 3 evaluation loop |
| `src/evaluation/vlm_judge.py` | GPT-4o pairwise judge + Bradley-Terry | research.md §4.6 |
| `src/evaluation/significance.py` | Paired bootstrap + Wilson CI + LaTeX table | Paper §5 statistics |
| `scripts/run_baselines.py` | Dynamic baseline execution runner | SLURM baseline jobs |
| `scripts/tune_lr.py` | Sequential LR sweep with curves | G2 configuration |
| `scripts/tune_reward.py` | Three reward-profile comparison | G2 configuration |
| `scripts/optuna_sweep.py` | 20-trial Optuna reward hyperparameter sweep | G2 configuration |

### SLURM Scripts

| File | GPU | Time | Purpose |
|------|-----|------|---------|
| `scripts/run_baselines.slurm` | 1× A100-40GB | 4 h | Run any set of baselines on DrawBench |
| `scripts/eval_metrics.slurm` | 1× A100-40GB | 2 h | Compute all metrics for one method's output dir |
| `scripts/tune_lr.slurm` | 1× A100-40GB | 6 h | LR sweep (3 configs × 500 iters) |
| `scripts/optuna_sweep.slurm` | 1× A100-40GB | 12 h | Optuna sweep (20 trials × 200 iters, resumable) |

### Test Files

| File | GPU? | Tests |
|------|------|-------|
| `tests/test_baselines_fixed.py` | Yes | DDIM20/50, DPMSolver20/50, Euler20, UniPC20, PNDM20 smoke tests |
| `tests/test_baselines_search.py` | Yes | RandomSearch (n=2), OracleStop (max_steps=5) |
| `tests/test_baselines_attention.py` | Yes | SAG, Attend-and-Excite + token index unit test |
| `tests/test_baselines_distill.py` | Yes | LCM-4, SDXL-Turbo-4 (VRAM guard) |
| `tests/test_compute_metrics.py` | No | RBF kernel shape/symmetry, CMMD self-similarity≈0, Wilson CI coverage |
| `tests/test_vlm_judge.py` | No | Mock OpenAI API: pair comparison, order randomization, win-rate aggregation, Bradley-Terry |
| `tests/test_significance.py` | No | Bootstrap calibration, Wilson CI known values, p-value uniformity under H₀ |

---

## CMMD Implementation Note

CMMD has no pip package. The implementation in `compute_metrics.py` follows arXiv:2401.09603 exactly:

```python
def compute_cmmd(real_embeds, gen_embeds, sigma=10.0):
    # L2-normalise (CLIP output is unit-norm, but this is enforced explicitly)
    real_embeds = real_embeds / np.linalg.norm(real_embeds, axis=1, keepdims=True)
    gen_embeds  = gen_embeds  / np.linalg.norm(gen_embeds,  axis=1, keepdims=True)
    xx = rbf_kernel(real_embeds, real_embeds, sigma)
    yy = rbf_kernel(gen_embeds,  gen_embeds,  sigma)
    xy = rbf_kernel(real_embeds, gen_embeds,  sigma)
    return float(xx.mean() + yy.mean() - 2 * xy.mean())
```

Reference implementations: `google-research/google-research/cmmd` (JAX), `sayakpaul/cmmd-pytorch`.

---

## Verified API Corrections Made During Phase 2

Several API assumptions in the initial plan were corrected through pre-flight testing:

| Issue | Wrong Assumption | Correct Behavior |
|-------|-----------------|-----------------|
| HPS v2 input type | Takes `list[path]` | Takes `PIL.Image` + `str`; returns `list[float]` |
| clean-fid function | `fid.list_custom_stats()` exists | Use `fid.test_stats_exists()` |
| DrawBench URL | `parti` GitHub CSV accessible | URL returns 404; use `hf_hub_download(repo_id="sayakpaul/drawbench", ...)` |
| DPMSolver kwarg | `algorithm_type="dpmsolver++"` needed | Already the default; omit |
| PNDM kwarg | `skip_prk_steps=True` | Default is `False`; do not change |
| LCM guidance | `guidance_scale=1.0` | Should be `0` (CFG fully off) |
| VLM judge SDK | `client.messages.create()` | OpenAI SDK: `client.chat.completions.create()` |
| Optuna storage | SQLite | `JournalStorage(JournalFileBackend(...))` — SQLite is NOT NFS-safe on SLURM |

---

## Decision Gate G2 Criteria (plan.md)

G2 is evaluated after Week 7 tuning by training the top Optuna config for 500 iterations and comparing to DDIM-20 on DrawBench:

```bash
# Train top Optuna config
sbatch scripts/train.slurm --config configs/optuna_top1.yaml --n_iters 500 \
    --output_dir outputs/tuning/final/optuna_top1/

# Evaluate
sbatch scripts/eval_metrics.slurm --method optuna_top1
```

| Criterion | Pass Condition |
|-----------|---------------|
| CLIP Score vs DDIM-20 | Agent ≥ DDIM-20 mean (p < 0.05 via paired bootstrap) |
| ImageReward vs DDIM-20 | Agent ≥ DDIM-20 mean (p < 0.05 via paired bootstrap) |
| FID vs DDIM-20 | Agent FID ≤ DDIM-20 FID |
| NFE budget | Mean agent NFE ≤ 30 (60% of DDIM-50 budget) |

**G2 pass condition:** ≥ 1 quality metric beats DDIM-20 at comparable or lower NFE.

**If G2 fails:** Use `quality_heavy` reward profile instead (more aggressive CLIP/ImageReward weighting). If still failing, extend tuning to 1K iterations or investigate reward signal quality.

---

## Connection to Phase 3

Phase 3 (Weeks 8–10) builds directly on these Phase 2 deliverables:

### Full Training (Week 8)
- Uses the best Optuna config from `configs/optuna_top3.yaml` as the training configuration
- `scripts/tune_lr.py` result determines the final learning rate
- Trains for 5K iterations with curriculum (simple → mixed → full prompts from COCO-118K)
- Checkpoints to `checkpoints/` every 50 iterations

### Ablation Studies (Week 9)
- Uses `src/evaluation/compute_metrics.py` to score each ablation's outputs
- Uses `src/evaluation/significance.py` to report p-values in comparison tables
- Ablations: A1 (remove reward components), A2 (stop/continue only), A3 (state features), A4 (region selection), A5 (no reward normalization)
- Each ablation is a config override on the trained agent; baselines already generated in Phase 2 serve as the comparison set

### Full Benchmarks (Week 10)
- Runs `scripts/run_baselines.py` for all 13 baselines on the full DrawBench + PartiPrompts + COCO-30K sets
- Runs `src/evaluation/compute_metrics.py` for FID, CLIP, CMMD across all methods
- Runs `src/evaluation/vlm_judge.py` for the agent vs DDIM-50 head-to-head (200 pairs)
- Compares against all Phase 2 baseline outputs; statistical significance via `significance.py`
- Decision gate G3: H1 test (agent NFE ≤ 30 matches DDIM-50 quality)

---

*Document Version: 1.0*
*Created: March 2026*
*Sources: plan.md (v1.0), research.md (v1.0), discovery.md (v1.0), phase2-plan.md (internal)*
