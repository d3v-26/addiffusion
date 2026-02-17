# AdDiffusion: Research Execution Plan

> Actionable project plan derived from `research.md` (v1.0). All phase references, risk IDs, and experiment IDs trace back to `research.md` and `discovery.md`.

---

## Objective Statement

This plan guides the execution of the AdDiffusion project: training and evaluating an RL-based adaptive inference agent for diffusion models. The plan is organized into 6 sequential phases spanning 14 weeks, covering environment setup, baseline implementation, core agent development, ablation studies, full-scale evaluation, and paper writing. Each phase has explicit deliverables, success criteria, risk checkpoints, and decision gates. The target outcome is a submission-ready paper demonstrating that agent-based adaptive inference achieves equivalent quality to 50-step fixed inference at $\leq 30$ average NFE with interpretable prompt-dependent behavior.

---

## Phase Breakdown

### Phase 0: Environment & Infrastructure Setup
**Timeline:** Week 1 (5 days)
**Dependencies:** None
**Risk Checkpoint:** discovery.md D-25 (GPU memory), D-14 (latent space)

| Day | Task | Deliverable |
|-----|------|-------------|
| 1 | Create project directory structure per `setup_guide.md` | Working directory layout |
| 1 | Install uv, create virtual environment, install all dependencies | Passing `verify_setup.py` |
| 2 | Download diffusion models (SD 1.5, SDXL, LCM-LoRA, SDXL-Turbo) | Models cached in `$HF_HOME` |
| 2 | Download prompt datasets (COCO-30K, DrawBench, PartiPrompts, GenEval, T2I-CompBench) | Data in `data/` |
| 3 | Download and verify metric model weights (ImageReward, CLIP, DINO) | All metric models loadable |
| 3 | Create base config `configs/default.yaml` per `setup_guide.md` | Config file with all hyperparameters |
| 4 | Run sanity test: generate one image with SD 1.5 pipeline | `outputs/sanity_check.png` |
| 4 | Profile GPU memory for SD 1.5 + CLIP + ImageReward co-loading | Memory audit document |
| 5 | Set up W&B project, SLURM templates, git repository | Logging and job infrastructure |

**Success Criteria:**
- `verify_setup.py` reports all checks passed
- Sanity check image generates without OOM
- GPU memory audit confirms feasibility of co-loading models on A100-40GB (for SD 1.5)

**Decision Gate G0:** If SD 1.5 + all metric models exceed 35GB on A100-40GB → implement serialized reward computation (load/unload models per component) before proceeding.

---

### Phase 1: Core Agent Development
**Timeline:** Weeks 2–4 (15 days)
**Dependencies:** Phase 0 complete
**Risk Checkpoint:** discovery.md D-10 (NFE formula), D-14 (latent space), D-18 (episode structure), D-19 (refine interaction)

#### Week 2: Diffusion Wrappers & State Features

| Day | Task | Deliverable |
|-----|------|-------------|
| 1–2 | Implement diffusion inference wrapper (`src/diffusion/pipeline.py`) supporting DDIM, DPM-Solver schedulers | Wrapper that generates images with step-by-step access |
| 3 | Implement one-step clean prediction ($\hat{z}_0^{(i)}$) extraction from scheduler internals | `predict_clean()` function |
| 4 | Implement state feature extraction (`src/agent/state.py`): CLIP-img, CLIP-txt, timestep embedding, quality vector | $\phi(s)$ computation with correct dimensions |
| 5 | Implement cross-attention map extraction for UNet (`src/diffusion/attention.py`) | Attention maps at latent resolution |

**Validation:** State features have expected dimensions (d ≈ 1280). Attention maps are $h \times w \times L$. All operations work in latent space (discovery.md D-14).

#### Week 3: Agent Architecture & Actions

| Day | Task | Deliverable |
|-----|------|-------------|
| 1 | Implement policy network $\pi_\psi$ and value network $V_\omega$ (`src/agent/networks.py`) | Networks with correct I/O dimensions |
| 2 | Implement action execution: `continue`, `stop` (`src/agent/actions.py`) | Working continue/stop actions |
| 3 | Implement `RegionRefine` algorithm with soft masks (`src/diffusion/refine.py`) | Refinement with Gaussian-blurred masks (D-26) |
| 3 | Implement post-refinement state transition: re-noise to next schedule point (D-19) | Correct $z_{t_{i+1}}$ after refinement |
| 4 | Implement episode loop: agent interacts with diffusion pipeline (D-18) | Full episode with warmup (3 mandatory continues) |
| 5 | Unit tests for all components | Test suite passing |

**Validation:** Agent can run a complete episode (noise → decisions → final image). Refine action produces images without boundary artifacts. Episode respects $N_\text{max}$.

#### Week 4: Reward & PPO Training

| Day | Task | Deliverable |
|-----|------|-------------|
| 1 | Implement reward computation (`src/rewards/`) with normalization (D-37) | $R(s, a, s')$ computation |
| 1 | Verify reward component scales match normalization targets | Scale validation log |
| 2–3 | Implement PPO training loop (`src/agent/ppo.py`) with GAE, clipping, entropy bonus | Training script |
| 4 | Debug on toy dataset (10 prompts, 100 iterations) | Training converges; loss decreases |
| 5 | Full integration test: 64-batch training iteration on COCO subset | One PPO update completes without errors |

**Success Criteria:**
- Training loss decreases over 100 iterations on toy dataset
- Agent actions are non-degenerate (not 100% one action)
- Reward components are in comparable scales after normalization
- One full training iteration (64 trajectories → PPO update) completes in < 10 minutes on A100

**Decision Gate G1:** If training does not converge after 200 iterations on toy data → check reward scales (D-37), increase entropy coefficient, simplify to stop/continue only (fallback per research.md §6.4).

---

### Phase 2: Baselines & Hyperparameter Tuning
**Timeline:** Weeks 5–7 (15 days)
**Dependencies:** Phase 1 complete (working agent prototype)
**Risk Checkpoint:** discovery.md D-22 (AdaDiff citation), D-29 (sample efficiency)

#### Week 5: Baseline Implementations

| Day | Task | Deliverable |
|-----|------|-------------|
| 1 | Implement fixed-step baselines B1 (`src/baselines/fixed_step.py`): DDIM-20/50, DPM-Solver-20/50, Euler-20, UniPC-20, PNDM-20 | All 7 baselines generating images |
| 2 | Implement SAG baseline B6 (via diffusers `StableDiffusionSAGPipeline`) | SAG generating images |
| 2 | Implement Attend-and-Excite baseline B6 (via diffusers `StableDiffusionAttendAndExcitePipeline`) | A&E generating images |
| 3 | Implement LCM baseline B5 (via diffusers + LCM-LoRA) | LCM-4 generating images |
| 3 | Implement SDXL-Turbo baseline B5 | SDXL-Turbo generating images |
| 4 | Implement Random-Search-N baseline B2 | Best-of-8 with CLIP scoring |
| 5 | Implement Oracle-Stop baseline B4 (evaluate CLIP at each step, stop at peak) | Oracle generating images |

#### Week 6: Evaluation Pipeline

| Day | Task | Deliverable |
|-----|------|-------------|
| 1–2 | Implement full metric computation (`src/evaluation/`): FID, CLIP Score, CMMD, Aesthetic, IS, ImageReward, HPS v2 | Metric computation pipeline |
| 3 | Implement VLM-as-Judge evaluation script (GPT-4o API, pairwise comparison) | VLM evaluation script |
| 4 | Run all baselines on DrawBench (200 prompts) | Baseline results on DrawBench |
| 5 | Validate metrics match expected ranges from literature | Metric sanity check report |

#### Week 7: Hyperparameter Tuning

| Day | Task | Deliverable |
|-----|------|-------------|
| 1–2 | Sequential tuning Round 1: learning rate {1e-4, 3e-4, 1e-3} on COCO-1K subset, 500 iterations each | Best learning rate |
| 3 | Sequential tuning Round 2: reward ratios {quality-heavy, balanced, efficiency-heavy} | Best reward ratio profile |
| 4 | Optuna sweep (20 trials, 200 iterations each): $\alpha_1$–$\alpha_4$, $c_\text{nfe}$, $\beta_1$–$\beta_3$ | Top-3 hyperparameter configs |
| 5 | Full training with top-3 configs (500 iterations each), evaluate on DrawBench subset | Selected final config |

**Success Criteria:**
- All baselines produce images at expected quality levels
- Metric pipeline matches literature-reported values (±5%)
- Best hyperparameter config shows clear improvement over random policy

**Decision Gate G2:** If after tuning, the agent does not outperform DDIM-20 on any metric → investigate: (1) reward function debugging, (2) state feature quality, (3) fallback to offline RL (discovery.md D-33). If agent outperforms DDIM-20 but not DDIM-50 at comparable NFE → proceed but lower H1 expectations.

---

### Phase 3: Full-Scale Training & Evaluation
**Timeline:** Weeks 8–10 (15 days)
**Dependencies:** Phase 2 complete (baselines working, hyperparameters selected)
**Risk Checkpoint:** discovery.md D-08 (adaptive behavior), D-28 (reward hacking), D-29 (sample inefficiency)

#### Week 8: Full Agent Training

| Day | Task | Deliverable |
|-----|------|-------------|
| 1–3 | Train agent on COCO-30K prompts with curriculum (D-32): simple → medium → complex | Agent v1.0 checkpoint |
| 4 | Monitor training: check for reward hacking (D-28) via VLM spot checks on 20 generated images | Training health report |
| 5 | Evaluate agent v1.0 on DrawBench; compare against baselines | Preliminary results table |

#### Week 9: Ablation Studies

| Day | Task | Deliverable |
|-----|------|-------------|
| 1 | Run ablation A1 (reward components): 4 variants, 1000 iterations each | A1 results table |
| 2 | Run ablation A2 (action space): 4 variants, 1000 iterations each | A2 results table |
| 3 | Run ablation A3 (state representation): 4 variants, 1000 iterations each | A3 results table |
| 4 | Run ablation A4 (region selection): 4 variants, 1000 iterations each | A4 results table |
| 5 | Run ablation A5 (reward normalization): 2 variants, 1000 iterations each | A5 results table |

#### Week 10: Full Benchmark Evaluation

| Day | Task | Deliverable |
|-----|------|-------------|
| 1 | COCO-30K evaluation (30K images × 5 seeds for FID confidence) | FID, CLIP Score with 95% CI |
| 2 | PartiPrompts evaluation (1.6K prompts) with per-category breakdown | Per-category analysis |
| 3 | GenEval evaluation (counting, spatial reasoning) | Accuracy metrics |
| 3 | T2I-CompBench evaluation (attribute binding, spatial, non-spatial) | Compositional scores |
| 4 | VLM-as-Judge evaluation on DrawBench + PartiPrompts subset (GPT-4o + Gemini) | Win rates with Bradley-Terry model |
| 5 | Statistical significance testing: paired bootstrap, confidence intervals | Statistical report |

**Success Criteria:**
- H1 confirmed: Avg NFE $\leq 30$ with CLIP $\geq$ DDIM-50
- H3 testable: NFE varies across PartiPrompts categories
- All ablations produce interpretable results
- No evidence of reward hacking in VLM spot checks

**Decision Gate G3:** If H1 is not confirmed (NFE > 35 or CLIP < DDIM-50):
- If close (NFE 30–35): proceed with adjusted claims
- If far (NFE > 40 or CLIP significantly below): pivot to "quality improvement at equal budget" framing (H2 as primary)
- If agent learns degenerate policy: fall back to No-Refine variant from A2 ablation

---

### Phase 4: Generalization & Deep Analysis
**Timeline:** Weeks 11–12 (10 days)
**Dependencies:** Phase 3 complete
**Risk Checkpoint:** discovery.md D-27 (UNet → DiT), D-30 (stochastic samplers)

#### Week 11: Generalization Experiments

| Day | Task | Deliverable |
|-----|------|-------------|
| 1 | Test SD 1.5-trained agent on SDXL (zero-shot transfer) | Transfer results |
| 2 | If zero-shot fails (>5% CLIP drop): train architecture-specific head for SDXL (D-27) | SDXL-adapted agent |
| 3 | Test on Flux.1-schnell with DiT attention interface | DiT transfer results |
| 4 | Test with alternative schedulers (Euler, UniPC, PNDM) without retraining (D-30) | Scheduler robustness results |
| 5 | Cross-dataset evaluation: train on COCO, eval on DrawBench/PartiPrompts | Robustness analysis |

#### Week 12: Deep Analysis

| Day | Task | Deliverable |
|-----|------|-------------|
| 1 | Visualize agent behavior: action distribution across timesteps | Action distribution figure |
| 2 | Stopping time distribution per prompt complexity tier | Stopping histogram figure |
| 3 | Refinement mask visualization: sample masks for diverse prompts | Mask gallery figure |
| 4 | Quality trajectory visualization: CLIP/ImageReward over denoising steps | Trajectory figure |
| 5 | Failure case analysis: identify prompt categories where agent underperforms | Failure analysis document |

**Success Criteria:**
- Agent transfers to at least one new backbone (SDXL or Flux.1) without catastrophic failure
- Visualizations show interpretable adaptive behavior
- Failure cases are identified and documented

**Decision Gate G4:** If zero-shot transfer fails on all new backbones:
- Report SD 1.5 results as primary; note generalization as a limitation
- If architecture-specific heads work: report both zero-shot and adapted results

---

### Phase 5: Paper Writing & Finalization
**Timeline:** Weeks 13–14 (10 days)
**Dependencies:** Phase 4 complete

#### Week 13: Paper Draft

| Day | Task | Deliverable |
|-----|------|-------------|
| 1 | Write Abstract + Introduction (based on research.md §1) | Draft sections |
| 2 | Write Related Work (based on research.md §2) | Draft section |
| 3 | Write Method (based on research.md §3, incorporating all D-xx fixes) | Draft section |
| 4 | Write Experiments (based on research.md §4 + actual results) | Draft section |
| 5 | Write Results & Analysis (based on Phase 3–4 outputs) | Draft section |

#### Week 14: Finalization

| Day | Task | Deliverable |
|-----|------|-------------|
| 1 | Write Limitations (based on research.md §6), Conclusion, Future Work | Draft sections |
| 2 | Create all figures: Pareto plots, action distributions, masks, trajectories | Publication-ready figures |
| 3 | Create all tables: main results, ablations, generalization | Publication-ready tables |
| 4 | Internal review: check consistency, cross-references, statistical claims | Revised draft |
| 5 | Final polish, supplementary materials, code cleanup | Submission-ready paper |

**Success Criteria:**
- Complete paper with all sections, figures, tables, and references
- All claims supported by experimental evidence with statistical significance
- Code repository cleaned and reproducible

---

## Resource Requirements

### Compute

| Component | GPU Hours | GPU Type | Phase |
|-----------|-----------|----------|-------|
| Environment setup & verification | 2 | A100-40GB | 0 |
| Agent development & debugging | 20 | A100-40GB | 1 |
| Baseline evaluation (all) | 80 | A100-40GB | 2 |
| Hyperparameter search | 45 | A100-40GB | 2 |
| Full agent training (SD 1.5) | 200 | A100-40GB | 3 |
| Ablation studies (18 variants) | 150 | A100-40GB | 3 |
| Full benchmark evaluation (COCO-30K × 5 seeds + others) | 100 | A100-40GB | 3 |
| Generalization experiments (SDXL, Flux.1) | 200 | A100-80GB | 4 |
| VLM-as-Judge API calls | ~$200 | GPT-4o API | 3–4 |
| **Total** | **~797 + $200 API** | — | — |

> Budget of ~930 GPU hours (from experimental_design_adaptive_diffusion.md §6.1) provides ~15% margin.

### Storage

| Data | Size | Phase |
|------|------|-------|
| Diffusion model weights | ~30 GB | 0 |
| Prompt datasets | ~5 GB | 0 |
| Metric model weights | ~5 GB | 0 |
| COCO-30K generated images (5 seeds × 30K × multiple methods) | ~80 GB | 3 |
| Agent checkpoints | ~5 GB | 1–3 |
| Evaluation outputs and logs | ~30 GB | 2–4 |
| **Total** | **~155 GB** | — |

### Software

| Tool | Purpose |
|------|---------|
| PyTorch 2.6.0 + CUDA 12.4 | Deep learning framework |
| diffusers | Diffusion model pipelines |
| open_clip_torch | CLIP scoring + state features |
| image-reward | ImageReward metric |
| stable-baselines3 | Reference PPO implementation |
| optuna | Hyperparameter optimization |
| wandb | Experiment tracking |
| hydra-core | Configuration management |

See `setup_guide.md` for complete dependency list and installation instructions.

---

## Milestone Table

| Phase | Milestone | Deliverable | Success Metric | Deadline |
|-------|-----------|-------------|----------------|----------|
| 0 | M0: Environment Ready | Passing verify_setup.py, sanity check image | All checks pass, no OOM | Week 1 |
| 1 | M1: Working Agent | Agent completes full episodes, PPO loss decreases | Training converges on toy data | Week 4 |
| 2 | M2: Baselines Complete | All baselines evaluated on DrawBench | Metrics within 5% of literature values | Week 6 |
| 2 | M3: Config Selected | Optimized hyperparameters | Agent outperforms DDIM-20 | Week 7 |
| 3 | M4: Agent v1.0 Trained | Full-scale trained agent | H1: NFE $\leq$ 30, CLIP $\geq$ DDIM-50 | Week 8 |
| 3 | M5: Ablations Complete | 18 ablation variant results | All ablations interpretable | Week 9 |
| 3 | M6: Full Evaluation | Results on all 6 benchmarks | Statistical significance confirmed | Week 10 |
| 4 | M7: Generalization Tested | Results on SDXL, Flux.1, alternative schedulers | Transfer to $\geq 1$ new backbone | Week 11 |
| 4 | M8: Analysis Complete | All visualizations and failure analysis | Interpretable agent behavior | Week 12 |
| 5 | M9: Paper Draft | Complete paper with all sections | Internally consistent draft | Week 13 |
| 5 | M10: Submission Ready | Polished paper + supplementary + code | Ready for venue submission | Week 14 |

---

## Decision Gates

| Gate | Trigger | Go Condition | No-Go Action |
|------|---------|-------------|--------------|
| **G0** (Week 1) | GPU memory audit | SD 1.5 + metrics fit on A100-40GB ($\leq$ 35GB) | Implement serialized reward computation |
| **G1** (Week 4) | Toy training convergence | Loss decreases, non-degenerate actions | Debug rewards → simplify to stop/continue → offline RL |
| **G2** (Week 7) | Tuned agent vs. DDIM-20 | Agent outperforms DDIM-20 on $\geq 1$ metric | Investigate reward function, state features; consider offline RL |
| **G3** (Week 10) | H1 evaluation | NFE $\leq 30$ with CLIP $\geq$ DDIM-50 | If close: adjust claims. If far: pivot to H2 framing |
| **G4** (Week 11) | Generalization | Transfer to $\geq 1$ backbone without catastrophe | Report SD 1.5 only; note limitation |

---

## Contingency Plans

| Risk (Discovery ID) | Trigger | Fallback Strategy | Estimated Recovery Time |
|---------------------|---------|-------------------|------------------------|
| Agent doesn't learn (D-08) | G1 fails | Simplify to stop/continue only (A2 No-Refine variant) | 1 week |
| PPO sample inefficiency (D-29) | Training too slow | Switch to offline RL (IQL on pre-collected trajectories) | 2 weeks |
| PPO unstable | Loss diverges | Use REINFORCE with baseline; DPO on paired trajectories (D-33) | 1 week |
| Reward hacking (D-28) | VLM spot check reveals degenerate outputs | Increase $\alpha_3$ (stability); add VLM-scored reward term | 3 days |
| GPU OOM (D-25) | G0 fails | Serialize metric loading; reduce batch to 32; gradient checkpointing | 2 days |
| Region refinement artifacts (D-26) | Visual inspection | Increase Gaussian blur $\sigma$; adopt SAG as refinement | 2 days |
| UNet → DiT transfer fails (D-27) | G4 fails | Train DiT-specific agent head; report separately | 1 week |
| Metrics disagree | FID improves but ImageReward degrades | Use VLM-as-Judge as tiebreaker | 0 days (analysis only) |

---

*Document Version: 1.0*
*Created: February 2026*
*Sources: `research.md` (v1.0), `discovery.md` (v1.0), `experimental_design_adaptive_diffusion.md` (v2.0)*
