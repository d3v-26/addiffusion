# AdDiffusion — Project Instructions

## What This Project Is

AdDiffusion is a research project that reformulates diffusion model inference as a Markov Decision Process (MDP). A lightweight PPO-trained policy network observes the evolving latent state during denoising and selects from three actions at each step: **continue** (standard denoising), **stop** (early termination), or **refine** (selective region re-denoising via cross-attention masks). The goal is prompt-dependent adaptive computation — simple prompts terminate early, complex prompts allocate additional effort to under-generated regions.

Primary backbone: Stable Diffusion 1.5. Generalization targets: SDXL, Flux.1-schnell.

## Key Documents

| Document | Purpose |
|----------|---------|
| `research.md` | Unified publication-structured document (methodology, experiments, expected results) |
| `discovery.md` | 38 analytical findings (D-01 through D-38) with severity tags and fixes |
| `plan.md` | 14-week execution plan with 5 decision gates (G0–G4) and contingency plans |
| `experiment.md` | Step-by-step experiment execution guide with commands and validation criteria |
| `setup_guide.md` | Environment setup (uv-based, HPC/SLURM) |

**Always consult `research.md` for the authoritative formulation.** It incorporates all fixes from `discovery.md`. The original `experimental_design_adaptive_diffusion.md` is superseded.

## Critical Design Decisions (from discovery.md)

These issues have been resolved in `research.md` but are easy to reintroduce in code. Enforce them strictly:

1. **D-09 — Notation collision resolved.** The discount factor is `gamma_d` (0.99). The NFE penalty weight is `c_nfe` (0.01). Never use bare `gamma` for both.
2. **D-10 — NFE(refine) = 1 + k**, not `1 + k * |m| / (h*w)`. The UNet/DiT processes the full spatial tensor regardless of mask size. Only use fractional scaling if patch-based inference is explicitly implemented.
3. **D-13 — No outer lambda weights.** The composite reward is `R = R_quality + R_efficiency + R_terminal` directly. All weighting is through internal parameters (alpha_k, c_nfe, beta_k). Do not introduce `lambda_1` or `lambda_2`.
4. **D-14 — Latent space, not pixel space.** All operations use latent representations `z_t` at resolution `h x w` (64x64 for SD 1.5). Pixel-space images `x` are obtained via VAE decode. Masks operate at latent resolution.
5. **D-15 — CFG doubles network evaluations.** Our "NFE" counts scheduler steps, not network forward passes. One continue action = 1 NFE (but 2 network evals with CFG). Document this distinction in any efficiency reporting.
6. **D-11 — Stability term is not a delta.** `R_stability = DINO_sim(x0_hat_i+1, x0_hat_i)` is a similarity score in [0,1], not a difference. Do not prefix it with "Delta".
7. **D-26 — Soft masks required.** Always apply Gaussian blur (sigma=3) to the binary attention mask before compositing in `RegionRefine`. Hard masks create boundary artifacts.
8. **D-37 — Normalize reward components.** Scale each component to approximately [-1, 1] before applying alpha weights: divide delta_CLIP by 0.05, delta_Aesthetic by 0.3, delta_ImageReward by 0.2. R_stability is already in [0,1].

## Architecture Summary

```
State: phi(s_i) = [CLIP_img(x0_hat), CLIP_txt(c), emb(t_i), q_i]  ~  d=1280
Policy: Linear(d,512) -> ReLU -> Linear(512,256) -> ReLU -> Linear(256,3) -> Softmax
Value:  Linear(d,512) -> ReLU -> Linear(512,256) -> ReLU -> Linear(256,1)
Actions: |A| = 3  {continue, stop, refine}
Mask: m = GaussianBlur(1[max_l A[:,:,l] < tau], sigma=3)  (deterministic from state)
```

The agent decides **whether** to refine, not **where**. The mask is computed deterministically from cross-attention maps. This collapses the combinatorial action space to 3 discrete actions.

## Episode Structure

- Agent wraps a base scheduler (e.g., DDIM-50). N_max matches the scheduler's step count.
- First 3 steps are mandatory `continue` (warmup — one-step predictions are unreliable at high noise).
- After warmup, agent chooses continue/stop/refine at each step.
- `stop` decodes the current clean prediction and ends the episode.
- `refine` runs RegionRefine (k=2 iterations), then re-noises to next schedule point and continues.
- If step N_max is reached without stopping, the final output is decoded.

## Naming Conventions

- **Experiment IDs:** `EXP-{B/T/A/E/G}{number}` (B=baseline, T=training, A=ablation, E=evaluation, G=generalization)
- **Discovery IDs:** `D-{01..38}` — reference `discovery.md` for details
- **Ablations:** A1 (reward components), A2 (action space), A3 (state representation), A4 (region selection), A5 (reward normalization)
- **Hypotheses:** H1 (efficiency: NFE<=30 matching DDIM-50 quality), H2 (quality at equal budget), H3 (adaptive prompt-dependent behavior), H4 (refinement > extra steps)
- **Decision gates:** G0 (GPU memory), G1 (toy convergence), G2 (beats DDIM-20), G3 (H1 evaluation), G4 (generalization)

## Project Structure

```
addiffusion/
  src/
    agent/         # Policy/value networks, state features, PPO training, actions
    diffusion/     # Pipeline wrapper, attention extraction, region refinement
    rewards/       # Reward computation with normalization
    evaluation/    # Metrics (FID, CLIP, ImageReward, etc.), VLM-as-Judge, significance
    baselines/     # Fixed-step, SAG, Attend-and-Excite, LCM, oracle
    utils/
  configs/         # Hydra YAML configs (default.yaml, sdxl.yaml, flux.yaml)
  scripts/         # SLURM templates, verify_setup.py
  checkpoints/     # Agent weights (every 50 iterations + final)
  data/            # Prompt datasets (COCO, DrawBench, PartiPrompts, GenEval, T2I-CompBench)
  outputs/         # Generated images and metrics (see experiment.md §4 for full layout)
  logs/
```

## Hyperparameters (from research.md §3.6)

| Parameter | Symbol | Value |
|-----------|--------|-------|
| Learning rate | — | 3e-4 (Adam) |
| Batch size | — | 64 trajectories |
| PPO clip | epsilon | 0.2 |
| GAE lambda | lambda_GAE | 0.95 |
| Discount | gamma_d | 0.99 |
| Entropy coeff | c_2 | 0.01 |
| Value coeff | c_1 | 0.5 |
| Max grad norm | — | 0.5 |
| CLIP delta weight | alpha_1 | 1.0 |
| Aesthetic delta weight | alpha_2 | 0.5 |
| Stability weight | alpha_3 | 0.2 |
| ImageReward delta weight | alpha_4 | 0.8 |
| NFE penalty | c_nfe | 0.01 |
| Terminal CLIP | beta_1 | 2.0 |
| Terminal Aesthetic | beta_2 | 1.0 |
| Terminal ImageReward | beta_3 | 1.5 |
| Refinement iterations | k | 2 |
| Mask threshold | tau | 0.5 |
| Noise reduction ratio | r_noise | 0.5 |

## Environment

- Python 3.10, PyTorch 2.6.0, CUDA 12.4
- Package manager: `uv` (not pip)
- Config management: Hydra
- Experiment tracking: W&B
- Compute: A100-40GB (SD 1.5), A100-80GB (SDXL)
- Run commands with `uv run python ...`

## Code Conventions

- Use latent-space variable names (`z_t`, `z_0`) in diffusion code, pixel-space (`x`, `x_0`) only after VAE decode.
- Use `c_nfe` not `gamma` for the NFE penalty weight. Use `gamma_d` for discount factor.
- Attention extraction must be a pluggable interface: separate implementations for UNet (SD 1.5, SDXL) and DiT (Flux.1). See D-27.
- Train with deterministic samplers (DDIM, DPM-Solver). Only use stochastic samplers at eval time. See D-30.
- All reward components must be normalized before weighting. See D-37.
- Report both NFE (scheduler steps) and wall-clock time (including agent overhead). See D-23.

## Common Pitfalls to Avoid

1. Using `gamma` ambiguously for both discount and NFE penalty (D-09)
2. Scaling NFE(refine) by mask area — the full tensor is always processed (D-10)
3. Computing masks at pixel resolution instead of latent resolution (D-14)
4. Forgetting that RegionRefine output must be re-noised to the next schedule point (D-19)
5. Using hard binary masks without Gaussian blur (D-26)
6. Treating DINO consistency as a delta when it's an absolute similarity (D-11)
7. Not normalizing reward components before weighting (D-37)
8. Assuming CLIP features are informative at early timesteps — they're not (D-16)

## Execution Plan Phases

| Phase | Timeline | Key Deliverable |
|-------|----------|----------------|
| 0: Setup | Week 1 | Environment ready, GPU memory audit (G0) |
| 1: Core Agent | Weeks 2-4 | Working agent with PPO training (G1) |
| 2: Baselines & Tuning | Weeks 5-7 | All baselines, hyperparameter selection (G2) |
| 3: Full Training & Eval | Weeks 8-10 | Trained agent, ablations, full benchmarks (G3) |
| 4: Generalization | Weeks 11-12 | SDXL/Flux.1 transfer, visualizations (G4) |
| 5: Paper | Weeks 13-14 | Submission-ready paper |

## Fallback Strategies

- If full agent fails: simplify to stop/continue only (no refine) — ablation A2 No-Refine
- If PPO too sample-inefficient: switch to offline RL (IQL) or DPO on paired trajectories (D-33)
- If region refinement creates artifacts: adopt SAG as the refinement mechanism
- If UNet-to-DiT transfer fails: train architecture-specific heads, report separately
- If quality metrics disagree: use VLM-as-Judge (GPT-4o) as tiebreaker
