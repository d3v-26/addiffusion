# Research Discovery Log: AdDiffusion

> Comprehensive analytical pass on `experimental_design_adaptive_diffusion.md` (v2.0).
> Each entry includes a concise title, explanation, severity tag, and source reference.

---

## 1. Core Findings & Theoretical Insights

### D-01: MDP Formulation of Diffusion Inference Is the Central Contribution
**Priority:** Critical | **Reference:** experimental_design_adaptive_diffusion.md §1.2

Casting diffusion inference as a Markov Decision Process — where the agent observes the evolving latent state and selects from `{continue, stop, refine}` — is the primary theoretical contribution. This reframes a fixed computational pipeline as a sequential decision-making problem, enabling prompt-dependent resource allocation. The novelty lies not in any single component (PPO, CLIP scoring, region refinement) but in their integration under a unified MDP framework over the denoising trajectory.

### D-02: Dense Reward Shaping via Quality Deltas Enables Credit Assignment
**Priority:** High | **Reference:** experimental_design_adaptive_diffusion.md §1.4

The per-step quality reward ($\Delta\text{CLIP} + \Delta\text{Aesthetic} + \Delta\text{ImageReward}$) provides a dense learning signal, which is critical because the alternative — terminal-only reward — creates a severe credit assignment problem over 20–50 step episodes. The ablation A1 (Terminal-Only variant) directly tests this design choice. This is a well-founded decision; sparse reward with PPO in short-to-medium horizon tasks is known to cause slow convergence.

### D-03: Deterministic Mask Generation Collapses the Combinatorial Action Space
**Priority:** High | **Reference:** experimental_design_adaptive_diffusion.md §1.2, §1.6

The mask $m = \mathbb{1}[Q_\text{local}(x, c) < \tau]$ is computed deterministically from the current state, meaning the agent decides *whether* to refine but not *where*. This reduces the action space from a combinatorial mask-selection problem to three discrete actions ($|\mathcal{A}| = 3$), making PPO feasible. Without this design choice, the agent would need to learn both the decision and the spatial allocation simultaneously, dramatically increasing sample complexity.

### D-04: Frozen Encoder State Representation Avoids Co-Adaptation
**Priority:** Medium | **Reference:** experimental_design_adaptive_diffusion.md §1.2, §1.3

Using frozen CLIP and DINO encoders for state features $\phi(s)$ means the policy network receives stable, well-structured representations from the start of training. This avoids the co-adaptation problem where both the representation and policy shift simultaneously (as in end-to-end training), which would destabilize PPO. The trade-off is that the agent cannot learn task-specific visual features, but given the small policy network (two-layer MLP), stable inputs are more valuable than learned features.

### D-05: PPO Is Well-Matched to Expensive, Short-Horizon Environments
**Priority:** Medium | **Reference:** experimental_design_adaptive_diffusion.md §1.5; rl_ppo_explainer.md §4.4

Each environment step costs ~100ms (one diffusion forward pass), making sample efficiency critical. PPO's ability to perform K gradient epochs per data collection round (vs. REINFORCE's single-use trajectories) directly addresses this. With 50-step episodes and batch size 64, each PPO iteration collects ~3,200 transitions — PPO can extract more learning from these expensive samples than on-policy alternatives. The fallback to offline RL (IQL/CQL) or DPO is well-considered for the case where even PPO's sample efficiency is insufficient.

### D-06: Multi-Metric Reward Composition Mitigates Reward Hacking
**Priority:** High | **Reference:** experimental_design_adaptive_diffusion.md §1.4, §5.1

Including ImageReward (human preference proxy) alongside CLIP and Aesthetic scores provides redundancy against reward hacking. CLIP score alone is known to be exploitable — adversarial images can achieve high CLIP similarity without visual quality. ImageReward, trained on human preference data, penalizes such degenerate outputs. The DINO consistency term further constrains the agent from making erratic changes between steps. This multi-signal design is essential for training stability.

### D-07: Cross-Attention Maps Provide a Principled Refinement Signal
**Priority:** Medium | **Reference:** experimental_design_adaptive_diffusion.md §1.6

Using cross-attention maps $A \in \mathbb{R}^{H \times W \times L}$ to identify under-attended regions is grounded in the observation that diffusion models allocate denoising effort proportionally to attention magnitude. Regions with low maximum attention across all text tokens are likely under-generated. This is more principled than random or grid-based alternatives (tested in ablation A4), though it introduces an architecture dependency (UNet attention extraction differs from DiT).

### D-08: Prompt-Dependent Adaptive Behavior Is the Key Testable Hypothesis
**Priority:** Critical | **Reference:** experimental_design_adaptive_diffusion.md §4.1 (H3)

Hypothesis H3 — that the agent learns prompt-dependent behavior (simple prompts stop early, complex prompts use more steps/refinement) — is the most important experimental claim. If the trained agent does *not* exhibit this adaptive behavior and instead converges to a near-fixed schedule, the MDP formulation offers no advantage over learned step-count predictors like AdaDiff. The correlation between prompt complexity and agent NFE must be convincingly demonstrated, ideally with per-category breakdowns on PartiPrompts.

---

## 2. Literature Gaps & Open Questions

### D-09: Notation Collision — `γ` Used for Both Discount Factor and NFE Penalty
**Priority:** Critical | **Reference:** experimental_design_adaptive_diffusion.md §2.5 (Lines 367, 375)

The symbol $\gamma$ is used as the PPO discount factor ($\gamma = 0.99$, training hyperparameters) *and* as the NFE penalty weight ($\gamma = 0.01$, reward hyperparameters). These are unrelated quantities with a 100x difference in value. This will cause confusion in implementation and paper writing. **Fix:** Rename the NFE penalty to $c_\text{nfe}$ or $\eta$ and reserve $\gamma$ exclusively for the discount factor.

### D-10: NFE Formula for Refine Action Is Incorrect
**Priority:** Critical | **Reference:** experimental_design_adaptive_diffusion.md §1.2 (Line 114), §1.6 (Lines 154–161)

The document states $\text{NFE}(a_\text{refine}(m)) = 1 + k \cdot |m|/HW$, implying compute scales with mask area. However, the `RegionRefine` algorithm runs full forward passes through the denoising network (`Denoise(x_t^masked, c, t')`) regardless of mask size — UNet/DiT architectures process the entire spatial tensor. The correct formula is $\text{NFE}(a_\text{refine}) = 1 + k$. The $|m|/HW$ factor would only be valid with patch-based inference (e.g., processing only masked patches through the network), which is not described. **Fix:** Either correct the formula to $1+k$ or describe and implement a patch-based denoising strategy.

### D-11: `ΔConsistency` Is Not a Delta — Notation Inconsistency
**Priority:** Medium | **Reference:** experimental_design_adaptive_diffusion.md §1.4 (Line 103)

The quality reward terms $\Delta\text{CLIP}$, $\Delta\text{Aesthetic}$, and $\Delta\text{ImageReward}$ are all defined as differences (step $i+1$ minus step $i$). However, $\Delta\text{Consistency} = \text{DINO}_\text{sim}(\hat{x}_0^{(i+1)}, \hat{x}_0^{(i)})$ is a *similarity score*, not a difference. This inconsistency is misleading — a reader would expect all $\Delta$-prefixed terms to be differences. **Fix:** Rename to $R_\text{stability}$ or $\text{Sim}_\text{DINO}$.

### D-12: Appendix Summary Equation Drops the Consistency Term
**Priority:** Medium | **Reference:** experimental_design_adaptive_diffusion.md Appendix (Line 632) vs. §1.4 (Line 98)

The summary equation in the Appendix reads: $R = \alpha_1 \Delta\text{CLIP} + \alpha_2 \Delta\text{Aesthetic} + \alpha_4 \Delta\text{ImageReward} - \gamma \cdot \text{NFE} + R_\text{terminal}$. This omits $\alpha_3 \cdot \Delta\text{Consistency}$ (the DINO stability term), which appears in the full reward definition at Line 98. **Fix:** Add the missing term to the summary.

### D-13: Composite Reward Weights `λ₁, λ₂` Are Undefined
**Priority:** High | **Reference:** experimental_design_adaptive_diffusion.md §1.4 (Line 95), §2.5

The composite reward is defined as $R = R_\text{quality} + \lambda_1 R_\text{efficiency} + \lambda_2 R_\text{terminal}$, but $\lambda_1$ and $\lambda_2$ do not appear in any hyperparameter table. Meanwhile, $R_\text{efficiency}$ already contains an internal weight $\gamma$ (the NFE penalty). If $\lambda_1 = 1$, the notation is redundant. **Fix:** Either specify $\lambda_1, \lambda_2$ values in the hyperparameter table, or remove them from the composite formula and let the internal weights ($\gamma$, $\beta_i$) serve as the effective scaling factors.

### D-14: Formulation Uses Pixel Space but Implementation Operates in Latent Space
**Priority:** High | **Reference:** experimental_design_adaptive_diffusion.md §1.1, §1.6

The mathematical formulation defines $x_t \in \mathbb{R}^{H \times W \times C}$ and masks $m \in \{0,1\}^{H \times W}$, which is pixel-space notation. However, Stable Diffusion (and all listed backbones) operate in latent space: $z_t \in \mathbb{R}^{h \times w \times 4}$ where $h = H/8, w = W/8$. This distinction matters concretely: (1) the refinement mask must operate at latent resolution ($64 \times 64$ for SD 1.5, not $512 \times 512$), (2) cross-attention maps have latent-space spatial dimensions, and (3) quality metrics like CLIP require decoding $z_t$ to pixel space via the VAE. **Fix:** Rewrite the formulation in latent space, or add a notation paragraph clarifying that $x_t$ denotes the latent representation and that pixel-space operations require VAE decode.

### D-15: Classifier-Free Guidance Is Not Modeled
**Priority:** High | **Reference:** experimental_design_adaptive_diffusion.md §1.1, §1.2

The formulation treats denoising as $p_\theta(x_{t-1} | x_t)$, but in practice all listed models use classifier-free guidance (CFG): $\hat{\epsilon} = \epsilon_\theta(x_t, \varnothing) + w \cdot (\epsilon_\theta(x_t, c) - \epsilon_\theta(x_t, \varnothing))$, where $w$ is the guidance scale (7.5 in default config). CFG doubles the NFE per step (two forward passes: conditional + unconditional). This means: (1) the NFE accounting should reflect $\text{NFE}(a_\text{continue}) = 2$ (not 1), or the document should clarify that "NFE" counts denoising steps, not network evaluations; (2) the guidance scale $w$ could itself be an adaptive parameter (see D-30).

### D-16: One-Step Clean Prediction Is Unreliable at High Noise Levels
**Priority:** Medium | **Reference:** experimental_design_adaptive_diffusion.md §1.2 (Line 44)

The state features include $\text{CLIP}_\text{img}(\hat{x}_0^{(i)})$, where $\hat{x}_0^{(i)}$ is the one-step clean prediction from $x_{t_i}$. At early timesteps (high noise, $t_i$ close to $T$), this prediction is extremely blurry and semantically meaningless — CLIP features of such images carry little information. This means the agent's early decisions (steps 1–5) are effectively blind, operating mainly on the timestep embedding and text features. This is not necessarily fatal (the agent can learn to always `continue` at early steps), but it limits the agent's ability to make informed early-stopping decisions for very simple prompts. **Consider:** adding a "warmup" period where the agent always continues for the first $k$ steps, reducing the decision space.

### D-17: `FID_target` in Quality-Adjusted NFE Is Undefined
**Priority:** Medium | **Reference:** experimental_design_adaptive_diffusion.md §2.4 (Line 303)

The composite metric $\text{Quality-Adjusted NFE} = \text{NFE} \times (1 + \max(0, \text{FID} - \text{FID}_\text{target}))$ references $\text{FID}_\text{target}$ without specifying its value. This makes the metric unreproducible. **Fix:** Define $\text{FID}_\text{target}$ (e.g., DDIM-50's FID on the evaluation set, or a fixed threshold like 10.0).

### D-18: Episode Structure and Scheduler Interaction Are Underspecified
**Priority:** High | **Reference:** experimental_design_adaptive_diffusion.md §1.2, §2.5

The agent makes decisions at each denoising step, but the relationship between the agent's episode and the diffusion scheduler is unclear. If the scheduler runs 50 steps (DDIM-50), does the agent make 50 decisions? If the agent stops at step 20, does it return the one-step prediction $\hat{x}_0^{(20)}$ or run the scheduler's remaining 30 steps as a "free" computation? Additionally, $N_\text{max} = 50$ in the hyperparameters matches the DDIM-50 baseline, but the document also proposes DDIM-20 and DPM-Solver-20 baselines — the agent's behavior when wrapping a 20-step scheduler is not described. **Fix:** Explicitly define: (1) how the agent interfaces with different schedulers, (2) what "stop" means mechanically, and (3) whether $N_\text{max}$ is fixed or adapter to the scheduler.

### D-19: Refine Action's Interaction with the Denoising Schedule Is Ambiguous
**Priority:** Medium | **Reference:** experimental_design_adaptive_diffusion.md §1.6, §1.2

After the agent takes $a_\text{refine}$, the `RegionRefine` algorithm re-noises the masked region to $t' < t$ and denoises $k$ times. But what happens next? Does the agent resume the normal schedule at $t_{i+1}$ with the refined $\hat{x}_0$ (requiring re-noising to $t_{i+1}$ noise level)? Or does the refined output replace $x_{t_i}$ and the next step proceeds from there? The algorithm returns $\hat{x}_0$ (a clean image), but the next denoising step expects a noisy input at $t_{i+1}$. **Fix:** Specify the state transition after refinement — likely: $x_{t_{i+1}} = \sqrt{\bar{\alpha}_{t_{i+1}}} \hat{x}_0^\text{refined} + \sqrt{1 - \bar{\alpha}_{t_{i+1}}} \epsilon$ (re-noising the refined prediction to the next schedule point).

### D-20: No Mechanism for Preserving Batch-Level Diversity
**Priority:** Medium | **Reference:** experimental_design_adaptive_diffusion.md §1.4, §2.4

The reward function optimizes per-image quality (CLIP score, ImageReward, Aesthetic). However, FID — a primary evaluation metric — is a distributional metric that penalizes low diversity. An agent that produces individually high-quality but semantically similar images (mode collapse) would score well on per-image metrics but poorly on FID. There is no diversity-preserving term in the reward. **Mitigation:** This may not be a practical issue if the agent only modifies the denoising *schedule* (not the noise initialization), preserving the diversity of the initial noise samples. Monitor FID during training to detect mode collapse.

### D-21: PPO Subscript `t` Collides with Diffusion Timestep `t`
**Priority:** Low | **Reference:** experimental_design_adaptive_diffusion.md §1.5 (Line 132)

The PPO ratio $r_t(\psi)$ uses subscript $t$ for the trajectory step index, but $t$ is already used for diffusion timesteps throughout the document. In a paper where both appear in the same equation, this would cause confusion. **Fix:** Use $i$ or $n$ for trajectory step index.

### D-22: AdaDiff Baseline Lacks Citation
**Priority:** Low | **Reference:** experimental_design_adaptive_diffusion.md §2.2 (Line 203)

AdaDiff is listed as an adaptive baseline ("Learned step selection per prompt") but no citation is provided. If this is a published method, it needs a reference. If it is a method to be implemented for this project, it should be described in sufficient detail to reproduce.

---

## 3. Technical Dependencies & Risk Factors

### D-23: CLIP Inference Overhead Is Not Reflected in Efficiency Accounting
**Priority:** Medium | **Reference:** experimental_design_adaptive_diffusion.md §1.2 (Line 41), §2.4

Computing state features requires a CLIP forward pass on $\hat{x}_0^{(i)}$ at every step (~5–10ms per image on A100). Over 28 average steps, this adds 140–280ms per image, which is ~5–9% of the 3.2s expected generation time. The NFE metric counts only diffusion model evaluations, so the overhead is invisible. Additionally, DINO features (for consistency) and ImageReward (for reward computation) add further overhead. **Impact:** The wall-clock time comparison with baselines may be less favorable than NFE comparisons suggest. **Fix:** Report both NFE and wall-clock time, and note the overhead of agent-related computation.

### D-24: Statistical Power Is Insufficient for DrawBench Comparisons
**Priority:** Medium | **Reference:** experimental_design_adaptive_diffusion.md §2.4 (Line 342)

The document states minimum detectable effect on DrawBench (200 prompts) is $\approx 0.05$ CLIP score difference at $\alpha=0.05$, power=0.8. However, the expected results table shows most inter-method differences are 0.01–0.03 (e.g., DDIM-20 at 0.28 vs DPM-Solver-20 at 0.29). These differences are below the detection threshold and would not reach statistical significance on DrawBench alone. **Mitigation:** Use DrawBench primarily for VLM-as-Judge pairwise comparisons (which test holistic preference, not metric deltas) and COCO-30K for metric-level significance testing.

### D-25: GPU Memory Pressure for Full Pipeline on Single A100
**Priority:** High | **Reference:** experimental_design_adaptive_diffusion.md §2.5, §6.1; setup_guide.md

During training, the following models must coexist in GPU memory: (1) frozen diffusion model (SD 1.5: ~3.4GB fp16; SDXL: ~6.5GB fp16), (2) frozen CLIP encoder (~1.5GB), (3) frozen DINO encoder (~0.3GB), (4) ImageReward model (~1.5GB), (5) policy + value networks (~50MB), (6) intermediate activations and latents for batch of 64. For SD 1.5 on A100-40GB, this is tight but feasible with mixed precision. For SDXL on A100-40GB, this is likely infeasible — the 80GB variant is needed (correctly specified in §6.1 for SDXL training). **Risk:** If not all metric models can be loaded simultaneously, reward computation must serialize model loading/unloading, significantly slowing training. **Mitigation:** Compute rewards in a separate pass after trajectory collection; use gradient checkpointing; precompute CLIP text embeddings.

### D-26: Hard Binary Masks Will Produce Boundary Artifacts
**Priority:** Medium | **Reference:** experimental_design_adaptive_diffusion.md §1.6 (Line 158)

The `RegionRefine` algorithm uses hard binary masks: $x_t^\text{masked} = m \odot \text{AddNoise}(\hat{x}_0, t') + (1-m) \odot \hat{x}_0$. At the boundary between noised (masked=1) and clean (masked=0) regions, there is a discontinuous jump in noise level. This creates visible seam artifacts in the output, a well-documented problem in image inpainting. **Fix:** Apply Gaussian blur to the mask before compositing (e.g., $m_\text{soft} = \text{GaussianBlur}(m, \sigma=3)$), or use a feathered mask with linear interpolation in a boundary band.

### D-27: UNet → DiT Cross-Attention Extraction Requires Architecture-Specific Code
**Priority:** High | **Reference:** experimental_design_adaptive_diffusion.md §4.3, §1.6

The region refinement mask depends on cross-attention maps extracted from the diffusion model's UNet. Flux.1 uses a Diffusion Transformer (DiT) architecture where attention is structured differently (joint attention over text and image tokens, not separate cross-attention layers). Extracting equivalent spatial attention maps from DiT requires fundamentally different code. The document acknowledges this (§4.3 note on UNet → DiT transfer) but underestimates the engineering effort. **Fix:** Design the attention extraction as a pluggable interface from the start, with separate implementations for UNet-based (SD 1.5, SDXL) and DiT-based (Flux.1) models.

### D-28: CLIP Reward Hacking Remains a Residual Risk Despite Multi-Metric Design
**Priority:** Medium | **Reference:** experimental_design_adaptive_diffusion.md §5.1 (Line 581)

While the multi-metric reward (D-06) mitigates CLIP hacking, ImageReward itself was trained on CLIP features and may share some failure modes. If the agent discovers denoising trajectories that jointly optimize CLIP and ImageReward at the expense of visual quality (e.g., high-frequency textures that score well but look artificial), the consistency term ($\alpha_3 = 0.2$) may be too weak to prevent this. **Mitigation:** Monitor visual outputs during training with periodic VLM-as-Judge spot checks; consider increasing $\alpha_3$ if degenerate outputs appear; implement an "output diversity check" that flags batches with unusually high metric scores.

### D-29: PPO Sample Inefficiency with ~100ms Per Environment Step
**Priority:** High | **Reference:** experimental_design_adaptive_diffusion.md §5.1 (Line 582); rl_ppo_explainer.md §4.1

Each PPO iteration collects 64 trajectories × ~30 steps × ~100ms/step ≈ 192 seconds of environment interaction. With 2000 training iterations, that is ~107 hours of pure environment time, not counting PPO gradient updates and metric computation. This is within the 200 GPU-hour SD 1.5 budget but leaves little margin for hyperparameter search. **Mitigation:** (1) Reduce trajectory collection batch size for initial experiments; (2) use the one-step prediction $\hat{x}_0^{(i)}$ (already computed by the scheduler) to avoid extra forward passes; (3) consider offline RL on pre-collected trajectories as a fallback (per §5.2).

### D-30: Stochastic Samplers Introduce Non-Determinism in State Transitions
**Priority:** Low | **Reference:** experimental_design_adaptive_diffusion.md §2.2, §1.2

The baseline list includes DDIM (deterministic), DPM-Solver (deterministic), and Euler (can be stochastic). When wrapping a stochastic sampler, the same state-action pair can lead to different next states, violating the deterministic MDP assumption. This increases variance in advantage estimation. The document does not specify whether stochastic samplers should be used with fixed noise seeds during training. **Fix:** Use deterministic samplers (DDIM, DPM-Solver) during agent training; evaluate with stochastic samplers only at test time to measure robustness.

---

## 4. Opportunities & Extensions

### D-31: Dynamic Guidance Scale as an Additional Action Dimension
**Priority:** Medium | **Reference:** experimental_design_adaptive_diffusion.md §1.2

Classifier-free guidance scale $w$ significantly affects the quality-diversity tradeoff and is typically fixed at 7.5. Making guidance scale an adaptive parameter — either as a continuous action dimension or a discrete set (e.g., $w \in \{3.0, 5.0, 7.5, 10.0, 15.0\}$) — would allow the agent to reduce guidance early (for diversity/speed) and increase it later (for detail). This is a natural extension that would strengthen the "adaptive computation" narrative. Not essential for the initial submission but a strong follow-up.

### D-32: Curriculum Learning on Prompt Complexity Would Accelerate Training
**Priority:** Medium | **Reference:** experimental_design_adaptive_diffusion.md §5.1 (Line 583)

The document mentions curriculum learning as a risk mitigation but does not define a concrete curriculum. A natural structure: Phase 1 — train on single-object prompts from COCO ("a cat", "a red car"); Phase 2 — multi-object prompts ("a cat on a chair"); Phase 3 — complex compositional prompts from PartiPrompts. This would help the agent learn basic stop/continue behavior before tackling refinement decisions on complex scenes.

### D-33: Offline RL / DPO as a Computationally Cheaper Alternative
**Priority:** High | **Reference:** experimental_design_adaptive_diffusion.md §5.2 (Line 589)

The document mentions DPO-style optimization on paired trajectories as a fallback. This deserves more attention — it could be the primary training method. By generating images at multiple fixed step counts (e.g., 10, 20, 30, 40, 50 steps) and scoring them with ImageReward, one can construct preference pairs without any RL infrastructure. A lightweight classifier trained on these pairs could approximate the full agent's stopping behavior at a fraction of the training cost. This could serve as a strong ablation baseline (Agent-DPO vs. Agent-PPO).

### D-34: Application to Video Diffusion Models
**Priority:** Low | **Reference:** experimental_design_adaptive_diffusion.md (not discussed)

Video diffusion models (e.g., Stable Video Diffusion, Sora-class models) have an even stronger need for adaptive computation — temporal consistency varies across frames, and some frames need more denoising than others. The MDP formulation extends naturally to video by adding frame-index to the state. This is a compelling future direction but out of scope for the initial paper.

### D-35: Integration with ControlNet / IP-Adapter for Conditional Generation
**Priority:** Low | **Reference:** experimental_design_adaptive_diffusion.md (not discussed)

When using additional conditioning (depth maps, pose, reference images), the quality landscape changes — some conditioned regions may converge faster. The agent could learn to allocate less compute to well-conditioned regions. This extension would broaden the practical applicability of the system.

### D-36: Hierarchical Agent — Meta-Policy Selects Strategy, Sub-Policy Executes
**Priority:** Medium | **Reference:** experimental_design_adaptive_diffusion.md §1.3

The current flat policy maps state → action at every step. An alternative is a hierarchical design: a meta-policy first classifies the prompt difficulty (easy/medium/hard) and selects a denoising strategy (e.g., fast 15-step, standard 30-step, quality 50-step + refine), then a sub-policy handles fine-grained decisions within that strategy. This would reduce the per-step decision burden and could improve training stability. Worth exploring if the flat policy struggles to learn diverse behaviors.

### D-37: Ablation Target — Reward Component Normalization
**Priority:** Medium | **Reference:** experimental_design_adaptive_diffusion.md §1.4; rl_ppo_explainer.md §6.5

The reward components have different natural scales: $\Delta\text{CLIP} \in [-0.05, 0.05]$, $\Delta\text{Aesthetic} \in [-0.3, 0.3]$, $\Delta\text{ImageReward} \in [-0.2, 0.2]$, $\text{NFE} = 1$. The $\alpha$ weights are intended to balance these, but the current values ($\alpha_1=1.0$, $\alpha_2=0.5$, $\alpha_4=0.8$) do not clearly normalize to comparable scales. The PPO explainer (§6.5) correctly identifies this and suggests normalizing each component to $[-1, 1]$ before weighting. This normalization should be an explicit step in the reward computation, not left implicit. An ablation comparing raw vs. normalized reward components would be valuable.

### D-38: |A| Should Be Explicitly Stated in the Policy Architecture
**Priority:** Low | **Reference:** experimental_design_adaptive_diffusion.md §1.3 (Line 84)

The policy outputs $\pi(a|s) \in \Delta^{|\mathcal{A}|}$ but $|\mathcal{A}|$ is never explicitly stated. Given D-03 (deterministic masks), $|\mathcal{A}| = 3$. This should be stated in the architecture description for clarity, especially since a reader unfamiliar with the mask design might assume a much larger action space.

---

## Summary of Findings by Severity

### Critical (Must fix before implementation)
| ID | Title | Section |
|----|-------|---------|
| D-09 | `γ` notation collision | §2.5 |
| D-10 | NFE(refine) formula incorrect | §1.2, §1.6 |
| D-01 | MDP formulation is the core contribution (ensure paper framing) | §1.2 |
| D-08 | Adaptive behavior is the key hypothesis (must be convincingly demonstrated) | §4.1 |

### High (Should fix before implementation)
| ID | Title | Section |
|----|-------|---------|
| D-13 | `λ₁, λ₂` undefined | §1.4, §2.5 |
| D-14 | Latent vs. pixel space ambiguity | §1.1, §1.6 |
| D-15 | Classifier-free guidance not modeled | §1.1 |
| D-18 | Episode structure underspecified | §1.2, §2.5 |
| D-25 | GPU memory pressure | §2.5, §6.1 |
| D-27 | UNet → DiT attention extraction | §4.3, §1.6 |
| D-29 | PPO sample inefficiency | §5.1 |
| D-33 | Offline RL/DPO as primary alternative | §5.2 |

### Medium (Address during development)
| ID | Title | Section |
|----|-------|---------|
| D-11 | `ΔConsistency` not a delta | §1.4 |
| D-12 | Appendix drops `α₃` | Appendix |
| D-16 | One-step prediction unreliable early | §1.2 |
| D-17 | `FID_target` undefined | §2.4 |
| D-19 | Refine → schedule interaction | §1.6, §1.2 |
| D-20 | No batch diversity mechanism | §1.4 |
| D-23 | CLIP inference overhead | §1.2, §2.4 |
| D-24 | DrawBench statistical power | §2.4 |
| D-26 | Mask boundary artifacts | §1.6 |
| D-28 | Residual reward hacking risk | §5.1 |
| D-31 | Dynamic guidance scale | §1.2 |
| D-32 | Curriculum learning | §5.1 |
| D-36 | Hierarchical agent architecture | §1.3 |
| D-37 | Reward normalization ablation | §1.4 |

### Low (Nice to have / future work)
| ID | Title | Section |
|----|-------|---------|
| D-21 | PPO subscript `t` collision | §1.5 |
| D-22 | AdaDiff missing citation | §2.2 |
| D-30 | Stochastic sampler interaction | §2.2 |
| D-34 | Video diffusion extension | — |
| D-35 | ControlNet integration | — |
| D-38 | `|A|` not stated | §1.3 |

---

*Document Version: 1.0*
*Created: February 2026*
*Source: `experimental_design_adaptive_diffusion.md` (v2.0), `rl_ppo_explainer.md`, `setup_guide.md`*
