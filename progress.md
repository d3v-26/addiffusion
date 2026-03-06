# AdDiffusion — Progress Summary

> Plain-English overview of everything built so far, with a reference to every file.
> For full technical detail see `phase1.md` (core agent) and `phase2.md` (baselines + evaluation + tuning).

---

## The Big Picture

The goal of this project is to make image generation smarter about when to stop. A standard diffusion model runs a fixed number of denoising steps — say, 50 — for every prompt, regardless of how simple or complex the image is. We're training a small decision-making agent (using reinforcement learning) that watches the image form step by step and decides: *keep going*, *stop early*, or *re-do a specific region that looks wrong*. The hope is that simple prompts finish in 20 steps instead of 50, while complex ones get extra attention where they need it.

The project is built on top of **Stable Diffusion 1.5** and the agent is trained with **PPO** (Proximal Policy Optimization), a standard RL algorithm.

We are now two phases into a five-phase plan. Phase 1 built the agent from scratch. Phase 2 built all the tools needed to measure whether the agent is actually any good.

---

## Phase 1: Build the Agent (Weeks 2–4) ✓

**What we did:** Wrote every piece of code needed to run one training iteration — from loading the diffusion model to updating the policy network.

### The Diffusion Pipeline

**`src/diffusion/pipeline.py`**
The heart of the project. Wraps Stable Diffusion so we can run it one step at a time instead of all at once. Normally SD runs as a black box; this file exposes each denoising step so the agent can look at the intermediate image and decide what to do next. Also handles encoding text prompts, sampling initial noise, and decoding the final latent into a pixel image.

**`src/diffusion/attention.py`**
At each denoising step, the model's internal cross-attention maps show which parts of the image correspond to which words in the prompt. This file hooks into the model and extracts those maps. The maps are later used to decide *where* to re-do regions. Written as a pluggable interface so it works for both SD 1.5/SDXL (UNet-based) and Flux.1 (DiT-based).

**`src/diffusion/refine.py`**
Implements the "refine" action. When the agent decides a region needs more work, this file: (1) finds which pixels are underperforming by looking at the attention maps, (2) applies a soft Gaussian blur to the mask edges so there are no harsh boundaries, (3) re-adds a controlled amount of noise to just that region, and (4) runs a few extra denoising steps on it. After refining, the latent is re-noised to the correct noise level so the main denoising loop can continue normally.

### The Agent

**`src/agent/state.py`**
At each step, the agent needs a description of "what is the current state of the image?" This file computes that description as a 1672-dimensional vector by combining: a CLIP embedding of the current one-step image prediction (768d), a CLIP embedding of the text prompt (768d, cached once), a sinusoidal embedding of the current timestep (128d), and a small quality vector (8d). All encoders are frozen — only the policy head trains.

**`src/agent/networks.py`**
Two small neural networks that sit on top of the state vector:
- **Policy network:** takes the 1672d state, runs it through two hidden layers, and outputs probabilities for three actions — continue, stop, refine.
- **Value network:** same architecture but outputs a single number estimating how much total reward we expect from here. Used during PPO training to reduce variance.

**`src/agent/episode.py`**
Runs one complete image generation as an agent episode. The first 3 steps are always "continue" (the image is too noisy early on for the agent's judgment to be meaningful). After that, the agent picks an action at each step. The file records every (state, action, reward) tuple along the way, which is the data PPO trains on.

**`src/agent/ppo.py`**
The training algorithm. Takes a batch of completed episodes, computes advantages using GAE (a technique to estimate how much better an action was than average), then updates the policy and value networks to make good actions more likely. Clips the update size so training stays stable.

### Reward

**`src/rewards/reward.py`**
Defines what "good" means. After each agent action, a reward is computed from three components:
- **Quality reward:** did the image get better this step? Measured as the change in CLIP score (image-text alignment) and ImageReward score, plus a DINO similarity score for consistency.
- **Efficiency reward:** a small penalty proportional to how many denoising steps were used (`-0.01 × NFE`).
- **Terminal reward:** when the episode ends, a final score based on the absolute CLIP and ImageReward values of the output image.

All components are normalized to comparable scales before being combined — no single metric dominates.

### Training Entry Point

**`src/train.py`**
The main script that ties everything together. Loads the config, initializes the diffusion pipeline and agent components, then runs a loop: generate episodes, compute rewards, update networks, save checkpoints. Supports a curriculum that starts training on simple prompts and gradually introduces harder ones.

**`configs/default.yaml`**
All hyperparameters in one place: learning rate (3e-4), PPO clip (0.2), discount factor (0.99), reward weights, refinement settings, etc.

**`scripts/train.slurm`** / **`scripts/g1_test.slurm`**
Job submission templates for running training on the HPC cluster (all GPU work happens there, not locally).

### Phase 1 Tests

All tests live in `tests/`. Tests marked "GPU" must run on the cluster.

| File | GPU | What It Checks |
|------|-----|----------------|
| `tests/test_networks.py` | No | Policy and value network shapes, action space size |
| `tests/test_ppo.py` | No | GAE computation, PPO update mechanics, learning signal |
| `tests/test_pipeline.py` | Yes | Step-by-step denoising, reproducibility, scheduler switching |
| `tests/test_attention.py` | Yes | Attention map dimensions (64×64×77), value validity |
| `tests/test_refine.py` | Yes | Mask generation, soft blur, NFE count, state transition after refine |
| `tests/test_state.py` | Yes | State vector dimensions (1672d), CLIP encoding, caching |
| `tests/test_reward.py` | Yes | Each reward component in isolation, normalization, no-lambda check |
| `tests/test_episode.py` | Yes | Full episode runs, warmup enforced, transitions recorded |
| `tests/test_training_toy.py` | Yes | End-to-end: 2 iterations don't crash, no NaN |
| `tests/test_g1.py` | Yes | Gate 1 validation: loss decreases, actions non-degenerate |
| `tests/run_all.py` | — | Runs all of the above; supports `--cpu-only` flag |

**Gate G1 result: ✓ PASSED** — loss decreases, action distribution is non-degenerate, full iteration runs under 10 min on A100.

---

## Phase 2: Measure and Tune (Weeks 5–7) ✓

**What we did:** Built everything needed to (a) compare the agent against other methods, (b) measure image quality rigorously, and (c) find the best hyperparameters before committing to a full training run.

### Week 5: Baselines — Other Methods to Compare Against

Before claiming the agent is good, we need to know what "good" looks like. These files implement the comparison methods.

**`src/baselines/base.py`**
The shared foundation. Defines the `BaseBaseline` abstract class (every comparison method must implement a `generate()` function) and the `BaselineResult` container (image, prompt, seed, NFE count, wall-clock time). Also contains a helper that saves generated images as PNGs and writes a `results.jsonl` log file in a consistent format that the evaluation pipeline can read.

**`src/baselines/fixed_step.py`**
Seven "just run N steps" baselines: DDIM-20, DDIM-50, DPM-Solver-20, DPM-Solver-50, Euler-20, UniPC-20, and PNDM-20. These are the standard comparison points. DDIM-20 is the main low-NFE reference; DDIM-50 is the "full quality" reference.

**`src/baselines/oracle.py`**
The theoretical ceiling. Runs all 50 denoising steps, decodes the intermediate clean prediction at every step, scores each with CLIP, and returns the image from whichever step had the highest score. In a real deployment you can't do this (you don't know the future), but it shows the best a stop-action policy could ever achieve.

**`src/baselines/random_search.py`**
Generates 8 different images for the same prompt (with different random seeds) and picks the one with the highest CLIP score. Uses the same total NFE budget (8×20=160) as the agent might use. Shows whether the agent's adaptivity beats simple "try multiple times and pick the best."

**`src/baselines/sag.py`**
Self-Attention Guidance — an existing method that uses the model's own attention to sharpen generation. Runs as a drop-in replacement for the standard DDIM pipeline. Included as a quality-boosted comparison at the same NFE budget.

**`src/baselines/attend_excite.py`**
Attend-and-Excite — encourages every token in the prompt to attend to a distinct region of the image. Particularly useful for compositional prompts ("a cat and a dog"). Runs 50 steps (this method needs more steps to work effectively).

**`src/baselines/lcm.py`**
Latent Consistency Models via LoRA. Generates images in just 4 steps by distilling the diffusion model into a consistency model. Represents the "distillation" approach to fast generation — a fundamentally different strategy from our adaptive agent.

**`src/baselines/sdxl_turbo.py`**
SDXL-Turbo — adversarially distilled SDXL in 4 steps. Larger model than SD 1.5 but also distilled to be extremely fast. Requires ~24 GB VRAM (A100-80GB node). Included to show where distillation methods stand on a stronger backbone.

**`src/baselines/__init__.py`**
Exports all baseline classes from one place so they can be imported as `from src.baselines import DDIM20, SAGBaseline`, etc.

Baseline tests (GPU required, run on cluster):

| File | Baselines Covered |
|------|------------------|
| `tests/test_baselines_fixed.py` | DDIM-20/50, DPM-Solver-20/50, Euler-20, UniPC-20, PNDM-20 |
| `tests/test_baselines_search.py` | RandomSearch, OracleStop |
| `tests/test_baselines_attention.py` | SAG, Attend-and-Excite |
| `tests/test_baselines_distill.py` | LCM-4, SDXL-Turbo-4 |

### Week 6: Evaluation Pipeline — How to Score Images

Once baselines and the agent generate images, we need objective ways to measure quality.

**`src/evaluation/compute_metrics.py`**
The main metrics script. Point it at a folder of generated images and a prompts file, and it computes:
- **FID** — how similar is the distribution of generated images to real images? Lower is better.
- **CLIP Score** — does each image actually match its text prompt? Measured per image, then averaged.
- **CMMD** — a newer distance metric that uses CLIP features and an RBF kernel. More sensitive than FID for detecting subtle quality differences.
- **Inception Score (IS)** — are the images both diverse and recognizable?
- **ImageReward** — a model trained on human preferences that scores image quality.
- **HPS v2** — another human-preference model, specifically tuned for comparing generative methods.
- **Aesthetic score** — predicts how aesthetically pleasing the image is.

CMMD has no pip package so it's implemented from scratch using the formula from the paper (arXiv:2401.09603).

**`src/evaluation/vlm_judge.py`**
A GPT-4o-based judge. Shows two images to GPT-4o and asks which is better for a given prompt. The judge randomly swaps which image appears "first" in the request (50% of the time) to avoid the model always preferring whichever it sees first. Results are cached incrementally so a large evaluation run can be interrupted and resumed. Includes a Bradley-Terry model to convert pairwise wins into a ranking.

**`src/evaluation/significance.py`**
Statistical testing. Just because method A scores higher than method B on average doesn't mean the difference is real — it could be noise. This file implements: paired bootstrap testing (resample the per-image scores 10,000 times and check if A consistently beats B), Wilson score confidence intervals for win rates, and a `compare_all_methods()` function that tests every baseline against a reference and formats the results as a LaTeX table for the paper.

Tests for the evaluation pipeline (all CPU-safe, no GPU needed):

| File | What It Checks |
|------|----------------|
| `tests/test_compute_metrics.py` | CMMD kernel shape and symmetry, self-similarity ≈ 0, Wilson CI coverage |
| `tests/test_vlm_judge.py` | Mock OpenAI API: winner parsing, order-swap logic, win-rate aggregation |
| `tests/test_significance.py` | Bootstrap calibration, p-values uniform under null, CI coverage |

SLURM scripts for running baselines and metrics on the cluster:

| File | What It Does |
|------|-------------|
| `scripts/run_baselines.py` | Python runner: takes a list of baseline names, dynamically loads their classes, generates images on a prompt set |
| `scripts/run_baselines.slurm` | SLURM wrapper for the above (1 GPU, 40 GB, 4 h) |
| `scripts/eval_metrics.slurm` | SLURM wrapper for `compute_metrics.py` — computes all metrics for one method's output folder (1 GPU, 40 GB, 2 h) |

### Week 7: Tuning — Finding the Best Settings

Before running a long full training (Phase 3), we sweep hyperparameters on short 200–500-iteration runs to find settings worth committing to.

**`scripts/tune_lr.py`** / **`scripts/tune_lr.slurm`**
Tries three learning rates — 1e-4, 3e-4, and 1e-3 — running 500 training iterations each. Plots learning curves for all three and saves the best to `outputs/tuning/lr_sweep/results.json`.

**`scripts/tune_reward.py`**
Compares three reward "personalities": quality-heavy (prioritize image quality over speed), balanced (the default settings), and efficiency-heavy (strongly penalize extra steps). Runs 500 iterations per profile and writes the winning profile as `configs/tuned_reward.yaml`.

**`scripts/optuna_sweep.py`** / **`scripts/optuna_sweep.slurm`**
A more thorough search using Optuna. Runs 20 trials, each with a randomly sampled set of reward weights (8 hyperparameters total). Each trial trains for 200 iterations and is scored by `mean_reward - 0.5 × std_reward` (we want high average reward with low variance). The top 3 configurations are saved to `configs/optuna_top3.yaml`.

**Important detail:** HPC clusters use a shared network filesystem (NFS) for storage. SQLite databases (the usual Optuna default) corrupt when accessed by multiple jobs simultaneously over NFS. The sweep uses `JournalStorage` with a simple append-only log file instead, which is NFS-safe and supports multiple parallel SLURM jobs sharing a study.

**Gate G2 condition:** Train the top Optuna config for 500 iterations and check whether it beats DDIM-20 on at least one quality metric (CLIP Score, ImageReward, or FID) at a comparable or lower step count (NFE ≤ 30).

---

## Documentation Files

| File | What It Is |
|------|-----------|
| `CLAUDE.md` | Project-wide instructions for the AI assistant — architecture, conventions, pitfalls, phase status |
| `phase1.md` | Detailed technical report on everything built in Phase 1 |
| `phase2.md` | Detailed technical report on everything built in Phase 2, including API corrections found during development |
| `plan.md` | The full 14-week research plan with decision gates (not modified, internal reference) |
| `research.md` | The formal methodology document — treated as ground truth for formulas and definitions |
| `discovery.md` | 38 numbered findings from design review — bugs, ambiguities, and fixes applied to the implementation |

---

## Where We Are Now

```
Phase 0: Setup          ✓ Done
Phase 1: Core Agent     ✓ Done  (G1 passed)
Phase 2: Baselines & Tuning  ✓ Done  (G2 pending cluster validation)
Phase 3: Full Training  ← Next
Phase 4: Generalization
Phase 5: Paper
```

**Total files written so far:** 47 (27 source/script files + 19 test files + 1 config)

**What happens in Phase 3 (Weeks 8–10):**
- Take the best hyperparameters from Phase 2 tuning and run a full 5,000-iteration training on COCO-118K prompts
- Run the five ablation studies (remove reward components, simplify action space, etc.) to understand what actually matters
- Run all baselines and the trained agent on the full benchmark datasets — DrawBench, PartiPrompts, COCO-30K
- Use `compute_metrics.py` and `vlm_judge.py` to produce the final numbers for the paper
- Gate G3: agent achieves quality matching DDIM-50 while using ≤ 30 steps on average

---

*Last updated: March 2026 — Phase 2 complete*
