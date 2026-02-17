# AdDiffusion: Adaptive Agent-Based Inference for Diffusion Models

AdDiffusion reformulates diffusion model inference as a Markov Decision Process. A lightweight PPO-trained policy network observes the evolving latent state during denoising and selects from three actions at each step: **continue**, **stop** (early termination), or **refine** (selective region re-denoising). This enables prompt-dependent adaptive computation — simple prompts terminate early, complex prompts allocate additional effort to under-generated regions.

## Key Idea

Standard diffusion inference uses a fixed number of steps regardless of prompt complexity. AdDiffusion wraps any pretrained diffusion model (without modifying its weights) and learns *when to stop*, *when to continue*, and *when to selectively refine* — all conditioned on the prompt and current generation quality.

```
Noise ──→ [Agent decides at each step] ──→ Final Image
              │
              ├─ continue: standard denoising step
              ├─ stop: early termination (save compute)
              └─ refine: re-denoise under-generated regions
```

## Hypotheses

- **H1 (Efficiency):** Equivalent quality to 50-step inference at ≤30 average NFE
- **H2 (Quality):** Higher quality than fixed-step methods at equal compute budget
- **H3 (Adaptive):** Prompt-dependent behavior — simple prompts stop early, complex prompts use more steps
- **H4 (Refinement):** Selective region refinement outperforms additional full-image steps

## Architecture

| Component | Description |
|-----------|-------------|
| **State** | CLIP image + text features, timestep embedding, quality metrics (~1672-dim) |
| **Policy** | 2-layer MLP → 3-action softmax (continue/stop/refine) |
| **Value** | 2-layer MLP → scalar estimate |
| **Reward** | Quality deltas (CLIP, ImageReward) + DINO stability - NFE penalty + terminal bonus |
| **Refinement** | Cross-attention masks with Gaussian blur, k=2 denoising iterations |

## Project Structure

```
addiffusion/
├── src/
│   ├── diffusion/        # Pipeline wrapper, attention extraction, region refinement
│   ├── agent/            # Policy/value networks, state features, PPO, episode loop
│   ├── rewards/          # Composite reward with normalization
│   ├── evaluation/       # Metrics, VLM-as-Judge, significance testing
│   └── baselines/        # Fixed-step, SAG, Attend-and-Excite, oracle
├── tests/                # Unit + integration tests
├── configs/              # Hydra YAML configs
├── scripts/              # SLURM templates
├── discovery.md          # 38 analytical findings with fixes
├── experiment.md         # Step-by-step experiment execution guide
├── plan.md               # 14-week execution plan with decision gates
└── phase1.md             # Phase 1 completion report
```

## Quick Start

```bash
# Install dependencies (requires Python 3.10, CUDA 12.4)
uv venv --python 3.10 && source .venv/bin/activate
uv sync

# Run CPU-only tests (no GPU needed)
uv run python tests/run_all.py --cpu-only

# Run full test suite (requires GPU + model weights)
uv run python tests/run_all.py

# Train the agent
uv run python src/train.py --config configs/default.yaml

# Or submit to SLURM
sbatch scripts/train.slurm
```

## Evaluation

Benchmarks: COCO-30K, DrawBench, PartiPrompts, GenEval, T2I-CompBench

Metrics: FID, CLIP Score, ImageReward, HPS v2, Aesthetic Score, VLM-as-Judge (GPT-4o + Gemini)

Baselines: DDIM, DPM-Solver, LCM, SDXL-Turbo, SAG, Attend-and-Excite, Oracle-Stop

## Backbones

| Model | Role |
|-------|------|
| SD 1.5 | Primary training backbone |
| SDXL | Scale generalization |
| Flux.1-schnell | Architecture generalization (DiT) |

## Requirements

- Python 3.10, PyTorch 2.6.0, CUDA 12.4
- A100-40GB (SD 1.5) or A100-80GB (SDXL)
- ~800 GPU hours for full pipeline (training + baselines + evaluation)
