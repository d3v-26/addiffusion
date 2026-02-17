# AdDiffusion: Experiment Execution Guide

> Step-by-step protocol for executing all experiments defined in `research.md` and `plan.md`.
> Technical setup follows `setup_guide.md`. All experiment IDs trace to hypotheses in `research.md` §4.1.

---

## 1. Prerequisites Checklist

Complete all items before starting any experiment. References to `setup_guide.md` steps are in parentheses.

### Hardware
- [ ] Access to SLURM-managed HPC cluster with A100 GPUs
- [ ] A100-40GB available for SD 1.5 experiments
- [ ] A100-80GB available for SDXL experiments (plan.md Phase 4)
- [ ] At least 200 GB free storage on scratch filesystem
- [ ] Internet access on login node for model downloads

### Software Environment
- [ ] `uv` installed and on PATH (setup_guide.md Step 2)
- [ ] CUDA 12.4 and GCC 11.4 available via `module load` (setup_guide.md Step 2)
- [ ] Python 3.10 virtual environment created via `uv venv` (setup_guide.md Step 2)
- [ ] `pyproject.toml` has `requires-python = "==3.10.*"` (setup_guide.md Step 2)
- [ ] PyTorch CUDA index configured in `pyproject.toml` with `explicit = true` (setup_guide.md Step 3a)
- [ ] All dependencies installed: core, RL, metrics, baselines, experiment management (setup_guide.md Steps 3a–3e)
- [ ] `uv.lock` generated and committed to git (setup_guide.md Step 3f)

### Models & Data
- [ ] SD 1.5 weights downloaded to `$HF_HOME` (setup_guide.md Step 9a)
- [ ] SDXL weights downloaded (setup_guide.md Step 9a)
- [ ] LCM-LoRA weights downloaded (setup_guide.md Step 9a)
- [ ] SDXL-Turbo weights downloaded (setup_guide.md Step 9a)
- [ ] COCO-30K captions in `data/coco/` (setup_guide.md Step 9b)
- [ ] DrawBench prompts in `data/drawbench/drawbench.json` (setup_guide.md Step 9b)
- [ ] PartiPrompts in `data/partiprompts/parti_prompts.json` (setup_guide.md Step 9b)
- [ ] T2I-CompBench repo cloned to `data/t2i_compbench/repo/` (setup_guide.md Step 9b)
- [ ] GenEval repo cloned to `data/geneval/repo/` (setup_guide.md Step 9b)
- [ ] ImageReward model loadable (setup_guide.md Step 9c)
- [ ] CLIP ViT-L-14 loadable (setup_guide.md Step 9c)

### Configuration
- [ ] `configs/default.yaml` created per setup_guide.md Step 10
- [ ] W&B API key configured (`wandb login`)
- [ ] SLURM template `scripts/train.slurm` created per setup_guide.md Step 11

### Verification
- [ ] `uv run python scripts/verify_setup.py` passes all checks (setup_guide.md Step 12)
- [ ] Sanity check image generated successfully (setup_guide.md Step 13)

---

## 2. Environment Setup

### 2.1 Initial Setup (One-Time)

Follow `setup_guide.md` Steps 1–3 exactly. Key commands:

```bash
# 1. Create project structure
mkdir -p addiffusion/{src/{agent,diffusion,rewards,evaluation,baselines,utils},configs,scripts,checkpoints,data,logs,outputs}

# 2. Install uv and create environment
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
module load cuda/12.4.1
module load gcc/11.4.0
cd addiffusion
uv init --python 3.10
# Edit pyproject.toml: set requires-python = "==3.10.*"
# Edit pyproject.toml: add PyTorch CUDA index (see setup_guide.md Step 3a)
uv venv --python 3.10
source .venv/bin/activate

# 3. Install all dependencies
uv add torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
uv add xformers==0.0.29.post2
uv add diffusers transformers accelerate safetensors
uv add gymnasium stable-baselines3
uv add image-reward hpsv2 open_clip_torch clean-fid torch-fidelity timm
uv add sentencepiece protobuf
uv add optuna wandb tensorboard hydra-core omegaconf
uv add tqdm pillow scipy matplotlib seaborn pandas datasets
uv lock
```

**Expected output of `uv lock`:** `Resolved N packages in X.XXs`. No errors.

**Common failure:** If `uv add torch==2.6.0` fails with resolution errors, verify that the `[[tool.uv.index]]` section in `pyproject.toml` has `explicit = true`. Without it, uv tries to resolve non-torch packages from the PyTorch index.

**Verification:**
```bash
uv run python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```
Expected: `PyTorch 2.6.0, CUDA: True`

### 2.2 Download Models and Data

Follow `setup_guide.md` Steps 9a–9c. On HPC login node:

```bash
export HF_HOME=/scratch/$USER/hf_cache

# Models (Step 9a)
uv run python -c "
from huggingface_hub import snapshot_download
import os
cache = os.environ['HF_HOME']
snapshot_download('stable-diffusion-v1-5/stable-diffusion-v1-5', cache_dir=cache)
snapshot_download('stabilityai/stable-diffusion-xl-base-1.0', cache_dir=cache)
snapshot_download('latent-consistency/lcm-lora-sdv1-5', cache_dir=cache)
snapshot_download('stabilityai/sdxl-turbo', cache_dir=cache)
print('All models downloaded.')
"

# Datasets (Step 9b)
mkdir -p data/{coco,drawbench,partiprompts,geneval,t2i_compbench}
wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip -P data/coco/
unzip data/coco/annotations_trainval2014.zip -d data/coco/

uv run python -c "
from datasets import load_dataset
ds = load_dataset('shunk031/DrawBench', split='train')
ds.to_json('data/drawbench/drawbench.json')
print(f'DrawBench: {len(ds)} prompts')
"

uv run python -c "
from datasets import load_dataset
ds = load_dataset('nateraw/parti-prompts', split='train')
ds.to_json('data/partiprompts/parti_prompts.json')
print(f'PartiPrompts: {len(ds)} prompts')
"

git clone https://github.com/Karine-Huang/T2I-CompBench.git data/t2i_compbench/repo || true
git clone https://github.com/djghosh13/geneval.git data/geneval/repo || true

# Metric weights (Step 9c)
uv run python -c "import ImageReward as RM; model = RM.load('ImageReward-v1.0'); print('ImageReward OK')"
uv run python -c "import open_clip; m,_,p = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai'); print('CLIP OK')"
```

**Verification:** Run `uv run python scripts/verify_setup.py`. All checks must pass.

### 2.3 GPU Memory Audit (Decision Gate G0)

Before proceeding, verify models fit in GPU memory:

```bash
uv run python -c "
import torch
from diffusers import StableDiffusionPipeline
import open_clip
import ImageReward as RM

# Load all models simultaneously
pipe = StableDiffusionPipeline.from_pretrained(
    'stable-diffusion-v1-5/stable-diffusion-v1-5', torch_dtype=torch.float16
).to('cuda')
clip_model, _, _ = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
clip_model = clip_model.to('cuda').half()
ir_model = RM.load('ImageReward-v1.0')

allocated = torch.cuda.memory_allocated() / 1e9
reserved = torch.cuda.memory_reserved() / 1e9
print(f'Allocated: {allocated:.1f} GB, Reserved: {reserved:.1f} GB')

# Generate one image to test peak memory
image = pipe('a test prompt', num_inference_steps=20).images[0]
peak = torch.cuda.max_memory_allocated() / 1e9
print(f'Peak after generation: {peak:.1f} GB')
"
```

**Decision Gate G0:** If peak > 35 GB on A100-40GB → implement serialized reward computation (see plan.md G0).

---

## 3. Experiment Execution Protocol

### Naming Convention

All experiments follow the ID pattern: `EXP-{category}{number}` where categories are:
- `B` = Baseline
- `T` = Agent training
- `A` = Ablation
- `E` = Evaluation
- `G` = Generalization

### Configuration Override Convention

Each experiment uses a base config (`configs/default.yaml`) with specific overrides. Overrides are specified as Hydra command-line arguments:

```bash
uv run python src/{script}.py --config configs/default.yaml override.key=value
```

---

### EXP-B01: Fixed-Step Baseline Generation
**Objective:** Generate images with all fixed-step schedulers (research.md §4.3, B1). Provides reference quality for H1, H2.

**Procedure:**
1. For each scheduler in {DDIM, DPM-Solver, Euler, UniPC, PNDM}:
2. For each step count in {20, 50} (where applicable):
3. Generate images for all prompts in DrawBench (200) and COCO-30K (30,000):

```bash
# Example: DDIM-20 on DrawBench
uv run python src/baselines/fixed_step.py \
    --scheduler ddim \
    --steps 20 \
    --dataset drawbench \
    --prompt_file data/drawbench/drawbench.json \
    --output_dir outputs/baselines/ddim_20/drawbench/ \
    --model_id stable-diffusion-v1-5/stable-diffusion-v1-5 \
    --guidance_scale 7.5 \
    --seed 42
```

Repeat for all 7 scheduler-step combinations: DDIM-20, DDIM-50, DPM-Solver-20, DPM-Solver-50, Euler-20, UniPC-20, PNDM-20.

**Configuration:**

| Parameter | Value |
|-----------|-------|
| Model | SD 1.5 |
| Resolution | 512×512 |
| Guidance scale | 7.5 |
| Seeds | 42 (primary), {0,1,2,3,4} for FID confidence |
| Precision | float16 |

**Expected Outputs:**
- `outputs/baselines/{scheduler}_{steps}/{dataset}/` — generated images (PNG)
- `outputs/baselines/{scheduler}_{steps}/{dataset}/metadata.json` — prompts, seeds, timing

**Validation:**
- Check image count matches prompt count (200 for DrawBench, 30K for COCO)
- Spot-check 10 random images for visual sanity
- Verify wall-clock time is within expected range (2–6s per image for 20–50 steps)

**Logging:** W&B run tagged `baseline/{scheduler}_{steps}`

---

### EXP-B02: Distillation Baseline Generation
**Objective:** Generate images with LCM-4 and SDXL-Turbo-4 (research.md §4.3, B5). Establishes efficiency frontier lower bound.

**Procedure:**

```bash
# LCM-4 (SD 1.5 + LCM-LoRA)
uv run python src/baselines/fixed_step.py \
    --scheduler lcm \
    --steps 4 \
    --dataset drawbench \
    --prompt_file data/drawbench/drawbench.json \
    --output_dir outputs/baselines/lcm_4/drawbench/ \
    --model_id stable-diffusion-v1-5/stable-diffusion-v1-5 \
    --lora_id latent-consistency/lcm-lora-sdv1-5 \
    --guidance_scale 1.0 \
    --seed 42

# SDXL-Turbo-4
uv run python src/baselines/fixed_step.py \
    --scheduler euler \
    --steps 4 \
    --dataset drawbench \
    --prompt_file data/drawbench/drawbench.json \
    --output_dir outputs/baselines/sdxl_turbo_4/drawbench/ \
    --model_id stabilityai/sdxl-turbo \
    --guidance_scale 0.0 \
    --seed 42
```

**Configuration:**

| Parameter | LCM-4 | SDXL-Turbo-4 |
|-----------|-------|--------------|
| Model | SD 1.5 + LCM-LoRA | SDXL-Turbo |
| Steps | 4 | 4 |
| Guidance scale | 1.0 (LCM requires low CFG) | 0.0 (Turbo requires no CFG) |
| Resolution | 512×512 | 512×512 |

**Expected Outputs:** Same structure as EXP-B01.

**Validation:** LCM images should appear lower-quality but coherent. Turbo images should be comparable quality at 512×512.

---

### EXP-B03: Region-Aware Baseline Generation
**Objective:** Generate images with SAG and Attend-and-Excite (research.md §4.3, B6). These are the closest baselines to our region refinement approach.

**Procedure:**

```bash
# SAG (20 steps)
uv run python src/baselines/sag.py \
    --steps 20 \
    --dataset drawbench \
    --prompt_file data/drawbench/drawbench.json \
    --output_dir outputs/baselines/sag_20/drawbench/ \
    --model_id stable-diffusion-v1-5/stable-diffusion-v1-5 \
    --sag_scale 0.75 \
    --seed 42

# Attend-and-Excite (50 steps)
uv run python src/baselines/attend_excite.py \
    --steps 50 \
    --dataset drawbench \
    --prompt_file data/drawbench/drawbench.json \
    --output_dir outputs/baselines/attend_excite_50/drawbench/ \
    --model_id stable-diffusion-v1-5/stable-diffusion-v1-5 \
    --max_iter_to_alter 25 \
    --seed 42
```

**Configuration:**

| Parameter | SAG | Attend-and-Excite |
|-----------|-----|-------------------|
| Steps | 20 | 50 |
| SAG scale | 0.75 | N/A |
| Max iter to alter | N/A | 25 (first half of steps) |
| Guidance scale | 7.5 | 7.5 |

**Validation:** SAG images should show improved detail vs. DDIM-20. A&E images should show better text-image alignment for compositional prompts.

---

### EXP-B04: Oracle Baseline Generation
**Objective:** Establish upper bound on adaptive stopping (research.md §4.3, B4). Oracle-Stop has perfect knowledge of when to stop.

**Procedure:**

```bash
uv run python src/baselines/oracle.py \
    --mode stop \
    --max_steps 50 \
    --dataset drawbench \
    --prompt_file data/drawbench/drawbench.json \
    --output_dir outputs/baselines/oracle_stop/drawbench/ \
    --model_id stable-diffusion-v1-5/stable-diffusion-v1-5 \
    --metric clip_score \
    --seed 42
```

The oracle evaluates CLIP score at every step and returns the image from the step with the highest score.

**Expected Outputs:** Images + metadata including the optimal stopping step per prompt.

**Validation:** Oracle should achieve highest CLIP score of any method. Average stopping step should vary across prompts (some early, some late).

---

### EXP-T01: Agent Training (Primary)
**Objective:** Train the AdDiffusion agent on SD 1.5 (research.md §3, plan.md Phase 3). Tests all four hypotheses.

**Procedure:**

```bash
# Submit SLURM job for full training
sbatch scripts/train.slurm
```

Or run interactively for debugging:

```bash
uv run python src/train.py \
    --config configs/default.yaml \
    training.total_iterations=2000 \
    training.prompt_dataset=data/coco/annotations/captions_val2014.json \
    training.seed=42 \
    ppo.batch_size=64 \
    logging.use_wandb=true \
    logging.project=addiffusion
```

**Configuration:** All values from `configs/default.yaml` (see research.md §3.6 hyperparameter table). Key parameters:

| Parameter | Value | Source |
|-----------|-------|--------|
| Learning rate | 3e-4 | research.md §3.6 |
| Batch size | 64 | research.md §3.6 |
| PPO clip ε | 0.2 | research.md §3.6 |
| GAE λ | 0.95 | research.md §3.6 |
| Discount γ_d | 0.99 | research.md §3.6 (renamed per D-09) |
| NFE penalty c_nfe | 0.01 | research.md §3.6 (renamed per D-09) |
| Max steps | 50 | research.md §3.6 |
| Warmup steps | 3 | research.md §3.2 (mandatory continues) |
| Reward normalization | Enabled | research.md §3.4 (per D-37) |
| Mask blurring σ | 3 | research.md §3.5 (per D-26) |

**Curriculum schedule (per discovery.md D-32):**
- Iterations 1–500: Simple prompts only (single-object from COCO)
- Iterations 501–1200: Mixed simple + medium prompts
- Iterations 1201–2000: Full prompt distribution

**Expected Outputs:**
- `checkpoints/agent_v1_iter{N}.pt` — policy + value network weights (every 50 iterations)
- `logs/train_{jobid}.out` — SLURM output
- W&B dashboard with: average reward, average NFE, action distribution, CLIP score trajectory

**Validation Criteria:**
1. Training loss decreases over iterations
2. Average reward increases
3. Average NFE decreases from $N_\text{max}$ (agent learns to stop early)
4. Action distribution is non-degenerate (not 100% one action) by iteration 200
5. No evidence of reward hacking: spot-check 20 images at iterations 500, 1000, 1500, 2000

**Logging Requirements:**
- Per iteration: average reward (total + each component), average NFE, action distribution (% continue/stop/refine), average CLIP score, average ImageReward
- Per checkpoint: 8 sample images with their prompts and action sequences
- Training curves: loss, reward, NFE, entropy over iterations

**Troubleshooting:**
- *Agent always continues (never stops/refines):* Entropy coefficient too low → increase $c_2$ to 0.05. Or $c_\text{nfe}$ too low → increase to 0.02.
- *Agent always stops immediately:* $c_\text{nfe}$ too high → decrease to 0.005. Or terminal reward too dominant → reduce $\beta$ values.
- *Loss diverges:* Learning rate too high → reduce to 1e-4. Gradient explosion → reduce max_grad_norm to 0.1.
- *OOM during training:* Reduce batch size to 32. Enable gradient checkpointing on diffusion model. Serialize reward model loading.

---

### EXP-A01: Reward Component Ablation
**Objective:** Determine contribution of each reward component (research.md §4.4, A1). Tests whether dense reward (D-02) and multi-metric design (D-06) are necessary.

**Procedure:** Run 4 training variants, 1000 iterations each:

```bash
# Full (control)
uv run python src/train.py --config configs/default.yaml \
    training.total_iterations=1000 \
    experiment.name=ablation_A1_full

# No-Efficiency (remove NFE penalty)
uv run python src/train.py --config configs/default.yaml \
    training.total_iterations=1000 \
    reward.gamma_nfe=0.0 \
    experiment.name=ablation_A1_no_efficiency

# No-Quality (remove per-step quality deltas)
uv run python src/train.py --config configs/default.yaml \
    training.total_iterations=1000 \
    reward.alpha_1=0.0 reward.alpha_2=0.0 reward.alpha_3=0.0 reward.alpha_4=0.0 \
    experiment.name=ablation_A1_no_quality

# Terminal-Only (remove both per-step rewards)
uv run python src/train.py --config configs/default.yaml \
    training.total_iterations=1000 \
    reward.alpha_1=0.0 reward.alpha_2=0.0 reward.alpha_3=0.0 reward.alpha_4=0.0 \
    reward.gamma_nfe=0.0 \
    experiment.name=ablation_A1_terminal_only
```

**Expected Outputs:** 4 trained agents + evaluation on DrawBench.

**Validation Criteria:**
- Full $\geq$ No-Efficiency $>$ No-Quality $>$ Terminal-Only (expected quality ordering)
- No-Efficiency agent should use max steps (no incentive to stop)
- Terminal-Only should show slowest learning convergence

---

### EXP-A02: Action Space Ablation
**Objective:** Determine contribution of each action type (research.md §4.4, A2). Tests H4 (refinement value).

**Procedure:** Run 4 training variants, 1000 iterations each:

```bash
# Full (control) — same as A1_full
# No-Refine
uv run python src/train.py --config configs/default.yaml \
    training.total_iterations=1000 \
    agent.action_space='["continue","stop"]' \
    experiment.name=ablation_A2_no_refine

# No-Stop
uv run python src/train.py --config configs/default.yaml \
    training.total_iterations=1000 \
    agent.action_space='["continue","refine"]' \
    experiment.name=ablation_A2_no_stop

# Continue-Only
uv run python src/train.py --config configs/default.yaml \
    training.total_iterations=1000 \
    agent.action_space='["continue"]' \
    experiment.name=ablation_A2_continue_only
```

**Validation Criteria:**
- Continue-Only should match DDIM-50 (proves agent adds value above fixed schedule)
- No-Stop should use more NFE than Full (no early termination)
- No-Refine vs. Full should show quality difference on complex prompts specifically

---

### EXP-A03: State Representation Ablation
**Objective:** Determine which state features the agent uses (research.md §4.4, A3).

**Procedure:** Run 4 training variants, 1000 iterations each, modifying the state feature vector:

```bash
# Full (control)
# No-Quality (remove quality metrics from state)
uv run python src/train.py --config configs/default.yaml \
    training.total_iterations=1000 \
    agent.use_quality_features=false \
    experiment.name=ablation_A3_no_quality

# No-CLIP (remove CLIP image and text features)
uv run python src/train.py --config configs/default.yaml \
    training.total_iterations=1000 \
    agent.use_clip_features=false \
    experiment.name=ablation_A3_no_clip

# Minimal (only timestep embedding)
uv run python src/train.py --config configs/default.yaml \
    training.total_iterations=1000 \
    agent.use_clip_features=false \
    agent.use_quality_features=false \
    experiment.name=ablation_A3_minimal
```

**Validation Criteria:**
- Minimal should collapse to a learned fixed schedule (no content-awareness)
- No-CLIP should show significant degradation (agent can't understand content)
- No-Quality should show moderate degradation

---

### EXP-A04: Region Selection Ablation
**Objective:** Compare mask generation strategies (research.md §4.4, A4).

**Procedure:** Run 4 training variants, 1000 iterations each:

```bash
# Attention-based (default)
# Grid-based
uv run python src/train.py --config configs/default.yaml \
    training.total_iterations=1000 \
    refinement.method=grid \
    experiment.name=ablation_A4_grid

# Learned CNN
uv run python src/train.py --config configs/default.yaml \
    training.total_iterations=1000 \
    refinement.method=learned \
    experiment.name=ablation_A4_learned

# Random
uv run python src/train.py --config configs/default.yaml \
    training.total_iterations=1000 \
    refinement.method=random \
    experiment.name=ablation_A4_random
```

**Validation Criteria:**
- Random should be worst (proves targeted selection matters)
- Attention-based should be best or tied with Learned

---

### EXP-A05: Reward Normalization Ablation
**Objective:** Test whether reward normalization matters (discovery.md D-37, research.md §4.4 A5).

**Procedure:**

```bash
# Normalized (default) — reuse A1_full
# Raw (unnormalized)
uv run python src/train.py --config configs/default.yaml \
    training.total_iterations=1000 \
    reward.normalize=false \
    experiment.name=ablation_A5_raw
```

**Validation Criteria:**
- Normalized should show faster convergence and better final performance
- Raw may still converge but with more variance

---

### EXP-E01: Full Benchmark Evaluation
**Objective:** Evaluate the trained agent (from EXP-T01) on all benchmarks (research.md §4.2). Primary results for the paper.

**Procedure:** For each benchmark, generate images with the trained agent, then compute all metrics:

```bash
# Generate with agent on each benchmark
for BENCHMARK in drawbench partiprompts geneval t2i_compbench coco30k; do
    uv run python src/evaluate.py \
        --agent_checkpoint checkpoints/agent_v1_final.pt \
        --config configs/default.yaml \
        --dataset $BENCHMARK \
        --output_dir outputs/agent/$BENCHMARK/ \
        --seed 42
done

# Compute metrics
uv run python src/evaluation/compute_metrics.py \
    --generated_dir outputs/agent/ \
    --reference_dir data/coco/val2014/ \
    --metrics fid,clip_score,cmmd,aesthetic,is,image_reward,hps_v2 \
    --output outputs/agent/metrics.json

# FID with 5 seeds for confidence interval (COCO-30K only)
for SEED in 0 1 2 3 4; do
    uv run python src/evaluate.py \
        --agent_checkpoint checkpoints/agent_v1_final.pt \
        --config configs/default.yaml \
        --dataset coco30k \
        --output_dir outputs/agent/coco30k_seed${SEED}/ \
        --seed $SEED
    uv run python src/evaluation/compute_fid.py \
        --generated_dir outputs/agent/coco30k_seed${SEED}/ \
        --reference_dir data/coco/val2014/ \
        --output outputs/agent/fid_seed${SEED}.json
done
```

**Expected Outputs:**
- `outputs/agent/{benchmark}/` — generated images
- `outputs/agent/metrics.json` — all quality + efficiency metrics
- `outputs/agent/fid_seed{0-4}.json` — FID values per seed

**Validation Criteria:**
- FID confidence interval (5 seeds) has reasonable width (±1.0)
- CLIP score ≥ DDIM-50 CLIP score (H1)
- Average NFE ≤ 30 (H1)
- All metrics are within plausible ranges

---

### EXP-E02: VLM-as-Judge Evaluation
**Objective:** Pairwise comparison via GPT-4o and Gemini 2.0 Flash (research.md §4.6).

**Procedure:**

```bash
# Pairwise comparison: Agent vs each baseline on DrawBench
uv run python src/evaluation/vlm_judge.py \
    --method_a_dir outputs/agent/drawbench/ \
    --method_b_dir outputs/baselines/ddim_50/drawbench/ \
    --prompts_file data/drawbench/drawbench.json \
    --judge gpt-4o \
    --output outputs/vlm_judge/agent_vs_ddim50_gpt4o.json

# Repeat with Gemini for cross-validation
uv run python src/evaluation/vlm_judge.py \
    --method_a_dir outputs/agent/drawbench/ \
    --method_b_dir outputs/baselines/ddim_50/drawbench/ \
    --prompts_file data/drawbench/drawbench.json \
    --judge gemini-2.0-flash \
    --output outputs/vlm_judge/agent_vs_ddim50_gemini.json
```

Run for all key comparisons: Agent vs {DDIM-20, DDIM-50, DPM-Solver-20, SAG-20, Attend-Excite-50, LCM-4}.

**Expected Outputs:**
- `outputs/vlm_judge/agent_vs_{baseline}_{judge}.json` — per-prompt judgments (A/B/Tie + justification)
- Win rate computation with Bradley-Terry model
- Cohen's κ between GPT-4o and Gemini judges

**Validation:**
- Presentation order is randomized (check `order` field in output JSON)
- Inter-judge agreement (Cohen's κ) should be > 0.4 (moderate agreement)

---

### EXP-E03: Statistical Significance Testing
**Objective:** Confirm statistical significance of all comparisons (research.md §4.7).

**Procedure:**

```bash
uv run python src/evaluation/significance.py \
    --results_dir outputs/ \
    --metrics clip_score,fid,image_reward,hps_v2 \
    --method paired_bootstrap \
    --n_resamples 10000 \
    --alpha 0.05 \
    --output outputs/significance_report.json
```

**Expected Outputs:**
- `outputs/significance_report.json` — p-values and 95% CIs for all metric comparisons
- Wilson score intervals for VLM-judge win rates

**Validation:** Report only claims that are statistically significant at p < 0.05.

---

### EXP-G01: Cross-Backbone Generalization
**Objective:** Test agent trained on SD 1.5 on SDXL and Flux.1 (research.md §4.8, plan.md Phase 4).

**Procedure:**

```bash
# Zero-shot transfer to SDXL
uv run python src/evaluate.py \
    --agent_checkpoint checkpoints/agent_v1_final.pt \
    --config configs/sdxl.yaml \
    --dataset drawbench \
    --output_dir outputs/generalization/sdxl_zero_shot/ \
    --seed 42

# Zero-shot transfer to Flux.1
uv run python src/evaluate.py \
    --agent_checkpoint checkpoints/agent_v1_final.pt \
    --config configs/flux.yaml \
    --dataset drawbench \
    --output_dir outputs/generalization/flux_zero_shot/ \
    --seed 42

# If zero-shot fails (>5% CLIP drop): train architecture-specific head
uv run python src/train.py \
    --config configs/sdxl.yaml \
    training.total_iterations=500 \
    agent.load_backbone=checkpoints/agent_v1_final.pt \
    agent.train_head_only=true \
    experiment.name=generalization_sdxl_adapted
```

**Validation Criteria:**
- Zero-shot: CLIP score within 5% of SD 1.5 results
- If adapted: should recover to near-SD 1.5 performance

---

### EXP-G02: Scheduler Robustness
**Objective:** Test agent with schedulers it was not trained on (discovery.md D-30).

**Procedure:**

```bash
for SCHEDULER in euler unipc pndm; do
    uv run python src/evaluate.py \
        --agent_checkpoint checkpoints/agent_v1_final.pt \
        --config configs/default.yaml \
        --scheduler $SCHEDULER \
        --dataset drawbench \
        --output_dir outputs/generalization/scheduler_${SCHEDULER}/ \
        --seed 42
done
```

**Validation:** Performance degradation ≤ 5% CLIP score vs. training scheduler (DDIM). If > 5%, note as a limitation.

---

## 4. Data Management Protocol

### File Naming Convention

```
outputs/
├── baselines/
│   ├── {scheduler}_{steps}/
│   │   ├── {benchmark}/
│   │   │   ├── {prompt_id:05d}.png          # Generated image
│   │   │   ├── metadata.json                 # Prompts, seeds, timing
│   │   │   └── metrics.json                  # Computed metrics
│   │   └── ...
│   └── ...
├── agent/
│   ├── {benchmark}/
│   │   ├── {prompt_id:05d}.png
│   │   ├── {prompt_id:05d}_actions.json      # Per-step action log
│   │   ├── metadata.json
│   │   └── metrics.json
│   └── ...
├── ablations/
│   ├── {ablation_id}_{variant}/
│   │   └── {benchmark}/...
│   └── ...
├── generalization/
│   └── {backbone_or_scheduler}/
│       └── {benchmark}/...
├── vlm_judge/
│   └── {method_a}_vs_{method_b}_{judge}.json
└── significance_report.json
```

### Checkpoint Strategy

- Agent checkpoints saved every 50 iterations: `checkpoints/agent_v1_iter{N}.pt`
- Final checkpoint: `checkpoints/agent_v1_final.pt`
- Ablation checkpoints: `checkpoints/ablation_{id}_{variant}_final.pt`
- Each checkpoint contains: policy weights, value weights, optimizer state, iteration count, config

### Version Control

- All code in git; commit before each experiment
- `configs/` tracked in git
- `uv.lock` tracked in git (exact reproducibility)
- `outputs/`, `checkpoints/`, `data/` in `.gitignore` (too large)
- W&B tracks all experiment configs and metrics

---

## 5. Results Collection Template

### Per-Experiment Results Record

```yaml
experiment_id: EXP-XXX
date: YYYY-MM-DD
config_hash: <sha256 of config yaml>
git_commit: <commit hash>
wandb_run: <run URL>

results:
  metrics:
    clip_score: {mean: 0.XX, ci_95: [0.XX, 0.XX]}
    fid: {mean: XX.X, ci_95: [XX.X, XX.X]}
    image_reward: {mean: 0.XX, ci_95: [0.XX, 0.XX]}
    hps_v2: {mean: 0.XX}
    aesthetic: {mean: X.XX}
    avg_nfe: {mean: XX.X, std: X.X}
    avg_time_s: {mean: X.X, std: X.X}

  agent_behavior:
    action_distribution: {continue: 0.XX, stop: 0.XX, refine: 0.XX}
    avg_stop_step: XX.X
    nfe_by_complexity: {simple: XX, medium: XX, complex: XX}

  significance:
    vs_ddim50_clip: {p_value: 0.XXX, significant: true/false}
    vs_ddim50_fid: {p_value: 0.XXX, significant: true/false}

notes: "Free-text observations, anomalies, or issues encountered"
```

### Ablation Summary Table Template

| Ablation | Variant | CLIP Score | FID | ImageReward | Avg NFE | p-value (vs Full) |
|----------|---------|------------|-----|-------------|---------|-------------------|
| A1 | Full | | | | | — |
| A1 | No-Efficiency | | | | | |
| A1 | No-Quality | | | | | |
| A1 | Terminal-Only | | | | | |

---

## 6. Troubleshooting Guide

### Environment Issues

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `uv add torch` fails with resolution error | Missing `explicit = true` in PyTorch index config | Add `explicit = true` to `[[tool.uv.index]]` in pyproject.toml (setup_guide.md §3a) |
| `ModuleNotFoundError: No module named 'xformers'` | xformers version mismatch with torch | Ensure xformers==0.0.29.post2 matches torch==2.6.0 |
| `CUDA out of memory` during training | Too many models co-loaded | Serialize reward computation; reduce batch to 32; see D-25 |
| `CUDA not available` on compute node | Module not loaded | Add `module load cuda/12.4.1` to SLURM script |
| `uv run` uses wrong Python | Stale venv or system Python | `source .venv/bin/activate` first; verify with `which python` |

### Training Issues

| Symptom | Likely Cause | Fix | Reference |
|---------|-------------|-----|-----------|
| Loss diverges / NaN | Learning rate too high | Reduce to 1e-4; check reward normalization | D-37 |
| Agent always picks same action | Entropy too low; or reward dominated by one component | Increase $c_2$ to 0.05; check per-component reward magnitudes | D-37 |
| Agent never refines | Refine action has high NFE cost | Reduce $c_\text{nfe}$; verify refine action produces quality improvement | D-10 |
| Training very slow | PPO sample inefficiency | Reduce batch to 32; consider offline RL fallback | D-29 |
| Images look good but metrics bad | Reward hacking / metric mismatch | VLM spot check; increase $\alpha_3$ (stability) | D-28 |
| Boundary artifacts after refinement | Hard binary mask | Increase Gaussian blur σ from 3 to 5 | D-26 |
| Agent stops too early on complex prompts | $c_\text{nfe}$ too high | Reduce by 50%; verify terminal reward is reached | — |

### Evaluation Issues

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| FID much higher than expected | Wrong reference statistics; too few samples | Verify reference images are correct COCO val set; ensure 30K samples |
| CLIP score suspiciously high | Wrong CLIP model version | Verify using ViT-L-14 with openai pretrained weights |
| VLM judge shows strong positional bias | Order not randomized | Check `order` field in output; ensure 50/50 A-first / B-first split |
| ImageReward returns errors | Model loading issue | Re-download with `RM.load('ImageReward-v1.0')` |

---

*Document Version: 1.0*
*Created: February 2026*
*Sources: `setup_guide.md`, `research.md` (v1.0), `plan.md` (v1.0), `discovery.md` (v1.0)*
