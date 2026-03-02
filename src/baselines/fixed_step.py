"""Fixed-step scheduler baselines for SD 1.5 (Phase 2, Week 5).

Seven deterministic baselines that sweep scheduler type and step count.
These form the primary comparison points for H1 (efficiency) and H2 (quality)
hypotheses described in plan.md Phase 2 Week 5.

References:
    - plan.md Phase 2 Week 5 (baseline implementation)
    - research.md §4.1 (experimental baselines)
    - discovery.md D-15 (NFE = scheduler steps, not network evals)
    - discovery.md D-30 (use deterministic samplers for training comparisons)
"""

from __future__ import annotations

import time

import torch
from diffusers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler,
    PNDMScheduler,
    UniPCMultistepScheduler,
)
from PIL import Image

from src.baselines.base import BaseBaseline, BaselineResult, load_sd15_pipeline, save_results


# ---------------------------------------------------------------------------
# DDIM baselines
# ---------------------------------------------------------------------------


class DDIM20(BaseBaseline):
    """DDIM with 20 denoising steps.

    Primary efficiency baseline: NFE=20.  Used to check whether the agent
    matches DDIM-20 quality at equal or lower NFE (hypothesis H1).

    References:
        - research.md §4.1 (B1-DDIM baselines)
        - discovery.md D-30 (deterministic samplers)
    """

    name = "ddim20"
    nfe = 20

    def __init__(self, device: str = "cuda") -> None:
        self.device = device
        self.pipe = load_sd15_pipeline(device=device)
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)

    def generate(
        self,
        prompts: list[str],
        seeds: list[int],
        output_dir: str,
        **kwargs,
    ) -> list[BaselineResult]:
        """Generate one image per (prompt, seed) pair using DDIM-20."""
        results = []
        for idx, (prompt, seed) in enumerate(zip(prompts, seeds)):
            gen = torch.Generator(self.device).manual_seed(seed)
            t0 = time.time()
            image = self.pipe(
                prompt,
                num_inference_steps=self.nfe,
                generator=gen,
            ).images[0]
            results.append(
                BaselineResult(
                    image=image,
                    prompt=prompt,
                    seed=seed,
                    nfe=self.nfe,
                    time_s=time.time() - t0,
                    method_name=self.name,
                    metadata={},
                )
            )
        return save_results(results, output_dir)


class DDIM50(BaseBaseline):
    """DDIM with 50 denoising steps.

    Full-quality reference baseline matching the agent's N_max episode length.
    The agent aims to match DDIM-50 quality at lower NFE (hypothesis H1).

    References:
        - research.md §4.1 (B1-DDIM baselines)
        - plan.md §Episode Structure (N_max = scheduler step count)
    """

    name = "ddim50"
    nfe = 50

    def __init__(self, device: str = "cuda") -> None:
        self.device = device
        self.pipe = load_sd15_pipeline(device=device)
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)

    def generate(
        self,
        prompts: list[str],
        seeds: list[int],
        output_dir: str,
        **kwargs,
    ) -> list[BaselineResult]:
        """Generate one image per (prompt, seed) pair using DDIM-50."""
        results = []
        for idx, (prompt, seed) in enumerate(zip(prompts, seeds)):
            gen = torch.Generator(self.device).manual_seed(seed)
            t0 = time.time()
            image = self.pipe(
                prompt,
                num_inference_steps=self.nfe,
                generator=gen,
            ).images[0]
            results.append(
                BaselineResult(
                    image=image,
                    prompt=prompt,
                    seed=seed,
                    nfe=self.nfe,
                    time_s=time.time() - t0,
                    method_name=self.name,
                    metadata={},
                )
            )
        return save_results(results, output_dir)


# ---------------------------------------------------------------------------
# DPM-Solver++ baselines
# ---------------------------------------------------------------------------


class DPMSolver20(BaseBaseline):
    """DPM-Solver++ with 20 denoising steps.

    High-quality fast scheduler; default algorithm_type is "dpmsolver++"
    (do not override — see verified API notes in task spec).

    References:
        - research.md §4.1 (additional scheduler baselines)
        - discovery.md D-30 (deterministic samplers)
    """

    name = "dpm_solver20"
    nfe = 20

    def __init__(self, device: str = "cuda") -> None:
        self.device = device
        self.pipe = load_sd15_pipeline(device=device)
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )

    def generate(
        self,
        prompts: list[str],
        seeds: list[int],
        output_dir: str,
        **kwargs,
    ) -> list[BaselineResult]:
        """Generate one image per (prompt, seed) pair using DPM-Solver++-20."""
        results = []
        for idx, (prompt, seed) in enumerate(zip(prompts, seeds)):
            gen = torch.Generator(self.device).manual_seed(seed)
            t0 = time.time()
            image = self.pipe(
                prompt,
                num_inference_steps=self.nfe,
                generator=gen,
            ).images[0]
            results.append(
                BaselineResult(
                    image=image,
                    prompt=prompt,
                    seed=seed,
                    nfe=self.nfe,
                    time_s=time.time() - t0,
                    method_name=self.name,
                    metadata={},
                )
            )
        return save_results(results, output_dir)


class DPMSolver50(BaseBaseline):
    """DPM-Solver++ with 50 denoising steps.

    Full-quality DPM-Solver++ reference, complementing DDIM-50 for scheduler
    comparisons in experiment EXP-B2 (plan.md Phase 2 Week 5).

    References:
        - research.md §4.1 (additional scheduler baselines)
    """

    name = "dpm_solver50"
    nfe = 50

    def __init__(self, device: str = "cuda") -> None:
        self.device = device
        self.pipe = load_sd15_pipeline(device=device)
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )

    def generate(
        self,
        prompts: list[str],
        seeds: list[int],
        output_dir: str,
        **kwargs,
    ) -> list[BaselineResult]:
        """Generate one image per (prompt, seed) pair using DPM-Solver++-50."""
        results = []
        for idx, (prompt, seed) in enumerate(zip(prompts, seeds)):
            gen = torch.Generator(self.device).manual_seed(seed)
            t0 = time.time()
            image = self.pipe(
                prompt,
                num_inference_steps=self.nfe,
                generator=gen,
            ).images[0]
            results.append(
                BaselineResult(
                    image=image,
                    prompt=prompt,
                    seed=seed,
                    nfe=self.nfe,
                    time_s=time.time() - t0,
                    method_name=self.name,
                    metadata={},
                )
            )
        return save_results(results, output_dir)


# ---------------------------------------------------------------------------
# Euler baseline
# ---------------------------------------------------------------------------


class Euler20(BaseBaseline):
    """Euler Discrete scheduler with 20 denoising steps.

    Stochastic scheduler used here at fixed 20 steps for a scheduler-diversity
    comparison point.  Note: research.md D-30 recommends deterministic samplers
    for training; this class is used only for evaluation comparison.

    References:
        - research.md §4.1 (additional scheduler baselines)
        - discovery.md D-30 (note re: stochastic eval-only use)
    """

    name = "euler20"
    nfe = 20

    def __init__(self, device: str = "cuda") -> None:
        self.device = device
        self.pipe = load_sd15_pipeline(device=device)
        self.pipe.scheduler = EulerDiscreteScheduler.from_config(
            self.pipe.scheduler.config
        )

    def generate(
        self,
        prompts: list[str],
        seeds: list[int],
        output_dir: str,
        **kwargs,
    ) -> list[BaselineResult]:
        """Generate one image per (prompt, seed) pair using Euler-20."""
        results = []
        for idx, (prompt, seed) in enumerate(zip(prompts, seeds)):
            gen = torch.Generator(self.device).manual_seed(seed)
            t0 = time.time()
            image = self.pipe(
                prompt,
                num_inference_steps=self.nfe,
                generator=gen,
            ).images[0]
            results.append(
                BaselineResult(
                    image=image,
                    prompt=prompt,
                    seed=seed,
                    nfe=self.nfe,
                    time_s=time.time() - t0,
                    method_name=self.name,
                    metadata={},
                )
            )
        return save_results(results, output_dir)


# ---------------------------------------------------------------------------
# UniPC baseline
# ---------------------------------------------------------------------------


class UniPC20(BaseBaseline):
    """UniPC multistep scheduler with 20 denoising steps.

    Predictor-corrector scheduler known for high sample quality at low NFE.
    Serves as a challenging fixed-step comparison for efficiency hypothesis H1.

    References:
        - research.md §4.1 (additional scheduler baselines)
    """

    name = "unipc20"
    nfe = 20

    def __init__(self, device: str = "cuda") -> None:
        self.device = device
        self.pipe = load_sd15_pipeline(device=device)
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )

    def generate(
        self,
        prompts: list[str],
        seeds: list[int],
        output_dir: str,
        **kwargs,
    ) -> list[BaselineResult]:
        """Generate one image per (prompt, seed) pair using UniPC-20."""
        results = []
        for idx, (prompt, seed) in enumerate(zip(prompts, seeds)):
            gen = torch.Generator(self.device).manual_seed(seed)
            t0 = time.time()
            image = self.pipe(
                prompt,
                num_inference_steps=self.nfe,
                generator=gen,
            ).images[0]
            results.append(
                BaselineResult(
                    image=image,
                    prompt=prompt,
                    seed=seed,
                    nfe=self.nfe,
                    time_s=time.time() - t0,
                    method_name=self.name,
                    metadata={},
                )
            )
        return save_results(results, output_dir)


# ---------------------------------------------------------------------------
# PNDM baseline
# ---------------------------------------------------------------------------


class PNDM20(BaseBaseline):
    """PNDM (pseudo numerical methods) scheduler with 20 denoising steps.

    Classic HF default scheduler; skip_prk_steps left at default (False)
    per verified API notes (do not set True).

    References:
        - research.md §4.1 (additional scheduler baselines)
    """

    name = "pndm20"
    nfe = 20

    def __init__(self, device: str = "cuda") -> None:
        self.device = device
        self.pipe = load_sd15_pipeline(device=device)
        self.pipe.scheduler = PNDMScheduler.from_config(self.pipe.scheduler.config)

    def generate(
        self,
        prompts: list[str],
        seeds: list[int],
        output_dir: str,
        **kwargs,
    ) -> list[BaselineResult]:
        """Generate one image per (prompt, seed) pair using PNDM-20."""
        results = []
        for idx, (prompt, seed) in enumerate(zip(prompts, seeds)):
            gen = torch.Generator(self.device).manual_seed(seed)
            t0 = time.time()
            image = self.pipe(
                prompt,
                num_inference_steps=self.nfe,
                generator=gen,
            ).images[0]
            results.append(
                BaselineResult(
                    image=image,
                    prompt=prompt,
                    seed=seed,
                    nfe=self.nfe,
                    time_s=time.time() - t0,
                    method_name=self.name,
                    metadata={},
                )
            )
        return save_results(results, output_dir)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ALL_FIXED_STEP_BASELINES = [
    DDIM20,
    DDIM50,
    DPMSolver20,
    DPMSolver50,
    Euler20,
    UniPC20,
    PNDM20,
]
