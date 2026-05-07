"""Single-file standalone diagnostic for the AdDiffusion refine action.

This file intentionally does NOT import the repository's ``src`` package. It is
portable to a server that has the project dependencies installed.

What it checks:
1. Loads a Stable Diffusion pipeline.
2. Runs a few normal DDIM denoising steps.
3. Runs one observed denoise step without advancing the scheduler.
4. Reuses that observed prediction for one RegionRefine action.
5. Verifies NFE/state/history invariants and optional deterministic repeat.
6. Saves pre-refine and post-refine images plus a JSON summary.

Notebook example:
    summaries = main(
        prompt="five colorful balloons floating in a blue sky",
        seed=42,
        pre_steps=8,
        num_steps=20,
        repeat_determinism=True,
    )
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler, StableDiffusionPipeline
from PIL import Image


@dataclass
class StepOutput:
    z_t: torch.Tensor
    z0_pred: torch.Tensor
    timestep: int
    step_index: int
    nfe: int


@dataclass
class PipelineState:
    z_t: torch.Tensor
    timesteps: torch.Tensor
    step_index: int = 0
    total_nfe: int = 0
    prompt_embeds: Optional[torch.Tensor] = None
    negative_prompt_embeds: Optional[torch.Tensor] = None
    text_input_ids: Optional[torch.Tensor] = None
    is_done: bool = False
    history: list[StepOutput] = field(default_factory=list)


class StandaloneDiffusionPipeline:
    """Small step-by-step SD wrapper containing only what this diagnostic needs."""

    def __init__(
        self,
        model_id: str,
        device: str,
        dtype: torch.dtype,
        scheduler_name: str = "ddim",
    ):
        if scheduler_name != "ddim":
            raise ValueError("This standalone check currently supports --scheduler ddim only.")

        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            safety_checker=None,
        ).to(device)
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.scheduler = self.pipe.scheduler
        self.unet = self.pipe.unet
        self.vae = self.pipe.vae
        self.tokenizer = self.pipe.tokenizer
        self.text_encoder = self.pipe.text_encoder
        self.device = device
        self.dtype = dtype
        self.vae_scale_factor = self.vae.config.scaling_factor
        self.latent_channels = self.unet.config.in_channels
        self.downsample_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

    def get_latent_dims(self, height: int = 512, width: int = 512) -> tuple[int, int]:
        return height // self.downsample_factor, width // self.downsample_factor

    @torch.no_grad()
    def encode_prompt(self, prompt: str, negative_prompt: str = ""):
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.device)
        prompt_embeds = self.text_encoder(text_input_ids)[0]

        uncond_inputs = self.tokenizer(
            negative_prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        negative_prompt_embeds = self.text_encoder(
            uncond_inputs.input_ids.to(self.device)
        )[0]
        return prompt_embeds, negative_prompt_embeds, text_input_ids

    @torch.no_grad()
    def prepare(
        self,
        prompt: str,
        num_steps: int,
        seed: int,
        guidance_scale: float,
        height: int = 512,
        width: int = 512,
        negative_prompt: str = "",
    ) -> PipelineState:
        prompt_embeds, negative_prompt_embeds, text_input_ids = self.encode_prompt(
            prompt,
            negative_prompt=negative_prompt,
        )
        self.scheduler.set_timesteps(num_steps, device=self.device)
        h, w = self.get_latent_dims(height, width)

        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)
        z_t = torch.randn(
            (1, self.latent_channels, h, w),
            generator=generator,
            device=self.device,
            dtype=self.dtype,
        )
        z_t = z_t * self.scheduler.init_noise_sigma

        return PipelineState(
            z_t=z_t,
            timesteps=self.scheduler.timesteps,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            text_input_ids=text_input_ids,
        )

    @torch.no_grad()
    def denoise_step(self, state: PipelineState, guidance_scale: float) -> StepOutput:
        if state.is_done:
            raise RuntimeError("Episode is already done")
        if state.step_index >= len(state.timesteps):
            raise RuntimeError("No more timesteps in schedule")

        t = state.timesteps[state.step_index]
        latent_model_input = torch.cat([state.z_t, state.z_t])
        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
        combined_embeds = torch.cat([state.negative_prompt_embeds, state.prompt_embeds])

        noise_pred = self.unet(
            latent_model_input,
            t,
            encoder_hidden_states=combined_embeds,
        ).sample
        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_pred_cond - noise_pred_uncond
        )

        scheduler_output = self.scheduler.step(noise_pred, t, state.z_t)
        z_next = scheduler_output.prev_sample
        if (
            hasattr(scheduler_output, "pred_original_sample")
            and scheduler_output.pred_original_sample is not None
        ):
            z0_pred = scheduler_output.pred_original_sample
        else:
            alpha_prod_t = self.scheduler.alphas_cumprod[t.cpu().long()].to(
                device=state.z_t.device,
                dtype=self.dtype,
            )
            z0_pred = (state.z_t - (1 - alpha_prod_t).sqrt() * noise_pred) / alpha_prod_t.sqrt()

        return StepOutput(
            z_t=z_next,
            z0_pred=z0_pred,
            timestep=int(t.item()),
            step_index=state.step_index,
            nfe=1,
        )

    def advance_state(self, state: PipelineState, step_output: StepOutput) -> None:
        state.z_t = step_output.z_t
        state.step_index += 1
        state.total_nfe += step_output.nfe
        state.history.append(step_output)
        if state.step_index >= len(state.timesteps):
            state.is_done = True

    @torch.no_grad()
    def decode(self, z0: torch.Tensor) -> torch.Tensor:
        image = self.vae.decode(z0 / self.vae_scale_factor).sample
        return (image / 2 + 0.5).clamp(0, 1)

    @torch.no_grad()
    def add_noise(
        self,
        z0: torch.Tensor,
        timestep: int,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        noise = torch.randn(z0.shape, dtype=z0.dtype, device=z0.device, generator=generator)
        t_tensor = torch.tensor([timestep], device=self.device, dtype=torch.long)
        return self.scheduler.add_noise(z0, noise, t_tensor).to(dtype=self.dtype)

    @torch.no_grad()
    def unet_forward(
        self,
        z_t: torch.Tensor,
        t: int | torch.Tensor,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: torch.Tensor,
        guidance_scale: float,
    ) -> torch.Tensor:
        if isinstance(t, int):
            t = torch.tensor([t], device=self.device, dtype=torch.long)

        latent_model_input = torch.cat([z_t, z_t])
        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
        combined_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        noise_pred = self.unet(
            latent_model_input,
            t,
            encoder_hidden_states=combined_embeds,
        ).sample
        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
        return noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

    def predict_clean(
        self,
        z_t: torch.Tensor,
        noise_pred: torch.Tensor,
        t: int | torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(t, int):
            t = torch.tensor([t], device=self.device, dtype=torch.long)
        alpha_prod_t = self.scheduler.alphas_cumprod[t.cpu().long()].to(
            device=z_t.device,
            dtype=self.dtype,
        )
        while alpha_prod_t.dim() < z_t.dim():
            alpha_prod_t = alpha_prod_t.unsqueeze(-1)
        return (z_t - (1 - alpha_prod_t).sqrt() * noise_pred) / alpha_prod_t.sqrt()


class UNetAttentionExtractor:
    """Stores conditional cross-attention maps from diffusers UNet attention processors."""

    def __init__(self, latent_h: int = 64, latent_w: int = 64):
        self.latent_h = latent_h
        self.latent_w = latent_w
        self._attention_maps: list[torch.Tensor] = []
        self._original_processors = None

    def hook_with_processor(self, unet: torch.nn.Module) -> None:
        self.clear()
        self._original_processors = dict(unet.attn_processors)
        attn_procs = {}
        for name, module in unet.attn_processors.items():
            if "attn2" in name:
                attn_procs[name] = StoringAttnProcessor(self)
            else:
                attn_procs[name] = module
        unet.set_attn_processor(attn_procs)
        self._unet = unet

    def store_map(self, attn_map: torch.Tensor) -> None:
        self._attention_maps.append(attn_map)

    def get_attention_maps(self) -> torch.Tensor:
        if not self._attention_maps:
            raise RuntimeError("No attention maps stored. Run a UNet forward pass first.")

        aggregated = []
        for attn_map in self._attention_maps:
            attn_avg = attn_map.mean(dim=0)
            spatial = attn_avg.shape[0]
            seq_len = attn_avg.shape[1]
            side = int(spatial ** 0.5)
            if side * side != spatial:
                continue
            attn_2d = attn_avg.permute(1, 0).reshape(1, seq_len, side, side).float()
            attn_resized = F.interpolate(
                attn_2d,
                size=(self.latent_h, self.latent_w),
                mode="bilinear",
                align_corners=False,
            )
            aggregated.append(attn_resized.squeeze(0))

        if not aggregated:
            raise RuntimeError("No square spatial attention maps available.")
        combined = torch.stack(aggregated).mean(dim=0)
        return combined.permute(1, 2, 0)

    def clear(self) -> None:
        self._attention_maps.clear()

    def remove_hooks(self) -> None:
        if self._original_processors is not None and hasattr(self, "_unet"):
            self._unet.set_attn_processor(self._original_processors)
        self._original_processors = None
        self.clear()


class StoringAttnProcessor:
    """Minimal diffusers attention processor that stores cross-attention probs."""

    def __init__(self, extractor: UNetAttentionExtractor):
        self.extractor = extractor

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size = hidden_states.shape[0]
        sequence_length = (
            hidden_states.shape[1] if encoder_hidden_states is None else encoder_hidden_states.shape[1]
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        is_cross = encoder_hidden_states is not None
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        if is_cross:
            stored_probs = attention_probs.detach()
            if batch_size >= 2 and batch_size % 2 == 0:
                heads = stored_probs.shape[0] // batch_size
                stored_probs = stored_probs.view(batch_size, heads, stored_probs.shape[1], stored_probs.shape[2])
                stored_probs = stored_probs[batch_size // 2:].reshape(
                    -1,
                    stored_probs.shape[2],
                    stored_probs.shape[3],
                )
            self.extractor.store_map(stored_probs)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual
        return hidden_states / attn.rescale_output_factor


def filter_prompt_attention_maps(
    attention_maps: torch.Tensor,
    text_input_ids: Optional[torch.Tensor],
    tokenizer,
) -> torch.Tensor:
    if text_input_ids is None or tokenizer is None:
        return attention_maps

    token_ids = text_input_ids.reshape(-1).to(device=attention_maps.device)
    seq_len = attention_maps.shape[-1]
    if token_ids.numel() < seq_len:
        return attention_maps
    token_ids = token_ids[:seq_len]

    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is not None:
        special_ids.add(pad_id)
    if not special_ids:
        return attention_maps

    keep = torch.ones(seq_len, dtype=torch.bool, device=attention_maps.device)
    for special_id in special_ids:
        keep &= token_ids != int(special_id)

    if keep.any():
        return attention_maps[..., keep]
    return attention_maps


def compute_attention_entropy(attention_maps: torch.Tensor) -> float:
    h, w, _ = attention_maps.shape
    attention = attention_maps.sum(dim=-1).float()
    attention = attention / (attention.sum() + 1e-8)
    entropy = -(attention * (attention + 1e-8).log()).sum()
    return float((entropy / math.log(h * w)).clamp(0.0, 1.0))


def generate_refinement_mask(
    attention_maps: torch.Tensor,
    threshold: float,
    blur_sigma: float,
    blur_kernel_size: int = 15,
) -> torch.Tensor:
    max_attn = attention_maps.max(dim=-1).values
    attn_threshold = max_attn.quantile(threshold)
    hard_mask = (max_attn < attn_threshold).float()
    mask = hard_mask.unsqueeze(0).unsqueeze(0)
    mask = gaussian_blur(mask, kernel_size=blur_kernel_size, sigma=blur_sigma)
    return mask.clamp(0, 1)


def gaussian_blur(x: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
    coords = torch.arange(kernel_size, dtype=x.dtype, device=x.device) - kernel_size // 2
    gauss = torch.exp(-coords.pow(2) / (2 * sigma**2))
    gauss = gauss / gauss.sum()
    kernel_2d = gauss.outer(gauss).view(1, 1, kernel_size, kernel_size)
    return F.conv2d(x, kernel_2d, padding=kernel_size // 2)


def region_refine(
    pipeline: StandaloneDiffusionPipeline,
    state: PipelineState,
    mask: torch.Tensor,
    observed_step_out: StepOutput,
    k: int,
    r_noise: float,
    guidance_scale: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, int]:
    current_t = state.timesteps[state.step_index]
    t_prime = int(current_t.item() * r_noise)
    z0_hat = observed_step_out.z0_pred
    nfe = observed_step_out.nfe

    for _ in range(k):
        noise = torch.randn(z0_hat.shape, dtype=z0_hat.dtype, device=z0_hat.device, generator=generator)
        t_prime_tensor = torch.tensor([t_prime], device=pipeline.device, dtype=torch.long)
        alpha_prod_t = pipeline.scheduler.alphas_cumprod[t_prime_tensor.cpu().long()].to(
            device=z0_hat.device,
            dtype=pipeline.dtype,
        )
        while alpha_prod_t.dim() < z0_hat.dim():
            alpha_prod_t = alpha_prod_t.unsqueeze(-1)

        z_noised = alpha_prod_t.sqrt() * z0_hat + (1 - alpha_prod_t).sqrt() * noise
        z_composite = (mask * z_noised + (1 - mask) * z0_hat).to(dtype=pipeline.dtype)

        noise_pred = pipeline.unet_forward(
            z_composite,
            t_prime,
            state.prompt_embeds,
            state.negative_prompt_embeds,
            guidance_scale=guidance_scale,
        )
        z0_hat = pipeline.predict_clean(z_composite, noise_pred, t_prime)

    return z0_hat, nfe + k


def apply_refine_action(
    pipeline: StandaloneDiffusionPipeline,
    state: PipelineState,
    attention_extractor: UNetAttentionExtractor,
    observed_step_out: StepOutput,
    k: int,
    r_noise: float,
    mask_threshold: float,
    blur_sigma: float,
    guidance_scale: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, int, dict]:
    step_index = state.step_index
    current_t = int(state.timesteps[step_index].item())

    raw_attention = attention_extractor.get_attention_maps()
    filtered_attention = filter_prompt_attention_maps(raw_attention, state.text_input_ids, pipeline.tokenizer)
    entropy = compute_attention_entropy(filtered_attention)
    mask = generate_refinement_mask(filtered_attention, threshold=mask_threshold, blur_sigma=blur_sigma)

    z0_refined, nfe = region_refine(
        pipeline=pipeline,
        state=state,
        mask=mask,
        observed_step_out=observed_step_out,
        k=k,
        r_noise=r_noise,
        guidance_scale=guidance_scale,
        generator=generator,
    )

    if state.step_index + 1 < len(state.timesteps):
        next_t = int(state.timesteps[state.step_index + 1].item())
        state.z_t = pipeline.add_noise(z0_refined, next_t, generator=generator)
    else:
        state.z_t = z0_refined

    state.step_index += 1
    state.total_nfe += nfe
    state.history.append(
        StepOutput(
            z_t=state.z_t,
            z0_pred=z0_refined,
            timestep=current_t,
            step_index=step_index,
            nfe=nfe,
        )
    )
    if state.step_index >= len(state.timesteps):
        state.is_done = True

    diagnostics = {
        "raw_token_channels": raw_attention.shape[-1],
        "filtered_token_channels": filtered_attention.shape[-1],
        "attention_entropy": entropy,
        "mask_mean": mask.mean().item(),
        "mask_min": mask.min().item(),
        "mask_max": mask.max().item(),
    }
    return z0_refined, nfe, diagnostics


def tensor_to_png(image: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = image[0].detach().cpu().float().clamp(0, 1)
    array = (array.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    Image.fromarray(array).save(path)


def configure_reproducibility(seed: int, deterministic_algorithms: bool = False) -> None:
    """Make the diagnostic as repeatable as the active PyTorch/CUDA stack allows."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if deterministic_algorithms:
        torch.use_deterministic_algorithms(True, warn_only=True)


@torch.no_grad()
def run_refine_check(
    *,
    pipeline: StandaloneDiffusionPipeline,
    run_name: str,
    prompt: str,
    seed: int,
    num_steps: int,
    pre_steps: int,
    guidance_scale: float,
    k: int,
    r_noise: float,
    mask_threshold: float,
    blur_sigma: float,
    latent_h: int,
    latent_w: int,
    output_dir: str,
) -> dict:
    extractor = UNetAttentionExtractor(latent_h=latent_h, latent_w=latent_w)
    extractor.hook_with_processor(pipeline.unet)

    try:
        state = pipeline.prepare(
            prompt,
            num_steps=num_steps,
            seed=seed,
            guidance_scale=guidance_scale,
        )

        for _ in range(pre_steps):
            extractor.clear()
            step_out = pipeline.denoise_step(state, guidance_scale=guidance_scale)
            pipeline.advance_state(state, step_out)

        step_before = state.step_index
        nfe_before = state.total_nfe
        z_t_before = state.z_t.clone()

        extractor.clear()
        observed_step = pipeline.denoise_step(state, guidance_scale=guidance_scale)
        pre_refine_image = pipeline.decode(observed_step.z0_pred)

        generator = torch.Generator(device=pipeline.device)
        generator.manual_seed(seed + 1_000_003)

        z0_refined, refine_nfe, diagnostics = apply_refine_action(
            pipeline=pipeline,
            state=state,
            attention_extractor=extractor,
            observed_step_out=observed_step,
            k=k,
            r_noise=r_noise,
            mask_threshold=mask_threshold,
            blur_sigma=blur_sigma,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        post_refine_image = pipeline.decode(z0_refined)

        expected_nfe = observed_step.nfe + k
        assert refine_nfe == expected_nfe, f"refine_nfe={refine_nfe}, expected={expected_nfe}"
        assert state.step_index == step_before + 1, (
            f"state.step_index={state.step_index}, expected={step_before + 1}"
        )
        assert state.total_nfe == nfe_before + refine_nfe, (
            f"state.total_nfe={state.total_nfe}, expected={nfe_before + refine_nfe}"
        )
        assert state.history, "Refine action should append a history entry"
        assert state.history[-1].z0_pred.shape == z0_refined.shape

        z_t_delta = (state.z_t - z_t_before).abs().mean().item()
        image_delta = (post_refine_image - pre_refine_image).abs().mean().item()

        continued = False
        if not state.is_done:
            extractor.clear()
            next_step = pipeline.denoise_step(state, guidance_scale=guidance_scale)
            pipeline.advance_state(state, next_step)
            continued = True

        output_path = Path(output_dir)
        tensor_to_png(pre_refine_image, output_path / f"{run_name}_pre_refine.png")
        tensor_to_png(post_refine_image, output_path / f"{run_name}_post_refine.png")

        return {
            "run": run_name,
            "prompt": prompt,
            "seed": seed,
            "step_before_refine": step_before,
            "nfe_before_refine": nfe_before,
            "observed_step_nfe": observed_step.nfe,
            "refine_k": k,
            "refine_nfe": refine_nfe,
            "expected_refine_nfe": expected_nfe,
            "state_step_after_refine": state.step_index - (1 if continued else 0),
            "state_total_nfe_after_refine": state.total_nfe - (1 if continued else 0),
            "z_t_mean_abs_delta": z_t_delta,
            "image_mean_abs_delta": image_delta,
            "continued_after_refine": continued,
            "pre_refine_png": str(output_path / f"{run_name}_pre_refine.png"),
            "post_refine_png": str(output_path / f"{run_name}_post_refine.png"),
            **diagnostics,
        }
    finally:
        extractor.remove_hooks()


def main(
    prompt: str = "five colorful balloons floating in a blue sky",
    model_id: str = "stable-diffusion-v1-5/stable-diffusion-v1-5",
    scheduler: str = "ddim",
    seed: int = 42,
    num_steps: int = 20,
    pre_steps: int = 8,
    guidance_scale: float = 7.5,
    k: int = 2,
    r_noise: float = 0.5,
    mask_threshold: float = 0.5,
    blur_sigma: float = 3.0,
    latent_h: int = 64,
    latent_w: int = 64,
    output_dir: str = "outputs/refine_check",
    device: str | None = None,
    dtype: str = "float16",
    repeat_determinism: bool = True,
    strict_determinism: bool = False,
    determinism_max_pixel_tolerance: int = 0,
    determinism_mean_pixel_tolerance: float = 0.0,
    deterministic_algorithms: bool = False,
) -> list[dict]:
    """Notebook-friendly entrypoint.

    The refine logic assertions always stay strict. The repeated-image comparison is
    diagnostic by default because many GPU stacks are not bitwise deterministic
    through UNet attention and VAE decode. Set ``strict_determinism=True`` if your
    environment should produce identical PNGs.
    """
    if pre_steps >= num_steps:
        raise ValueError("pre_steps must be smaller than num_steps")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.float16 if dtype == "float16" else torch.float32
    configure_reproducibility(seed, deterministic_algorithms=deterministic_algorithms)

    print("=" * 72)
    print("Standalone AdDiffusion refine action check")
    print("=" * 72)
    print(f"device={device} dtype={torch_dtype} model={model_id}")
    print(f"prompt={prompt!r} seed={seed}")

    t0 = time.time()
    pipeline = StandaloneDiffusionPipeline(
        model_id=model_id,
        scheduler_name=scheduler,
        device=device,
        dtype=torch_dtype,
    )

    first = run_refine_check(
        pipeline=pipeline,
        run_name="run1",
        prompt=prompt,
        seed=seed,
        num_steps=num_steps,
        pre_steps=pre_steps,
        guidance_scale=guidance_scale,
        k=k,
        r_noise=r_noise,
        mask_threshold=mask_threshold,
        blur_sigma=blur_sigma,
        latent_h=latent_h,
        latent_w=latent_w,
        output_dir=output_dir,
    )
    summaries = [first]

    if repeat_determinism:
        second = run_refine_check(
            pipeline=pipeline,
            run_name="run2",
            prompt=prompt,
            seed=seed,
            num_steps=num_steps,
            pre_steps=pre_steps,
            guidance_scale=guidance_scale,
            k=k,
            r_noise=r_noise,
            mask_threshold=mask_threshold,
            blur_sigma=blur_sigma,
            latent_h=latent_h,
            latent_w=latent_w,
            output_dir=output_dir,
        )
        summaries.append(second)

        img1 = np.asarray(Image.open(first["post_refine_png"])).astype(np.int16)
        img2 = np.asarray(Image.open(second["post_refine_png"])).astype(np.int16)
        max_abs_pixel_diff = int(np.abs(img1 - img2).max())
        mean_abs_pixel_diff = float(np.abs(img1 - img2).mean())
        print(
            f"determinism: max_abs_pixel_diff={max_abs_pixel_diff} "
            f"mean_abs_pixel_diff={mean_abs_pixel_diff:.6f}"
        )
        first["determinism_max_abs_pixel_diff"] = max_abs_pixel_diff
        first["determinism_mean_abs_pixel_diff"] = mean_abs_pixel_diff
        second["determinism_max_abs_pixel_diff"] = max_abs_pixel_diff
        second["determinism_mean_abs_pixel_diff"] = mean_abs_pixel_diff

        determinism_ok = (
            max_abs_pixel_diff <= determinism_max_pixel_tolerance
            and mean_abs_pixel_diff <= determinism_mean_pixel_tolerance
        )
        if not determinism_ok:
            message = (
                "Determinism check exceeded tolerance: "
                f"max_abs_pixel_diff={max_abs_pixel_diff} "
                f"(tol={determinism_max_pixel_tolerance}), "
                f"mean_abs_pixel_diff={mean_abs_pixel_diff:.6f} "
                f"(tol={determinism_mean_pixel_tolerance:.6f})"
            )
            if strict_determinism:
                raise AssertionError(message)
            print(f"[WARN] {message}")
            print("[WARN] Refine invariants passed; only bitwise image repeatability differed.")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    summary_path = output_path / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summaries, f, indent=2)

    print("\nRefine check summary:")
    for key, value in first.items():
        print(f"  {key}: {value}")
    print(f"\n[PASS] Refine action invariants passed in {time.time() - t0:.1f}s")
    print(f"Summary: {summary_path}")
    return summaries


if __name__ == "__main__":
    main()
