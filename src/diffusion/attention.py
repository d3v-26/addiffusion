"""Cross-attention map extraction from diffusion models.

Pluggable interface (D-27): separate implementations for UNet (SD 1.5, SDXL)
and DiT (Flux.1) architectures.

References:
    - research.md §3.5 (mask generation from attention)
    - discovery.md D-07 (cross-attention as refinement signal)
    - discovery.md D-27 (UNet vs DiT extraction)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn.functional as F


class AttentionExtractor(ABC):
    """Base class for attention extraction from diffusion models."""

    @abstractmethod
    def hook(self, unet: torch.nn.Module) -> None:
        """Register forward hooks on attention layers."""

    @abstractmethod
    def get_attention_maps(self) -> torch.Tensor:
        """Return aggregated attention maps at latent resolution.

        Returns:
            Tensor of shape (h, w, L) where L is the number of text tokens.
        """

    @abstractmethod
    def clear(self) -> None:
        """Clear stored attention maps."""

    @abstractmethod
    def remove_hooks(self) -> None:
        """Remove all registered hooks."""


class UNetAttentionExtractor(AttentionExtractor):
    """Extract cross-attention maps from UNet-based models (SD 1.5, SDXL).

    Hooks into the cross-attention layers in mid-block and up-blocks.
    Attention maps from different resolutions are interpolated to the
    target latent resolution and averaged.
    """

    def __init__(self, latent_h: int = 64, latent_w: int = 64):
        self.latent_h = latent_h
        self.latent_w = latent_w
        self._attention_maps: list[torch.Tensor] = []
        self._hooks: list[torch.utils.hooks.RemovableHook] = []

    def hook(self, unet: torch.nn.Module) -> None:
        """Register hooks on all cross-attention Attention modules in the UNet."""
        self.clear()
        self.remove_hooks()

        for name, module in unet.named_modules():
            # diffusers stores cross-attention as Attention modules
            # We want cross-attention (not self-attention)
            if module.__class__.__name__ == "Attention" and hasattr(module, "to_k"):
                # Cross-attention has encoder_hidden_states input
                # We hook the attention processor
                handle = module.register_forward_hook(self._make_hook(name))
                self._hooks.append(handle)

    def _make_hook(self, name: str):
        def hook_fn(module, input, output):
            # The Attention module computes: attn_weights = softmax(Q @ K^T / sqrt(d))
            # We need to extract these weights. diffusers Attention stores them
            # if we use an appropriate processor.
            # Instead, we'll recompute from the module's internals during forward.
            pass

        return hook_fn

    def hook_with_processor(self, unet: torch.nn.Module) -> None:
        """Replace attention processors with ones that store attention maps.

        This is the recommended approach for diffusers >= 0.25.
        """
        self.clear()
        self.remove_hooks()

        from diffusers.models.attention_processor import Attention

        attn_procs = {}
        for name, module in unet.attn_processors.items():
            if "attn2" in name:  # attn2 = cross-attention in diffusers convention
                attn_procs[name] = StoringAttnProcessor(self, name)
            else:
                attn_procs[name] = module

        unet.set_attn_processor(attn_procs)

    def store_map(self, name: str, attn_map: torch.Tensor) -> None:
        """Called by StoringAttnProcessor to store an attention map.

        Args:
            name: Layer name.
            attn_map: Shape (batch*heads, spatial, seq_len).
        """
        self._attention_maps.append(attn_map)

    def get_attention_maps(self) -> torch.Tensor:
        """Aggregate stored attention maps to (h, w, L).

        Averages across layers and heads, interpolates to latent resolution.
        """
        if not self._attention_maps:
            raise RuntimeError("No attention maps stored. Run a UNet forward pass first.")

        aggregated = []
        for attn_map in self._attention_maps:
            # attn_map: (batch*heads, spatial, seq_len)
            # Average over batch and heads
            n_heads = attn_map.shape[0]
            attn_avg = attn_map.mean(dim=0)  # (spatial, seq_len)

            spatial = attn_avg.shape[0]
            seq_len = attn_avg.shape[1]
            side = int(spatial ** 0.5)

            # Reshape to (1, seq_len, side, side) for interpolation
            attn_2d = attn_avg.permute(1, 0).reshape(1, seq_len, side, side).float()

            # Interpolate to latent resolution
            attn_resized = F.interpolate(
                attn_2d,
                size=(self.latent_h, self.latent_w),
                mode="bilinear",
                align_corners=False,
            )  # (1, seq_len, latent_h, latent_w)

            aggregated.append(attn_resized.squeeze(0))  # (seq_len, h, w)

        # Average across all layers
        combined = torch.stack(aggregated).mean(dim=0)  # (seq_len, h, w)
        # Permute to (h, w, seq_len) as specified
        return combined.permute(1, 2, 0)  # (h, w, L)

    def clear(self) -> None:
        self._attention_maps.clear()

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


class StoringAttnProcessor:
    """Attention processor that stores cross-attention maps.

    Drop-in replacement for diffusers attention processors that captures
    the attention weights during the forward pass.
    """

    def __init__(self, extractor: UNetAttentionExtractor, name: str):
        self.extractor = extractor
        self.name = name

    def __call__(
        self,
        attn,  # Attention module
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        # Standard cross-attention computation
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
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

        # Store cross-attention maps (only for cross-attention, not self-attention)
        if is_cross:
            self.extractor.store_map(self.name, attention_probs.detach())

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # Linear projection
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)  # dropout

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class DiTAttentionExtractor(AttentionExtractor):
    """Placeholder for DiT-based models (Flux.1).

    DiT uses joint attention over image and text tokens (D-27).
    The image-to-text attention sub-matrix serves as the cross-attention equivalent.

    TODO: Implement when Flux.1 generalization is attempted (Phase 4).
    """

    def __init__(self, latent_h: int = 64, latent_w: int = 64):
        self.latent_h = latent_h
        self.latent_w = latent_w

    def hook(self, unet: torch.nn.Module) -> None:
        raise NotImplementedError("DiT attention extraction not yet implemented (Phase 4)")

    def get_attention_maps(self) -> torch.Tensor:
        raise NotImplementedError("DiT attention extraction not yet implemented (Phase 4)")

    def clear(self) -> None:
        pass

    def remove_hooks(self) -> None:
        pass


def create_attention_extractor(
    architecture: str = "unet",
    latent_h: int = 64,
    latent_w: int = 64,
) -> AttentionExtractor:
    """Factory for attention extractors.

    Args:
        architecture: "unet" for SD 1.5/SDXL, "dit" for Flux.1
        latent_h: Latent spatial height
        latent_w: Latent spatial width
    """
    if architecture == "unet":
        return UNetAttentionExtractor(latent_h, latent_w)
    elif architecture == "dit":
        return DiTAttentionExtractor(latent_h, latent_w)
    else:
        raise ValueError(f"Unknown architecture: {architecture}. Use 'unet' or 'dit'.")
