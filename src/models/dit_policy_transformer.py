#!/usr/bin/env python
# Adapted from:
# https://github.com/brysonjones/multitask_dit_policy
#
# Original copyright 2025 Bryson Jones.
# Licensed under the Apache License, Version 2.0.
#
# This local copy removes the third-party package/config dependencies and exposes
# the same backbone interface used by this repository's diffusion policies.

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1 + scale) + shift


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        head_dim: int,
        max_seq_len: int = 512,
        base: float = 10000.0,
    ):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE.")

        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2).float() / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._precompute_cache(max_seq_len)

    def _precompute_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            "_cos_cached", emb.cos()[None, None, :, :], persistent=False
        )
        self.register_buffer(
            "_sin_cached", emb.sin()[None, None, :, :], persistent=False
        )

    def _rotate_half(self, x: Tensor) -> Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, q: Tensor, k: Tensor) -> Tuple[Tensor, Tensor]:
        seq_len = q.shape[2]
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds RoPE max_seq_len "
                f"{self.max_seq_len}."
            )

        cos = self._cos_cached[:, :, :seq_len, :].to(dtype=q.dtype, device=q.device)
        sin = self._sin_cached[:, :, :seq_len, :].to(dtype=q.dtype, device=q.device)
        return (q * cos) + (self._rotate_half(q) * sin), (k * cos) + (
            self._rotate_half(k) * sin
        )


class RoPEAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.0,
        max_seq_len: int = 512,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads.")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.dropout = dropout
        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.rope = RotaryPositionalEmbedding(
            head_dim=self.head_dim,
            max_seq_len=max_seq_len,
            base=rope_base,
        )

    def forward(self, x: Tensor) -> Tensor:
        bsz, seq_len, _ = x.shape
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(bsz, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = self.rope(q, k)

        dropout_p = self.dropout if self.training else 0.0
        attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        attn_out = attn_out.transpose(1, 2).reshape(bsz, seq_len, self.hidden_size)
        return self.out_proj(attn_out)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_features: int,
        dropout: float = 0.0,
        use_rope: bool = False,
        max_seq_len: int = 512,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        self.use_rope = use_rope
        if use_rope:
            self.attn = RoPEAttention(
                hidden_size=hidden_size,
                num_heads=num_heads,
                dropout=dropout,
                max_seq_len=max_seq_len,
                rope_base=rope_base,
            )
        else:
            self.attn = nn.MultiheadAttention(
                hidden_size,
                num_heads=num_heads,
                batch_first=True,
                dropout=dropout,
            )

        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(num_features, 6 * hidden_size),
        )

    def forward(self, x: Tensor, features: Tensor) -> Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(features).chunk(6, dim=1)
        )

        attn_input = modulate(
            self.norm1(x), shift_msa.unsqueeze(1), scale_msa.unsqueeze(1)
        )
        if self.use_rope:
            attn_out = self.attn(attn_input)
        else:
            attn_out, _ = self.attn(attn_input, attn_input, attn_input)
        x = x + gate_msa.unsqueeze(1) * attn_out

        mlp_input = modulate(
            self.norm2(x), shift_mlp.unsqueeze(1), scale_mlp.unsqueeze(1)
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(mlp_input)
        return x


class DitPolicyTransformer(nn.Module):
    def __init__(
        self,
        action_dim: int,
        horizon: int,
        conditioning_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        diffusion_step_embed_dim: int = 256,
        use_positional_encoding: bool = True,
        use_rope: bool = False,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.horizon = horizon
        self.conditioning_dim = conditioning_dim
        self.hidden_size = hidden_dim
        self.timestep_embed_dim = diffusion_step_embed_dim
        self.cond_dim = self.timestep_embed_dim + self.conditioning_dim

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(self.timestep_embed_dim),
            nn.Linear(self.timestep_embed_dim, 2 * self.timestep_embed_dim),
            nn.GELU(),
            nn.Linear(2 * self.timestep_embed_dim, self.timestep_embed_dim),
            nn.GELU(),
        )
        self.input_proj = nn.Linear(self.action_dim, self.hidden_size)
        self.pos_embedding = (
            nn.Parameter(torch.empty(1, self.horizon, self.hidden_size).normal_(std=0.02))
            if use_positional_encoding
            else None
        )
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size=self.hidden_size,
                    num_heads=num_heads,
                    num_features=self.cond_dim,
                    dropout=dropout,
                    use_rope=use_rope,
                    max_seq_len=self.horizon,
                    rope_base=rope_base,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_proj = nn.Linear(self.hidden_size, self.action_dim)
        self._initialize_weights()

    def _initialize_weights(self):
        for block in self.transformer_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

    def forward(
        self,
        sample: Optional[Tensor] = None,
        timestep: Optional[Union[Tensor, float, int]] = None,
        global_cond: Optional[Tensor] = None,
        x: Optional[Tensor] = None,
        conditioning_vec: Optional[Tensor] = None,
    ) -> Tensor:
        if sample is None:
            sample = x
        if sample is None:
            raise ValueError("DitPolicyTransformer requires `sample` or `x`.")

        cond = global_cond if global_cond is not None else conditioning_vec
        if cond is None:
            if self.conditioning_dim != 0:
                raise ValueError("DitPolicyTransformer requires conditioning.")
            cond = sample.new_zeros((sample.shape[0], 0))
        if cond.dim() > 2:
            cond = cond.flatten(start_dim=1)

        if timestep is None:
            raise ValueError("DitPolicyTransformer requires `timestep`.")
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=sample.dtype, device=sample.device)
        elif timestep.dim() == 0:
            timestep = timestep[None].to(device=sample.device)
        timestep = timestep.expand(sample.shape[0]).to(device=sample.device)

        _, seq_len, _ = sample.shape
        timestep_features = self.time_mlp(timestep)
        cond = cond.to(device=sample.device, dtype=sample.dtype)
        cond_features = torch.cat([timestep_features, cond], dim=-1)

        hidden_seq = self.input_proj(sample)
        if self.pos_embedding is not None:
            hidden_seq = hidden_seq + self.pos_embedding[:, :seq_len, :]

        for block in self.transformer_blocks:
            hidden_seq = block(hidden_seq, cond_features)

        return self.output_proj(hidden_seq)
