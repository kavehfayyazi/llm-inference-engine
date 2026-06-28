"""Owned causal self-attention path, with optional KV cache."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from engine.model import ArchDims


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    # [a, b] -> [-b, a] over the last dim.
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q, k, cos, sin):
    # Rotate q and k by position; unsqueeze head dim to broadcast.
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    # Tile each KV head n_rep times to match query head count.
    if n_rep == 1:
        return x
    b, n_kv, t, d = x.shape
    x = x[:, :, None, :, :].expand(b, n_kv, n_rep, t, d)
    return x.reshape(b, n_kv * n_rep, t, d)


def attention(hidden, attn_module, cos, sin, dims: ArchDims, past_kv=None):
    # Self-attention for one layer; returns (out, (k_full, v_full)) to cache.
    b, t, _ = hidden.shape

    # Project and split into heads.
    q = attn_module.q_proj(hidden).view(b, t, dims.n_q_heads, dims.head_dim).transpose(1, 2)
    k = attn_module.k_proj(hidden).view(b, t, dims.n_kv_heads, dims.head_dim).transpose(1, 2)
    v = attn_module.v_proj(hidden).view(b, t, dims.n_kv_heads, dims.head_dim).transpose(1, 2)

    # Rope the new tokens at their positions (cos/sin already match start_pos).
    q, k = apply_rope(q, k, cos, sin)

    # Prepend cached k/v (already roped when stored). New k/v become the cache.
    if past_kv is not None:
        past_k, past_v = past_kv
        k = torch.cat([past_k, k], dim=2)
        v = torch.cat([past_v, v], dim=2)
    new_kv = (k, v)

    # Expand KV heads to query-head count (GQA).
    k = repeat_kv(k, dims.n_rep)
    v = repeat_kv(v, dims.n_rep)

    # Causal only on prefill (no past). Decode: 1 query attends all cached.
    is_causal = past_kv is None
    out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)

    out = out.transpose(1, 2).reshape(b, t, dims.n_q_heads * dims.head_dim)
    return attn_module.o_proj(out), new_kv
