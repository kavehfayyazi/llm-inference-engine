"""Reference paged attention: write k/v into blocks, gather them, run SDPA."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from engine.attention import apply_rope, repeat_kv
from engine.blocks import BlockPool, PagedKVCache
from engine.model import ArchDims


def _write(pool: BlockPool, req: PagedKVCache, layer: int, k_new, v_new, start_pos: int):
    # Place each new token's k/v at its (block, offset).
    t = k_new.shape[2]
    for j in range(t):
        block_id, off = req.locate(start_pos + j)
        pool.k[layer][block_id, :, off, :] = k_new[0, :, j, :]
        pool.v[layer][block_id, :, off, :] = v_new[0, :, j, :]


def _gather(pool: BlockPool, req: PagedKVCache, layer: int, total: int):
    # Pull this request's blocks back into one ordered [1, n_kv, total, head_dim].
    bs = pool.block_size
    n_kv, _, head_dim = pool.k[layer].shape[1], bs, pool.k[layer].shape[3]
    table = torch.tensor(req.block_table, device=pool.k[layer].device)
    k = pool.k[layer][table].permute(1, 0, 2, 3).reshape(n_kv, -1, head_dim)[:, :total]
    v = pool.v[layer][table].permute(1, 0, 2, 3).reshape(n_kv, -1, head_dim)[:, :total]
    return k.unsqueeze(0), v.unsqueeze(0)


def paged_attention_ref(hidden, attn_module, cos, sin, dims: ArchDims, pool, req, layer, start_pos):
    # One layer of paged self-attention via gather + SDPA.
    b, t, _ = hidden.shape

    q = attn_module.q_proj(hidden).view(b, t, dims.n_q_heads, dims.head_dim).transpose(1, 2)
    k = attn_module.k_proj(hidden).view(b, t, dims.n_kv_heads, dims.head_dim).transpose(1, 2)
    v = attn_module.v_proj(hidden).view(b, t, dims.n_kv_heads, dims.head_dim).transpose(1, 2)

    q, k = apply_rope(q, k, cos, sin)

    _write(pool, req, layer, k, v, start_pos)
    total = start_pos + t
    k_full, v_full = _gather(pool, req, layer, total)

    k_full = repeat_kv(k_full, dims.n_rep)
    v_full = repeat_kv(v_full, dims.n_rep)

    is_causal = start_pos == 0
    out = F.scaled_dot_product_attention(q, k_full, v_full, is_causal=is_causal)

    out = out.transpose(1, 2).reshape(b, t, dims.n_q_heads * dims.head_dim)
    return attn_module.o_proj(out)
