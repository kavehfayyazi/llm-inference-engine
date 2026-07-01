"""Paged attention: write k/v into blocks, then read via gather (ref) or kernel."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from engine.attention import apply_rope, repeat_kv
from engine.blocks import BlockPool, PagedKVCache
from engine.model import ArchDims
from engine.triton_paged_attention import paged_decode


def _mask(t: int, start_pos: int, device):
    # Attention mask for the current tokens vs all cached keys.
    #  t == 1 (decode): attend everything -> None.
    #  start_pos == 0 (full prefill): square causal -> handled by is_causal.
    #  chunk (start_pos > 0, t > 1): query i attends keys 0..start_pos+i.
    if t == 1 or start_pos == 0:
        return None
    total = start_pos + t
    row = torch.arange(t, device=device).unsqueeze(1)
    col = torch.arange(total, device=device).unsqueeze(0)
    return col <= (start_pos + row)


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
    n_kv, head_dim = pool.k[layer].shape[1], pool.k[layer].shape[3]
    table = torch.tensor(req.block_table, device=pool.k[layer].device)
    k = pool.k[layer][table].permute(1, 0, 2, 3).reshape(n_kv, -1, head_dim)[:, :total]
    v = pool.v[layer][table].permute(1, 0, 2, 3).reshape(n_kv, -1, head_dim)[:, :total]
    return k.unsqueeze(0), v.unsqueeze(0)


def paged_attention(hidden, attn_module, cos, sin, dims: ArchDims, pool, req, layer, start_pos, backend="reference"):
    # One layer of paged self-attention; decode may use the triton kernel.
    b, t, _ = hidden.shape

    q = attn_module.q_proj(hidden).view(b, t, dims.n_q_heads, dims.head_dim).transpose(1, 2)
    k = attn_module.k_proj(hidden).view(b, t, dims.n_kv_heads, dims.head_dim).transpose(1, 2)
    v = attn_module.v_proj(hidden).view(b, t, dims.n_kv_heads, dims.head_dim).transpose(1, 2)

    q, k = apply_rope(q, k, cos, sin)

    _write(pool, req, layer, k, v, start_pos)
    total = start_pos + t
    is_decode = t == 1 and start_pos > 0

    if backend == "triton" and is_decode:
        assert b == 1, "triton decode path is single-request for now"
        out_heads = paged_decode(q[0, :, 0, :], pool.k[layer], pool.v[layer], req.block_table, total, dims.n_rep)
        out = out_heads.view(1, dims.n_q_heads, 1, dims.head_dim)
    else:
        k_full, v_full = _gather(pool, req, layer, total)
        k_full = repeat_kv(k_full, dims.n_rep)
        v_full = repeat_kv(v_full, dims.n_rep)
        out = F.scaled_dot_product_attention(q, k_full, v_full, attn_mask=_mask(t, start_pos, hidden.device), is_causal=(t > 1 and start_pos == 0))

    out = out.transpose(1, 2).reshape(b, t, dims.n_q_heads * dims.head_dim)
    return attn_module.o_proj(out)
