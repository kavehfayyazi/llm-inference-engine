"""Triton paged decode kernel: flash-style online softmax over KV blocks."""

from __future__ import annotations

import math

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # Mac / no CUDA: module still imports, kernel undefined.
    HAS_TRITON = False


if HAS_TRITON:

    @triton.jit
    def _paged_decode_kernel(
        q_ptr, k_ptr, v_ptr, bt_ptr, out_ptr,
        seq_len, scale, n_rep,
        stride_kb, stride_kh, stride_ks, stride_kd,
        stride_vb, stride_vh, stride_vs, stride_vd,
        BLOCK_SIZE: tl.constexpr, HEAD_DIM: tl.constexpr,
    ):
        # One program per query head; single new token (batch 1).
        h = tl.program_id(0)
        kv_h = h // n_rep  # GQA: query head -> its KV head

        offs_d = tl.arange(0, HEAD_DIM)
        offs_s = tl.arange(0, BLOCK_SIZE)
        q = tl.load(q_ptr + h * HEAD_DIM + offs_d).to(tl.float32) * scale  # [HEAD_DIM]

        m_i = -float("inf")
        l_i = 0.0
        acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

        n_logical = tl.cdiv(seq_len, BLOCK_SIZE)
        for lb in range(0, n_logical):
            phys = tl.load(bt_ptr + lb)
            base_k = phys * stride_kb + kv_h * stride_kh
            base_v = phys * stride_vb + kv_h * stride_vh
            k = tl.load(k_ptr + base_k + offs_s[:, None] * stride_ks + offs_d[None, :] * stride_kd).to(tl.float32)
            v = tl.load(v_ptr + base_v + offs_s[:, None] * stride_vs + offs_d[None, :] * stride_vd).to(tl.float32)

            tok = lb * BLOCK_SIZE + offs_s
            valid = tok < seq_len
            scores = tl.sum(q[None, :] * k, axis=1)            # [BLOCK_SIZE]
            scores = tl.where(valid, scores, -float("inf"))

            m_new = tl.maximum(m_i, tl.max(scores, axis=0))
            p = tl.exp(scores - m_new)                          # [BLOCK_SIZE]
            corr = tl.exp(m_i - m_new)
            l_i = l_i * corr + tl.sum(p, axis=0)
            acc = acc * corr + tl.sum(p[:, None] * v, axis=0)   # [HEAD_DIM]
            m_i = m_new

        out = acc / l_i
        tl.store(out_ptr + h * HEAD_DIM + offs_d, out.to(out_ptr.dtype.element_ty))

    @triton.jit
    def _paged_decode_batched_kernel(
        q_ptr, k_ptr, v_ptr, bt_ptr, seqlen_ptr, out_ptr,
        scale, n_rep, n_q_heads, max_blocks,
        stride_kb, stride_kh, stride_ks, stride_kd,
        stride_vb, stride_vh, stride_vs, stride_vd,
        BLOCK_SIZE: tl.constexpr, HEAD_DIM: tl.constexpr,
    ):
        # One program per (request, query head); each request has its own length.
        b = tl.program_id(0)
        h = tl.program_id(1)
        kv_h = h // n_rep
        seq_len = tl.load(seqlen_ptr + b)

        offs_d = tl.arange(0, HEAD_DIM)
        offs_s = tl.arange(0, BLOCK_SIZE)
        q_base = (b * n_q_heads + h) * HEAD_DIM
        q = tl.load(q_ptr + q_base + offs_d).to(tl.float32) * scale

        m_i = -float("inf")
        l_i = 0.0
        acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

        n_logical = tl.cdiv(seq_len, BLOCK_SIZE)
        for lb in range(0, n_logical):
            phys = tl.load(bt_ptr + b * max_blocks + lb)   # this request's block table row
            base_k = phys * stride_kb + kv_h * stride_kh
            base_v = phys * stride_vb + kv_h * stride_vh
            k = tl.load(k_ptr + base_k + offs_s[:, None] * stride_ks + offs_d[None, :] * stride_kd).to(tl.float32)
            v = tl.load(v_ptr + base_v + offs_s[:, None] * stride_vs + offs_d[None, :] * stride_vd).to(tl.float32)

            tok = lb * BLOCK_SIZE + offs_s
            valid = tok < seq_len
            scores = tl.sum(q[None, :] * k, axis=1)
            scores = tl.where(valid, scores, -float("inf"))

            m_new = tl.maximum(m_i, tl.max(scores, axis=0))
            p = tl.exp(scores - m_new)
            corr = tl.exp(m_i - m_new)
            l_i = l_i * corr + tl.sum(p, axis=0)
            acc = acc * corr + tl.sum(p[:, None] * v, axis=0)
            m_i = m_new

        out = acc / l_i
        tl.store(out_ptr + q_base + offs_d, out.to(out_ptr.dtype.element_ty))


def paged_decode(q, k_pool_layer, v_pool_layer, block_table, seq_len, n_rep):
    # q: [n_q_heads, head_dim] (single token). Returns [n_q_heads, head_dim].
    if not HAS_TRITON:
        raise RuntimeError("triton backend requested but triton is unavailable (need CUDA)")
    n_q_heads, head_dim = q.shape
    block_size = k_pool_layer.shape[2]
    q = q.contiguous()
    bt = torch.tensor(block_table, dtype=torch.int32, device=q.device)
    out = torch.empty_like(q)
    scale = 1.0 / math.sqrt(head_dim)

    _paged_decode_kernel[(n_q_heads,)](
        q, k_pool_layer, v_pool_layer, bt, out,
        seq_len, scale, n_rep,
        *k_pool_layer.stride(), *v_pool_layer.stride(),
        BLOCK_SIZE=block_size, HEAD_DIM=head_dim,
    )
    return out


def paged_decode_batched(q, k_pool_layer, v_pool_layer, block_tables, seq_lens, n_rep):
    # q: [N, n_q_heads, head_dim]. One fused launch for all N requests.
    if not HAS_TRITON:
        raise RuntimeError("triton backend requested but triton is unavailable (need CUDA)")
    n, n_q_heads, head_dim = q.shape
    block_size = k_pool_layer.shape[2]
    max_blocks = max(len(bt) for bt in block_tables)

    bt = torch.zeros((n, max_blocks), dtype=torch.int32, device=q.device)
    for i, table in enumerate(block_tables):
        bt[i, :len(table)] = torch.tensor(table, dtype=torch.int32, device=q.device)
    seqlens = torch.tensor(seq_lens, dtype=torch.int32, device=q.device)

    q = q.contiguous()
    out = torch.empty_like(q)
    scale = 1.0 / math.sqrt(head_dim)

    _paged_decode_batched_kernel[(n, n_q_heads)](
        q, k_pool_layer, v_pool_layer, bt, seqlens, out,
        scale, n_rep, n_q_heads, max_blocks,
        *k_pool_layer.stride(), *v_pool_layer.stride(),
        BLOCK_SIZE=block_size, HEAD_DIM=head_dim,
    )
    return out
