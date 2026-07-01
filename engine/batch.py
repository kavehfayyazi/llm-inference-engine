"""Batched paged execution: prefill a request, decode all running together."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from engine.attention import apply_rope, repeat_kv
from engine.blocks import BlockPool
from engine.forward import forward_paged
from engine.model import LoadedModel
from engine.paged_attention import _gather
from engine.request import Request, State
from engine.triton_paged_attention import paged_decode_batched


@torch.no_grad()
def prefill(lm: LoadedModel, pool: BlockPool, req: Request):
    # Run the prompt through the paged forward, emit the first token.
    logits = forward_paged(lm, req.prompt_ids, pool, req.kv, start_pos=0)
    req.pos = req.prompt_ids.shape[1]
    req.last_token = logits[0, -1, :].argmax().item()
    req.generated.append(req.last_token)
    req.state = State.RUNNING


@torch.no_grad()
def decode_step(lm: LoadedModel, pool: BlockPool, running: list):
    # One decode token for every running request. Dense ops batched; attention
    # read/write per-request (ragged lengths, no padding).
    base = lm.model.model
    dims = lm.dims
    dev = lm.device
    n = len(running)

    tokens = torch.tensor([[r.last_token] for r in running], device=dev)     # [N, 1]
    positions = torch.tensor([[r.pos] for r in running], device=dev)         # [N, 1]
    for r in running:
        r.kv.reserve(r.pos + 1)

    hidden = base.embed_tokens(tokens)
    cos, sin = base.rotary_emb(hidden, positions)

    for i, layer in enumerate(base.layers):
        residual = hidden
        h = layer.input_layernorm(hidden)

        q = layer.self_attn.q_proj(h).view(n, 1, dims.n_q_heads, dims.head_dim).transpose(1, 2)
        k = layer.self_attn.k_proj(h).view(n, 1, dims.n_kv_heads, dims.head_dim).transpose(1, 2)
        v = layer.self_attn.v_proj(h).view(n, 1, dims.n_kv_heads, dims.head_dim).transpose(1, 2)
        q, k = apply_rope(q, k, cos, sin)

        # Write each new token's k/v into its own blocks.
        for j, r in enumerate(running):
            block_id, off = r.kv.locate(r.pos)
            pool.k[i][block_id, :, off, :] = k[j, :, 0, :]
            pool.v[i][block_id, :, off, :] = v[j, :, 0, :]

        # Read: each query attends only its own request's KV (ragged lengths).
        if lm.cfg.attention_backend == "triton":
            out_heads = paged_decode_batched(
                q[:, :, 0, :], pool.k[i], pool.v[i],
                [r.kv.block_table for r in running], [r.pos + 1 for r in running], dims.n_rep,
            )
            out = out_heads.view(n, dims.n_q_heads, 1, dims.head_dim)
        else:
            outs = []
            for j, r in enumerate(running):
                k_full, v_full = _gather(pool, r.kv, i, r.pos + 1)
                k_full = repeat_kv(k_full, dims.n_rep)
                v_full = repeat_kv(v_full, dims.n_rep)
                outs.append(F.scaled_dot_product_attention(q[j:j + 1], k_full, v_full, is_causal=False))
            out = torch.cat(outs, dim=0)
        out = out.transpose(1, 2).reshape(n, 1, dims.n_q_heads * dims.head_dim)

        hidden = residual + layer.self_attn.o_proj(out)
        residual = hidden
        h = layer.post_attention_layernorm(hidden)
        hidden = residual + layer.mlp(h)

    logits = lm.model.lm_head(base.norm(hidden))       # [N, 1, vocab]
    nxt = logits[:, -1, :].argmax(dim=-1)              # [N]
    for j, r in enumerate(running):
        r.pos += 1
        r.last_token = nxt[j].item()
        r.generated.append(r.last_token)
