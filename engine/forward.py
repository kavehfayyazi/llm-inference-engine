"""Owned forward pass: embed -> layers -> norm -> lm_head, optional KV cache."""

from __future__ import annotations

import torch

from engine.attention import attention
from engine.blocks import BlockPool, PagedKVCache
from engine.cache import KVCache
from engine.model import LoadedModel
from engine.paged_attention import paged_attention


@torch.no_grad()
def forward_logits(lm: LoadedModel, input_ids: torch.Tensor, cache: KVCache = None, start_pos: int = 0):
    # input_ids [B,T] -> logits [B,T,vocab]. With cache: appends k/v, starts at start_pos.
    base = lm.model.model
    dims = lm.dims
    b, t = input_ids.shape

    hidden = base.embed_tokens(input_ids)

    # Positions start_pos..start_pos+T-1; cos/sin reused by every layer.
    position_ids = torch.arange(start_pos, start_pos + t, device=hidden.device).unsqueeze(0).expand(b, -1)
    cos, sin = base.rotary_emb(hidden, position_ids)

    for i, layer in enumerate(base.layers):
        # h = x + Attn(norm(x)).
        residual = hidden
        h = layer.input_layernorm(hidden)
        past = cache.get(i) if cache is not None else None
        h, new_kv = attention(h, layer.self_attn, cos, sin, dims, past)
        if cache is not None:
            cache.set(i, *new_kv)
        hidden = residual + h

        # h = h + MLP(norm(h)).
        residual = hidden
        h = layer.post_attention_layernorm(hidden)
        h = layer.mlp(h)
        hidden = residual + h

    hidden = base.norm(hidden)
    return lm.model.lm_head(hidden)


@torch.no_grad()
def forward_paged(lm: LoadedModel, input_ids: torch.Tensor, pool: BlockPool, req: PagedKVCache, start_pos: int = 0):
    # Like forward_logits but K/V live in paged blocks (reference read path).
    base = lm.model.model
    dims = lm.dims
    b, t = input_ids.shape

    req.reserve(start_pos + t)  # blocks for all tokens so far; shared across layers

    hidden = base.embed_tokens(input_ids)
    position_ids = torch.arange(start_pos, start_pos + t, device=hidden.device).unsqueeze(0).expand(b, -1)
    cos, sin = base.rotary_emb(hidden, position_ids)

    for i, layer in enumerate(base.layers):
        residual = hidden
        h = layer.input_layernorm(hidden)
        h = paged_attention(h, layer.self_attn, cos, sin, dims, pool, req, i, start_pos, lm.cfg.attention_backend)
        hidden = residual + h

        residual = hidden
        h = layer.post_attention_layernorm(hidden)
        h = layer.mlp(h)
        hidden = residual + h

    hidden = base.norm(hidden)
    return lm.model.lm_head(hidden)
