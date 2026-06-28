"""Owned forward pass: embed -> layers -> norm -> lm_head, no cache."""

from __future__ import annotations

import torch

from engine.attention import attention
from engine.model import LoadedModel


@torch.no_grad()
def forward_logits(lm: LoadedModel, input_ids: torch.Tensor) -> torch.Tensor:
    # input_ids [B,T] -> logits [B,T,vocab]; runs the full sequence.
    base = lm.model.model
    dims = lm.dims
    b, t = input_ids.shape

    hidden = base.embed_tokens(input_ids)

    # Positions 0..T-1; cos/sin computed once, reused by every layer.
    position_ids = torch.arange(t, device=hidden.device).unsqueeze(0).expand(b, -1)
    cos, sin = base.rotary_emb(hidden, position_ids)

    for layer in base.layers:
        # h = x + Attn(norm(x)).
        residual = hidden
        h = layer.input_layernorm(hidden)
        h = attention(h, layer.self_attn, cos, sin, dims)
        hidden = residual + h

        # h = h + MLP(norm(h)).
        residual = hidden
        h = layer.post_attention_layernorm(hidden)
        h = layer.mlp(h)
        hidden = residual + h

    hidden = base.norm(hidden)
    return lm.model.lm_head(hidden)
