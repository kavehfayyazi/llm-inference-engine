"""Per-request KV cache: stores K/V per layer, grows by concat."""

from __future__ import annotations

import torch


class KVCache:
    """Holds one (k, v) tensor pair per layer for a single request."""

    def __init__(self, n_layers: int):
        self.kv: list = [None] * n_layers

    def get(self, layer: int):
        # Past (k, v) for this layer, or None before prefill.
        return self.kv[layer]

    def set(self, layer: int, k: torch.Tensor, v: torch.Tensor):
        # Replace stored k/v with the full (past + new) tensors.
        self.kv[layer] = (k, v)

    def seq_len(self) -> int:
        # Cached token count (from layer 0's key length).
        first = self.kv[0]
        return 0 if first is None else first[0].shape[-2]
