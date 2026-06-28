"""Greedy decode loops: naive (full recompute) and cached."""

from __future__ import annotations

import torch

from engine.cache import KVCache
from engine.forward import forward_logits
from engine.model import LoadedModel


def _eos_ids(lm: LoadedModel) -> set:
    # Stop token id(s) from the generation config.
    eos = lm.model.generation_config.eos_token_id
    if eos is None:
        return set()
    return set(eos) if isinstance(eos, (list, tuple)) else {eos}


@torch.no_grad()
def generate(lm: LoadedModel, prompt: str, max_new_tokens: int | None = None):
    # Naive: re-feed full sequence each step. The reference oracle.
    max_new = max_new_tokens or lm.cfg.max_new_tokens
    eos_ids = _eos_ids(lm)

    enc = lm.tokenizer(prompt, return_tensors="pt").to(lm.device)
    ids = enc.input_ids
    prompt_len = ids.shape[1]

    for _ in range(max_new):
        logits = forward_logits(lm, ids)
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        ids = torch.cat([ids, next_id], dim=1)
        if next_id.item() in eos_ids:
            break

    text = lm.tokenizer.decode(ids[0, prompt_len:], skip_special_tokens=True)
    return ids, text


@torch.no_grad()
def generate_cached(lm: LoadedModel, prompt: str, max_new_tokens: int | None = None):
    # Cached: prefill once, then feed one token per step.
    max_new = max_new_tokens or lm.cfg.max_new_tokens
    eos_ids = _eos_ids(lm)

    enc = lm.tokenizer(prompt, return_tensors="pt").to(lm.device)
    ids = enc.input_ids
    prompt_len = ids.shape[1]

    cache = KVCache(lm.dims.n_layers)
    cur_input = ids
    pos = 0
    generated = []

    for _ in range(max_new):
        logits = forward_logits(lm, cur_input, cache, start_pos=pos)
        pos += cur_input.shape[1]
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(next_id)
        if next_id.item() in eos_ids:
            break
        cur_input = next_id

    ids = torch.cat([ids] + generated, dim=1)
    text = lm.tokenizer.decode(ids[0, prompt_len:], skip_special_tokens=True)
    return ids, text
