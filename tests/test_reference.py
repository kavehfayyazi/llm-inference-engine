"""Correctness test: our token IDs vs HF generate() reference, greedy."""

from __future__ import annotations

import torch

from engine.config import EngineConfig
from engine.generate import _eos_ids, generate, generate_cached, generate_paged
from engine.model import load

PROMPTS = [
    "The capital of France is",
    "Once upon a time,",
    "In 1969, the first humans landed on the Moon and",
    "def add(a, b):",
]

MAX_NEW = 24


def hf_greedy_ids(lm, prompt):
    # HF reference: greedy, same length and stop tokens as ours.
    enc = lm.tokenizer(prompt, return_tensors="pt").to(lm.device)
    out = lm.model.generate(
        **enc,
        max_new_tokens=MAX_NEW,
        do_sample=False,
        num_beams=1,
        pad_token_id=next(iter(_eos_ids(lm)), None),
    )
    return out[0].tolist()


def first_divergence(a, b):
    # Index of first differing element, or None if one is a prefix of the other.
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def _assert_matches_hf(lm, gen_fn):
    for prompt in PROMPTS:
        ours, _ = gen_fn(lm, prompt, max_new_tokens=MAX_NEW)
        ours = ours[0].tolist()
        hf = hf_greedy_ids(lm, prompt)
        i = first_divergence(ours, hf)
        assert i is None, (
            f"mismatch at index {i} for {prompt!r}\n"
            f"  ours[{i}]={ours[i] if i < len(ours) else None} "
            f"hf[{i}]={hf[i] if i < len(hf) else None}"
        )


@torch.no_grad()
def test_reference_match():
    _assert_matches_hf(load(EngineConfig(dtype="float32")), generate)


@torch.no_grad()
def test_cached_reference_match():
    _assert_matches_hf(load(EngineConfig(dtype="float32")), generate_cached)


@torch.no_grad()
def test_paged_reference_match():
    _assert_matches_hf(load(EngineConfig(dtype="float32")), generate_paged)


if __name__ == "__main__":
    lm = load(EngineConfig(dtype="float32"))
    _assert_matches_hf(lm, generate)
    _assert_matches_hf(lm, generate_cached)
    _assert_matches_hf(lm, generate_paged)
    print(f"REFERENCE OK: naive + cached + paged match HF greedy on {len(PROMPTS)} prompts")
