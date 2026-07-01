"""Verify the triton paged decode kernel matches reference + HF (run on CUDA)."""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.config import EngineConfig
from engine.generate import generate_paged
from engine.model import load
from tests.test_reference import MAX_NEW, PROMPTS, first_divergence, hf_greedy_ids


def main():
    assert torch.cuda.is_available(), "triton kernel needs CUDA; run this on a GPU box"
    lm = load(EngineConfig())

    for prompt in PROMPTS:
        lm.cfg.attention_backend = "reference"
        ref, _ = generate_paged(lm, prompt, max_new_tokens=MAX_NEW)
        lm.cfg.attention_backend = "triton"
        tri, _ = generate_paged(lm, prompt, max_new_tokens=MAX_NEW)
        hf = hf_greedy_ids(lm, prompt)

        ref, tri = ref[0].tolist(), tri[0].tolist()
        assert first_divergence(tri, ref) is None, f"triton != reference for {prompt!r}"
        assert first_divergence(tri, hf) is None, f"triton != HF for {prompt!r}"

    print(f"TRITON OK: kernel matches reference + HF on {len(PROMPTS)} prompts")


if __name__ == "__main__":
    main()
