"""A single in-flight request and its state."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

import torch

from engine.blocks import PagedKVCache


class State(enum.Enum):
    WAITING = 0   # arrived, not yet prefilled
    RUNNING = 1   # in the decode batch
    DONE = 2      # hit eos or max_new


@dataclass
class Request:
    id: int
    prompt_ids: torch.Tensor        # [1, P]
    kv: PagedKVCache
    max_new: int
    arrival: int = 0                # step tick the request shows up
    pos: int = 0                    # tokens whose KV is cached
    last_token: int = None          # most recent token, fed next step
    generated: list = field(default_factory=list)
    state: State = State.WAITING
    t_first: float = None           # wall seconds at first token (prefill done)
    t_finish: float = None          # wall seconds at completion
    s_first: int = None             # scheduler step at first token
    s_finish: int = None            # scheduler step at completion

    def full_ids(self) -> list:
        # Prompt + generated token ids.
        return self.prompt_ids[0].tolist() + self.generated
