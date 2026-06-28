"""Load the model + tokenizer and read architecture dims from config."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from engine.config import EngineConfig


@dataclass
class ArchDims:
    """Architecture numbers read from model.config."""

    n_layers: int
    hidden: int
    n_q_heads: int
    n_kv_heads: int
    head_dim: int
    vocab_size: int

    @property
    def n_rep(self) -> int:
        # Query heads per KV head (GQA group size).
        return self.n_q_heads // self.n_kv_heads

    @classmethod
    def from_config(cls, config) -> "ArchDims":
        n_q_heads = config.num_attention_heads
        # Use explicit head_dim if present, else hidden / heads.
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // n_q_heads
        return cls(
            n_layers=config.num_hidden_layers,
            hidden=config.hidden_size,
            n_q_heads=n_q_heads,
            n_kv_heads=config.num_key_value_heads,
            head_dim=head_dim,
            vocab_size=config.vocab_size,
        )


def resolve_device(device: str) -> torch.device:
    # "auto" -> cuda, else mps, else cpu.
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(dtype: str, device: torch.device) -> torch.dtype:
    # "auto" -> bf16 on cuda, fp16 on mps, fp32 on cpu.
    if dtype != "auto":
        return getattr(torch, dtype)
    if device.type == "cuda":
        return torch.bfloat16
    if device.type == "mps":
        return torch.float16
    return torch.float32


class LoadedModel:
    """Loaded model, tokenizer, resolved device/dtype, and dims."""

    def __init__(self, cfg: EngineConfig):
        self.cfg = cfg
        self.device = resolve_device(cfg.device)
        self.dtype = resolve_dtype(cfg.dtype, self.device)

        # Tokenizer and weights from the same model_id.
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_id, dtype=self.dtype
        )
        self.model.to(self.device)
        self.model.eval()

        self.dims = ArchDims.from_config(self.model.config)


def load(cfg: EngineConfig | None = None) -> LoadedModel:
    return LoadedModel(cfg or EngineConfig())
