"""Engine configuration."""

from dataclasses import dataclass


@dataclass
class EngineConfig:
    # Selects both weights and tokenizer.
    model_id: str = "meta-llama/Llama-3.2-1B"
    # "auto" resolves cuda -> mps -> cpu.
    device: str = "auto"
    # "auto" picks a dtype from the device.
    dtype: str = "auto"
    # Greedy-decode length cap.
    max_new_tokens: int = 32
