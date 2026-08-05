"""
Test module: sampling_params.py
Defines the SamplingParams dataclass used to control text generation.
"""

import math
from dataclasses import dataclass, field
from typing import Optional

VALID_EARLY_STOPPING = {"never", "best"}


@dataclass
class SamplingParams:
    """Parameters for controlling text generation sampling."""

    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    max_tokens: int = 16
    stop: list[str] = field(default_factory=list)
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    early_stopping: str = "never"
    seed: Optional[int] = None

    def __post_init__(self):
        self._validate()

    def _validate(self):
        if not 0.0 <= self.top_p <= 1.0:
            raise ValueError(f"top_p must be in [0, 1], got {self.top_p}")
        if self.temperature < 0.0:
            raise ValueError(f"temperature must be >= 0, got {self.temperature}")
        if self.max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {self.max_tokens}")
        if self.early_stopping not in VALID_EARLY_STOPPING:
            raise ValueError(
                f"early_stopping must be one of {VALID_EARLY_STOPPING}"
            )

    def is_greedy(self) -> bool:
        """Returns True if this config performs greedy decoding."""
        return self.temperature == 0.0

    def is_deterministic(self) -> bool:
        """Returns True if sampling is fully deterministic."""
        return self.is_greedy() or self.seed is not None


def default_sampling_params() -> SamplingParams:
    """Returns a default SamplingParams instance."""
    return SamplingParams(temperature=1.0, max_tokens=64)


def greedy_params(max_tokens: int = 256) -> SamplingParams:
    """Returns SamplingParams configured for greedy decoding."""
    return SamplingParams(temperature=0.0, max_tokens=max_tokens)
