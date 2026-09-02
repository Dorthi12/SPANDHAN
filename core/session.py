"""
Session module.
"""
from dataclasses import dataclass, field
from typing import Any
import numpy as np


@dataclass
class SignalData:
    signal: np.ndarray
    sampling_rate: float
    filename: str = ""
    domain: str = "general"
    source: str = ""
    channels: int = 1

    @property
    def num_samples(self) -> int:
        return len(self.signal)

    @property
    def duration(self) -> float:
        if self.sampling_rate <= 0:
            return 0.0
        return self.num_samples / self.sampling_rate


@dataclass
class SignalSession:
    raw: SignalData | None = None

    processed_signal: np.ndarray | None = None

    characteristics: dict[str, Any] = field(default_factory=dict)

    spectral: dict[str, Any] = field(default_factory=dict)

    noise: dict[str, Any] = field(default_factory=dict)

    ml: dict[str, Any] = field(default_factory=dict)

    diagnosis: dict[str, Any] = field(default_factory=dict)

    parameters: dict[str, Any] = field(default_factory=dict)

    log: list[str] = field(default_factory=list)

    def add_log(self, message: str) -> None:
        self.log.append(message)