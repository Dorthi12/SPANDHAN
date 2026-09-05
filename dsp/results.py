from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class DSPResult:
    method: str
    data: dict[str, Any] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)


@dataclass
class FrequencyResult(DSPResult):
    frequencies: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.float64)
    )

    def __post_init__(self) -> None:
        self.frequencies = np.asarray(
            self.frequencies,
            dtype=np.float64,
        )