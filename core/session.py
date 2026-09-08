"""
Session module for Spandhan DSP Application.
Maintains global session state, pipeline progress, and PySide6 Qt signals.
"""

from dataclasses import dataclass, field
from typing import Any, Optional
import numpy as np
from PySide6.QtCore import QObject, Signal


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
        return len(self.signal) if self.signal is not None else 0

    @property
    def duration(self) -> float:
        if self.sampling_rate <= 0 or self.signal is None:
            return 0.0
        return self.num_samples / self.sampling_rate


@dataclass
class SignalSession:
    raw: Optional[SignalData] = None
    processed_signal: Optional[np.ndarray] = None
    
    # Analysis results
    characteristics: dict[str, Any] = field(default_factory=dict)
    spectral: dict[str, Any] = field(default_factory=dict)
    high_res: dict[str, Any] = field(default_factory=dict)
    noise: dict[str, Any] = field(default_factory=dict)
    features: dict[str, Any] = field(default_factory=dict)
    domain_results: dict[str, Any] = field(default_factory=dict)
    ml: dict[str, Any] = field(default_factory=dict)
    diagnosis: dict[str, Any] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)
    log: list[str] = field(default_factory=list)

    def add_log(self, message: str) -> None:
        self.log.append(message)

    @property
    def active_signal(self) -> Optional[np.ndarray]:
        """Return processed signal if available, otherwise raw signal."""
        if self.processed_signal is not None:
            return self.processed_signal
        if self.raw is not None:
            return self.raw.signal
        return None

    @property
    def active_sampling_rate(self) -> float:
        """Return sampling rate from raw signal or default 1000.0."""
        if self.raw is not None and self.raw.sampling_rate > 0:
            return self.raw.sampling_rate
        return 1000.0


class SessionManager(QObject):
    """
    Singleton-style Session Manager that wraps SignalSession and emits Qt signals.
    """
    signal_loaded = Signal()
    session_updated = Signal()
    pipeline_updated = Signal()

    def __init__(self):
        super().__init__()
        self.session = SignalSession()
        self._ml_model_loaded = False
        self._ml_model_path = ""
        self.check_ml_model_status()

    def check_ml_model_status(self):
        """Check if the default trained noise model is present on disk."""
        import os
        from core.config import DEFAULT_MODEL_PATH
        model_path = DEFAULT_MODEL_PATH
        if os.path.exists(model_path):
            self._ml_model_loaded = True
            self._ml_model_path = model_path
        else:
            self._ml_model_loaded = False
            self._ml_model_path = ""

    @property
    def is_ml_model_loaded(self) -> bool:
        return self._ml_model_loaded

    @property
    def ml_model_path(self) -> str:
        return self._ml_model_path

    def load_signal(self, signal: np.ndarray, sampling_rate: float, filename: str = "", domain: str = "general", source: str = "file"):
        """Load a new raw signal into the session."""
        self.session.raw = SignalData(
            signal=np.asarray(signal, dtype=np.float64),
            sampling_rate=float(sampling_rate),
            filename=filename or "Loaded Signal",
            domain=domain,
            source=source,
            channels=1 if signal.ndim == 1 else signal.shape[1]
        )
        self.session.processed_signal = None
        self.session.characteristics.clear()
        self.session.spectral.clear()
        self.session.high_res.clear()
        self.session.noise.clear()
        self.session.features.clear()
        self.session.domain_results.clear()
        self.session.ml.clear()
        self.session.diagnosis.clear()
        self.session.add_log(f"Loaded signal '{self.session.raw.filename}' ({len(signal)} samples, {sampling_rate} Hz)")
        
        self.signal_loaded.emit()
        self.session_updated.emit()
        self.pipeline_updated.emit()

    def update_processed_signal(self, processed: np.ndarray, operation_desc: str = ""):
        """Update preprocessed signal."""
        self.session.processed_signal = np.asarray(processed, dtype=np.float64)
        if operation_desc:
            self.session.add_log(f"Applied preprocessing: {operation_desc}")
        self.session_updated.emit()
        self.pipeline_updated.emit()

    def reset_preprocessing(self):
        """Reset preprocessing to original raw signal."""
        self.session.processed_signal = None
        self.session.add_log("Reset preprocessing to original raw signal")
        self.session_updated.emit()
        self.pipeline_updated.emit()

    def get_pipeline_stages(self) -> list[dict[str, str]]:
        """
        Returns status of each pipeline stage:
        'INPUT', 'PREPROCESSING', 'DSP', 'NOISE', 'FEATURES', 'HIGH-RES', 'DOMAIN', 'REPORT'
        Statuses: 'Not started', 'Ready', 'Complete', 'Warning'
        """
        has_raw = self.session.raw is not None and len(self.session.raw.signal) > 0
        has_prep = self.session.processed_signal is not None
        has_dsp = bool(self.session.spectral)
        has_noise = bool(self.session.noise)
        has_features = bool(self.session.features)
        has_high_res = bool(self.session.high_res)
        has_domain = bool(self.session.domain_results)
        has_report = bool(self.session.diagnosis)

        return [
            {"name": "INPUT", "status": "Complete" if has_raw else "Not started", "desc": "Signal imported"},
            {"name": "PREPROCESSING", "status": "Complete" if has_prep else ("Ready" if has_raw else "Not started"), "desc": "Filtering & Denoising"},
            {"name": "DSP", "status": "Complete" if has_dsp else ("Ready" if has_raw else "Not started"), "desc": "Spectral & Time DSP"},
            {"name": "NOISE", "status": "Complete" if has_noise else ("Ready" if has_raw else "Not started"), "desc": "Noise Assessment & ML"},
            {"name": "FEATURES", "status": "Complete" if has_features else ("Ready" if has_raw else "Not started"), "desc": "20-Feature Vector"},
            {"name": "HIGH-RES", "status": "Complete" if has_high_res else ("Ready" if has_raw else "Not started"), "desc": "MUSIC & ESPRIT"},
            {"name": "DOMAIN", "status": "Complete" if has_domain else ("Ready" if has_raw else "Not started"), "desc": "Audio/ECG/MCSA"},
            {"name": "REPORT", "status": "Complete" if has_report else ("Ready" if has_raw else "Not started"), "desc": "Summary & Diagnostics"},
        ]


# Global session manager instance
session_manager = SessionManager()