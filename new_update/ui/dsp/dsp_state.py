"""
SPANDHAN — DSP State Management & Result Persistence
=====================================================
Centralized observable state store for DSP results across the workstation.
Enables reactive updates between ML classification, MATLAB execution,
and DSP visualization views.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union, Any, Tuple

from PySide6.QtCore import QObject, Signal
import scipy.io as sio

from ui.dsp.dsp_result_adapter import (
    AudioDSPResult,
    ImageDSPResult,
    parse_audio_dsp_result,
    parse_image_dsp_result,
    detect_modality_and_parse,
)


class DSPStateManager(QObject):
    """
    Singleton reactive state manager for SPANDHAN DSP.
    Preserves active results and alerts UI views when new analyses arrive.
    """

    audio_result_changed = Signal(object)      # Emits AudioDSPResult or None
    image_result_changed = Signal(object)      # Emits ImageDSPResult or None
    modality_changed = Signal(str)             # "audio" or "image"
    notification_posted = Signal(str, str)     # (message, "info" | "success" | "warning" | "error")

    _instance: Optional[DSPStateManager] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, parent: Optional[QObject] = None):
        if getattr(self, "_initialized", False):
            return
        super().__init__(parent)
        self._initialized = True

        self._audio_result: Optional[AudioDSPResult] = None
        self._image_result: Optional[ImageDSPResult] = None
        self._active_modality: str = "audio"

    # -----------------------------------------------------------------------
    # State Accessors
    # -----------------------------------------------------------------------

    @property
    def audio_result(self) -> Optional[AudioDSPResult]:
        return self._audio_result

    @property
    def image_result(self) -> Optional[ImageDSPResult]:
        return self._image_result

    @property
    def active_modality(self) -> str:
        return self._active_modality

    def set_active_modality(self, modality: str):
        mod = modality.lower().strip()
        if mod in ("audio", "image") and mod != self._active_modality:
            self._active_modality = mod
            self.modality_changed.emit(self._active_modality)

    def has_audio_result(self) -> bool:
        return self._audio_result is not None

    def has_image_result(self) -> bool:
        return self._image_result is not None

    # -----------------------------------------------------------------------
    # Result Setters
    # -----------------------------------------------------------------------

    def set_audio_result(self, result: Union[AudioDSPResult, dict, Any]):
        """Updates the active audio DSP result and emits change signal."""
        if isinstance(result, AudioDSPResult):
            self._audio_result = result
        else:
            self._audio_result = parse_audio_dsp_result(result)

        self.audio_result_changed.emit(self._audio_result)
        self.notification_posted.emit(
            f"Loaded audio DSP result for {self._audio_result.signal_class}", "success"
        )

    def set_image_result(self, result: Union[ImageDSPResult, dict, Any]):
        """Updates the active image DSP result and emits change signal."""
        if isinstance(result, ImageDSPResult):
            self._image_result = result
        else:
            self._image_result = parse_image_dsp_result(result)

        self.image_result_changed.emit(self._image_result)
        self.notification_posted.emit(
            f"Loaded image DSP result for {self._image_result.signal_class}", "success"
        )

    def clear_audio_result(self):
        self._audio_result = None
        self.audio_result_changed.emit(None)

    def clear_image_result(self):
        self._image_result = None
        self.image_result_changed.emit(None)

    # -----------------------------------------------------------------------
    # MATLAB .mat File Loader
    # -----------------------------------------------------------------------

    def load_mat_file(self, mat_path: Union[str, Path]) -> Tuple[bool, str]:
        """
        Loads a MATLAB output .mat file, auto-detects modality, and updates state.
        """
        p = Path(mat_path)
        if not p.exists():
            msg = f"File not found: {p}"
            self.notification_posted.emit(msg, "error")
            return False, msg

        try:
            mat_dict = sio.loadmat(str(p), squeeze_me=True, struct_as_record=False)

            # Look for "result" struct or root fields
            raw = mat_dict.get("result", mat_dict)
            parsed = detect_modality_and_parse(raw)

            if isinstance(parsed, AudioDSPResult):
                self.set_audio_result(parsed)
                self.set_active_modality("audio")
                return True, f"Successfully loaded Audio DSP result ({parsed.signal_class})"
            else:
                self.set_image_result(parsed)
                self.set_active_modality("image")
                return True, f"Successfully loaded Image DSP result ({parsed.signal_class})"

        except Exception as e:
            msg = f"Failed to load MATLAB .mat file: {e}"
            self.notification_posted.emit(msg, "error")
            return False, msg


def get_dsp_state() -> DSPStateManager:
    """Convenience getter for the global DSPStateManager singleton."""
    return DSPStateManager()
