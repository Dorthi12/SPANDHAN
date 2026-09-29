"""
SPANDHAN — System Status Bar Component
=======================================
Displays non-intrusive status indicators:
  Python ML ● Ready
  MATLAB DSP ● Ready
with dynamic status states (Ready, Initializing, Processing, Warning, Error).
"""

from typing import Optional
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QLabel,
    QFrame,
    QSizePolicy,
)

from ui.styles.theme import COLORS


class StatusIndicator(QWidget):
    """Subtle status indicator with colored dot and text label."""

    STATES = {
        "ready": ("●", COLORS.STATUS_READY, "Ready"),
        "initializing": ("○", COLORS.TEXT_MUTED, "Initializing"),
        "processing": ("○", COLORS.STATUS_PROCESSING, "Processing"),
        "warning": ("⚠", COLORS.STATUS_WARNING, "Warning"),
        "error": ("✕", COLORS.STATUS_ERROR, "Error"),
    }

    def __init__(self, system_name: str, initial_state: str = "ready", parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.system_name = system_name
        self._current_state = initial_state

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 2, 6, 2)
        layout.setSpacing(6)

        self.dot_label = QLabel()
        self.dot_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.text_label = QLabel()
        self.text_label.setStyleSheet(
            f"color: {COLORS.TEXT_SECONDARY}; font-size: 11px; font-weight: 500;"
        )

        layout.addWidget(self.dot_label)
        layout.addWidget(self.text_label)

        self.set_state(initial_state)

    def set_state(self, state: str, custom_text: Optional[str] = None):
        symbol, color, default_label = self.STATES.get(
            state.lower(), ("●", COLORS.STATUS_READY, "Ready")
        )
        self.dot_label.setText(symbol)
        self.dot_label.setStyleSheet(f"color: {color}; font-size: 10px; font-weight: bold;")

        label_str = custom_text if custom_text else default_label
        self.text_label.setText(f"{self.system_name} {symbol} {label_str}")


class SpandhanStatusBar(QFrame):
    """Persistent bottom status bar for SPANDHAN workstation."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setFixedHeight(30)
        self.setObjectName("SpandhanStatusBar")
        self.setStyleSheet(
            f"QFrame#SpandhanStatusBar {{ background-color: {COLORS.BG_SECONDARY}; "
            f"border-top: 1px solid {COLORS.BORDER_SUBTLE}; }}"
        )

        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 0, 16, 0)
        layout.setSpacing(20)

        # Left: System indicators
        self.ml_status = StatusIndicator("Python ML", "ready", self)
        self.matlab_status = StatusIndicator("MATLAB DSP", "ready", self)

        layout.addWidget(self.ml_status)
        layout.addWidget(self.matlab_status)

        layout.addStretch()

        # Right: Pipeline & Contract details
        self.model_status_lbl = QLabel("Model: image_signal_classifier.pkl (Loaded)")
        self.model_status_lbl.setStyleSheet(f"color: {COLORS.TEXT_MUTED}; font-size: 11px;")

        contract_lbl = QLabel("Contract: 128×128×1 Float32")
        contract_lbl.setStyleSheet(
            f"color: {COLORS.PERSIAN_GREEN}; font-size: 11px; font-weight: 600;"
        )

        layout.addWidget(self.model_status_lbl)
        layout.addWidget(contract_lbl)

    def set_ml_state(self, state: str, custom_text: Optional[str] = None):
        self.ml_status.set_state(state, custom_text)

    def set_matlab_state(self, state: str, custom_text: Optional[str] = None):
        self.matlab_status.set_state(state, custom_text)
