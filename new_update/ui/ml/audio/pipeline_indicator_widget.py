"""
SPANDHAN — Audio ML Pipeline Architecture Indicator
===================================================
Renders a subtle, scientific horizontal architecture diagram showing the 5 stages:
  INPUT → PREPROCESSING → FEATURE EXTRACTION → ML CLASSIFICATION → RESULT
Displays completed or active states dynamically during analysis.
"""

from typing import Optional, List
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QLabel,
    QFrame,
)

from ui.styles.theme import COLORS


class PipelineNode(QFrame):
    """Single architectural stage capsule in the pipeline."""

    def __init__(self, step_num: int, title: str, subtitle: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.step_num = step_num
        self.title_text = title
        self.subtitle_text = subtitle

        self.setFixedHeight(54)
        self.setObjectName("PipelineNode")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 7, 10, 7)
        layout.setSpacing(2)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        header_box = QHBoxLayout()
        header_box.setSpacing(5)
        header_box.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.status_icon = QLabel(f"0{step_num}")
        self.status_icon.setStyleSheet(
            f"color: {COLORS.TEXT_MUTED}; font-size: 9px; font-weight: 700; font-family: 'Consolas', monospace;"
        )
        self.title_lbl = QLabel(title)
        self.title_lbl.setStyleSheet(
            f"color: {COLORS.TEXT_SECONDARY}; font-size: 10px; font-weight: 700; letter-spacing: 0.8px;"
        )

        header_box.addWidget(self.status_icon)
        header_box.addWidget(self.title_lbl)
        layout.addLayout(header_box)

        self.sub_lbl = QLabel(subtitle)
        self.sub_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.sub_lbl.setStyleSheet(f"color: {COLORS.TEXT_MUTED}; font-size: 8px;")
        layout.addWidget(self.sub_lbl)

        self.set_state("idle")

    def set_state(self, state: str):
        """state can be 'idle', 'active', 'completed'."""
        if state == "completed":
            self.setStyleSheet(
                f"QFrame#PipelineNode {{ background-color: rgba(16, 185, 129, 0.10); "
                f"border: 1px solid {COLORS.PERSIAN_GREEN}; border-radius: 6px; }}"
            )
            self.status_icon.setText("✓")
            self.status_icon.setStyleSheet(f"color: {COLORS.PERSIAN_GREEN}; font-size: 10px; font-weight: 800;")
            self.title_lbl.setStyleSheet(f"color: {COLORS.TEXT_PRIMARY}; font-size: 10px; font-weight: 700;")
            self.sub_lbl.setStyleSheet(f"color: {COLORS.GREEN_LIGHT}; font-size: 8px;")
        elif state == "active":
            self.setStyleSheet(
                f"QFrame#PipelineNode {{ background-color: rgba(59, 130, 246, 0.16); "
                f"border: 1px solid {COLORS.BLUE_LIGHT}; border-radius: 6px; }}"
            )
            self.status_icon.setText("●")
            self.status_icon.setStyleSheet(f"color: {COLORS.CYAN}; font-size: 10px; font-weight: 800;")
            self.title_lbl.setStyleSheet(f"color: #FFFFFF; font-size: 10px; font-weight: 700;")
            self.sub_lbl.setStyleSheet(f"color: {COLORS.BLUE_LIGHT}; font-size: 8px;")
        else:  # idle
            self.setStyleSheet(
                f"QFrame#PipelineNode {{ background-color: #07111F; "
                f"border: 1px solid {COLORS.BORDER_SUBTLE}; border-radius: 6px; }}"
            )
            self.status_icon.setText(f"0{self.step_num}")
            self.status_icon.setStyleSheet(
                f"color: {COLORS.TEXT_MUTED}; font-size: 9px; font-weight: 700; font-family: 'Consolas', monospace;"
            )
            self.title_lbl.setStyleSheet(
                f"color: {COLORS.TEXT_SECONDARY}; font-size: 10px; font-weight: 700; letter-spacing: 0.8px;"
            )
            self.sub_lbl.setStyleSheet(f"color: {COLORS.TEXT_MUTED}; font-size: 8px;")


class AudioPipelineIndicatorWidget(QFrame):
    """Horizontal connected architecture flow diagram for Audio ML."""

    STAGES = [
        (1, "INPUT", "Audio Waveform"),
        (2, "PREPROCESSING", "Normalization"),
        (3, "FEATURE EXTRACTION", "35 DSP Metrics"),
        (4, "ML CLASSIFICATION", "Random Forest"),
        (5, "RESULT", "Class & Confidence"),
    ]

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("PipelineContainer")
        self.setStyleSheet(
            f"QFrame#PipelineContainer {{ background-color: {COLORS.BG_CARD}; "
            f"border: 1px solid {COLORS.BORDER_SUBTLE}; border-radius: 9px; padding: 12px 16px; }}"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Header
        h_col = QHBoxLayout()
        h_col.setSpacing(6)
        title = QLabel("AUDIO ML PIPELINE")
        title.setStyleSheet(f"color: {COLORS.TEXT_PRIMARY}; font-size: 11px; font-weight: 700; letter-spacing: 1.0px;")
        sub = QLabel("End-to-end signal processing and inference architecture")
        sub.setStyleSheet(f"color: {COLORS.TEXT_MUTED}; font-size: 10px;")
        h_col.addWidget(title)
        h_col.addWidget(sub)
        h_col.addStretch()
        layout.addLayout(h_col)

        # Stages flow
        flow_layout = QHBoxLayout()
        flow_layout.setContentsMargins(0, 4, 0, 0)
        flow_layout.setSpacing(8)

        self._nodes: List[PipelineNode] = []
        self._arrows: List[QLabel] = []

        for idx, (num, t, s) in enumerate(self.STAGES):
            node = PipelineNode(num, t, s, self)
            self._nodes.append(node)
            flow_layout.addWidget(node, stretch=1)

            if idx < len(self.STAGES) - 1:
                arrow = QLabel("▶")
                arrow.setAlignment(Qt.AlignmentFlag.AlignCenter)
                arrow.setStyleSheet(f"color: #1E293B; font-size: 10px;")
                self._arrows.append(arrow)
                flow_layout.addWidget(arrow)

        layout.addLayout(flow_layout)

    def set_stage_progress(self, current_stage: int):
        """
        current_stage:
          0: all idle
          1: input loaded
          2: preprocessing active
          3: feature extraction active
          4: ml inference active
          5: all completed
        """
        for i, node in enumerate(self._nodes):
            step = i + 1
            if current_stage == 5:
                node.set_state("completed")
            elif step < current_stage:
                node.set_state("completed")
            elif step == current_stage:
                node.set_state("active")
            else:
                node.set_state("idle")

        # Update arrow colors
        for i, arrow in enumerate(self._arrows):
            if current_stage >= i + 2:
                arrow.setStyleSheet(f"color: {COLORS.PERSIAN_GREEN}; font-size: 10px; font-weight: bold;")
            elif current_stage == i + 1:
                arrow.setStyleSheet(f"color: {COLORS.BLUE_LIGHT}; font-size: 10px;")
            else:
                arrow.setStyleSheet(f"color: #1E293B; font-size: 10px;")

    def reset(self):
        self.set_stage_progress(0)
