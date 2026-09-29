"""
SPANDHAN — DSP Analysis Workstation Page
=========================================
The primary scientific digital signal and image processing interface.
Integrates:
  - Header: Breadcrumbs, Title, Subtitle, Modality & Signal Class Badges, Status
  - Pipeline Stage Indicator: INPUT → PREPROCESSED → CLASSIFIED → [DSP ANALYZED] → VISUALIZED
  - Compact Technical Summary: Signal Class, DSP Status, Executed Modules, Sampling/Geometry, Duration
  - Interactive Modality & Canonical Sample Switcher
  - Dynamic Result Viewers: AudioDSPView and ImageDSPView (strict class-aware layout)
  - Diagnostic Drawer: Skipped and failed analyses with technical reasons
  - Empty State: Guided fallback when no analysis is active
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Tuple, Any

from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtGui import QColor, QFont, QLinearGradient, QPainter, QCursor
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QPushButton,
    QFrame,
    QScrollArea,
    QComboBox,
    QFileDialog,
    QSizePolicy,
    QProgressBar,
    QMessageBox,
)

from ui.styles.theme import COLORS
from ui.dsp.dsp_theme import DSP_THEME, get_chip_stylesheet
from ui.dsp.dsp_state import DSPStateManager, get_dsp_state
from ui.dsp.dsp_controller import DSPController
from ui.dsp.dsp_result_adapter import AudioDSPResult, ImageDSPResult
from ui.dsp.dsp_cards import DiagnosticSection
from ui.dsp.audio_dsp_view import AudioDSPView
from ui.dsp.image_dsp_view import ImageDSPView
from ui.dsp.matlab_runner import MATLABDSPRunner


# ---------------------------------------------------------------------------
# Pipeline Indicator Bar
# ---------------------------------------------------------------------------

class PipelineIndicatorBar(QFrame):
    """
    Compact scientific breadcrumb indicator showing the 5-stage processing pipeline:
    INPUT → PREPROCESSED → CLASSIFIED → [DSP ANALYZED] → VISUALIZED
    Highlights DSP ANALYZED in glowing Persian green when results are active.
    """

    STAGES = [
        ("INPUT", False),
        ("PREPROCESSED", False),
        ("CLASSIFIED", False),
        ("DSP ANALYZED", True),
        ("VISUALIZED", True),
    ]

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setFixedHeight(34)
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {DSP_THEME.BG_INPUT};
                border: 1px solid {DSP_THEME.BORDER_SOLID};
                border-radius: 6px;
            }}
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 0, 12, 0)
        layout.setSpacing(6)

        prefix_lbl = QLabel("PIPELINE:")
        prefix_lbl.setStyleSheet(f"color: {DSP_THEME.TEXT_MUTED}; font-size: 10px; font-weight: 700; letter-spacing: 1px;")
        layout.addWidget(prefix_lbl)
        layout.addSpacing(4)

        for i, (stage_name, is_dsp_stage) in enumerate(self.STAGES):
            if is_dsp_stage:
                badge = QLabel(stage_name)
                badge.setStyleSheet(f"""
                    color: #FFFFFF;
                    background-color: {DSP_THEME.INDIGO};
                    border: 1px solid {DSP_THEME.BLUE_LIGHT};
                    border-radius: 4px;
                    padding: 2px 7px;
                    font-size: 9px;
                    font-weight: 700;
                    letter-spacing: 0.5px;
                """)
                layout.addWidget(badge)
            else:
                lbl = QLabel(stage_name)
                lbl.setStyleSheet(f"color: {DSP_THEME.TEXT_SECONDARY}; font-size: 9px; font-weight: 600; letter-spacing: 0.5px;")
                layout.addWidget(lbl)

            if i < len(self.STAGES) - 1:
                arrow = QLabel("→")
                arrow.setStyleSheet(f"color: {DSP_THEME.TEXT_MUTED}; font-size: 11px;")
                layout.addWidget(arrow)

        layout.addStretch()


# ---------------------------------------------------------------------------
# Technical Analysis Summary Strip
# ---------------------------------------------------------------------------

class AnalysisSummaryStrip(QFrame):
    """
    Compact summary bar presenting core MATLAB execution parameters:
    Signal Class, Status, Executed Modules, Sampling/Geometry, Duration/Pixels.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("AnalysisSummaryStrip")
        self.setStyleSheet(f"""
            QFrame#AnalysisSummaryStrip {{
                background-color: {DSP_THEME.BG_CARD};
                border: 1px solid {DSP_THEME.BORDER_SUBTLE};
                border-radius: 8px;
            }}
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(20)

        self.class_val = self._add_stat(layout, "SIGNAL / CLASS", "--")
        self.status_val = self._add_stat(layout, "DSP STATUS", "--")
        self.modules_val = self._add_stat(layout, "EXECUTED MODULES", "--")
        self.sampling_val = self._add_stat(layout, "SAMPLING / GEOM", "--")
        self.extent_val = self._add_stat(layout, "DURATION / SAMPLES", "--")
        layout.addStretch()

    def _add_stat(self, parent_layout: QHBoxLayout, title: str, init_val: str) -> QLabel:
        col = QVBoxLayout()
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(2)

        lbl = QLabel(title)
        lbl.setStyleSheet(f"color: {DSP_THEME.TEXT_MUTED}; font-size: 9px; font-weight: 700; letter-spacing: 0.8px;")
        col.addWidget(lbl)

        val = QLabel(init_val)
        val.setStyleSheet(f"color: {DSP_THEME.TEXT_PRIMARY}; font-family: 'Consolas', monospace; font-size: 11px; font-weight: 600;")
        col.addWidget(val)

        parent_layout.addLayout(col)
        return val

    def update_audio(self, res: AudioDSPResult):
        self.class_val.setText(res.signal_class.upper())
        self.status_val.setText(res.status)
        self.status_val.setStyleSheet(
            f"color: {DSP_THEME.PERSIAN_GREEN if 'Completed' in res.status else DSP_THEME.AMBER}; "
            f"font-family: 'Consolas', monospace; font-size: 11px; font-weight: bold;"
        )
        mod_str = " · ".join(res.completed_analyses) if res.completed_analyses else "None"
        self.modules_val.setText(mod_str)
        self.sampling_val.setText(f"{res.sampling_frequency:,.0f} Hz ({res.sampling_frequency/1000:.1f} kHz)")
        self.extent_val.setText(f"{res.duration:.2f} s  ({res.signal_length:,} pts)")

    def update_image(self, res: ImageDSPResult):
        self.class_val.setText(res.signal_class.upper())
        self.status_val.setText(res.status)
        self.status_val.setStyleSheet(
            f"color: {DSP_THEME.PERSIAN_GREEN if 'Completed' in res.status else DSP_THEME.AMBER}; "
            f"font-family: 'Consolas', monospace; font-size: 11px; font-weight: bold;"
        )
        mod_str = " · ".join(res.completed_analyses) if res.completed_analyses else "None"
        self.modules_val.setText(mod_str)
        self.sampling_val.setText(f"{res.image_size[0]} × {res.image_size[1]} px")
        self.extent_val.setText(f"{res.image_size[0] * res.image_size[1]:,} pixels")



# ---------------------------------------------------------------------------
# Preprocessing Steps Panel
# ---------------------------------------------------------------------------

class PreprocessingStepsPanel(QFrame):
    """
    Collapsible card showing the MATLAB preprocessing steps that were applied
    to the signal before it reached the DSP analysis stage.
    Reflects the exact contract of preprocessAudioFile.m / preprocessImage.m.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("PreprocessingStepsPanel")
        self._collapsed = True
        self._steps: List[Tuple[str, str]] = []

        self.setStyleSheet(f"""
            QFrame#PreprocessingStepsPanel {{
                background-color: {DSP_THEME.BG_CARD};
                border: 1px solid {DSP_THEME.BORDER_SUBTLE};
                border-radius: 8px;
            }}
        """)

        self._root_layout = QVBoxLayout(self)
        self._root_layout.setContentsMargins(14, 10, 14, 10)
        self._root_layout.setSpacing(8)

        # Header row (always visible)
        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)

        icon_lbl = QLabel("⚙")
        icon_lbl.setStyleSheet(f"color: {DSP_THEME.BLUE_LIGHT}; font-size: 13px;")
        header_row.addWidget(icon_lbl)

        title_lbl = QLabel("MATLAB PREPROCESSING STEPS")
        title_lbl.setStyleSheet(
            f"color: {DSP_THEME.TEXT_SECONDARY}; font-size: 10px; font-weight: 700; "
            f"letter-spacing: 0.8px;"
        )
        header_row.addWidget(title_lbl)

        self.source_chip = QLabel("—")
        self.source_chip.setStyleSheet(
            f"color: {DSP_THEME.TEXT_MUTED}; font-size: 9px; font-family: 'Consolas', monospace;"
        )
        header_row.addWidget(self.source_chip)
        header_row.addStretch()

        self.toggle_btn = QPushButton("▸ Show Steps")
        self.toggle_btn.setFlat(True)
        self.toggle_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.toggle_btn.setStyleSheet(
            f"color: {DSP_THEME.CYAN}; font-size: 10px; font-weight: 600; "
            f"background: transparent; border: none; padding: 0px;"
        )
        self.toggle_btn.clicked.connect(self._toggle)
        header_row.addWidget(self.toggle_btn)

        self._root_layout.addLayout(header_row)

        # Steps container (collapsible)
        self._steps_frame = QFrame()
        self._steps_frame.setVisible(False)
        self._steps_layout = QVBoxLayout(self._steps_frame)
        self._steps_layout.setContentsMargins(0, 6, 0, 0)
        self._steps_layout.setSpacing(5)
        self._root_layout.addWidget(self._steps_frame)

    def set_steps(self, steps: List[Tuple[str, str]], source: str = ""):
        """
        Update the displayed preprocessing steps.
        steps: list of (step_name, description) tuples
        source: short label (e.g. "Audio: signal.wav")
        """
        self._steps = steps

        if source:
            self.source_chip.setText(f"• {source}")

        # Clear and rebuild the steps grid
        while self._steps_layout.count():
            item = self._steps_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for i, (step_name, description) in enumerate(steps):
            row = QFrame()
            row.setStyleSheet(
                f"background-color: {DSP_THEME.BG_INPUT}; border-radius: 4px; "
                f"border: 1px solid {DSP_THEME.BORDER_SUBTLE};"
            )
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(10, 5, 10, 5)
            row_layout.setSpacing(10)

            num = QLabel(f"{i+1:02d}")
            num.setFixedWidth(22)
            num.setStyleSheet(
                f"color: {DSP_THEME.INDIGO_LIGHT}; font-size: 9px; font-weight: 700; "
                f"font-family: 'Consolas', monospace;"
            )

            step_lbl = QLabel(step_name)
            step_lbl.setFixedWidth(200)
            step_lbl.setStyleSheet(
                f"color: {DSP_THEME.CYAN}; font-size: 10px; font-weight: 600; "
                f"font-family: 'Consolas', monospace;"
            )

            desc_lbl = QLabel(description)
            desc_lbl.setStyleSheet(f"color: {DSP_THEME.TEXT_SECONDARY}; font-size: 10px;")

            check = QLabel("✓")
            check.setFixedWidth(16)
            check.setStyleSheet(f"color: {DSP_THEME.PERSIAN_GREEN}; font-size: 11px; font-weight: bold;")
            check.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

            row_layout.addWidget(num)
            row_layout.addWidget(step_lbl)
            row_layout.addWidget(desc_lbl, stretch=1)
            row_layout.addWidget(check)

            self._steps_layout.addWidget(row)

    def clear_steps(self):
        self._steps = []
        self.source_chip.setText("—")
        while self._steps_layout.count():
            item = self._steps_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _toggle(self):
        self._collapsed = not self._collapsed
        self._steps_frame.setVisible(not self._collapsed)
        self.toggle_btn.setText("▾ Hide Steps" if not self._collapsed else "▸ Show Steps")


# ---------------------------------------------------------------------------
# MATLAB Input Controls Panel
# ---------------------------------------------------------------------------

class MATLABInputPanel(QFrame):
    """
    Primary input panel for the DSP page.
    Provides three input pathways:
      1. Use processed signal from Audio/Image ML page (auto-populated after classification)
      2. Browse for a raw audio/image file and run the full MATLAB pipeline
      3. Import an existing MATLAB .mat result file
    Includes a MATLAB availability indicator and progress bar during execution.
    """

    run_pipeline_requested  = Signal(str, str)    # (modality, file_path)
    import_mat_requested    = Signal()
    use_ml_signal_requested = Signal()            # Trigger DSP on pending ML signal

    def __init__(self, modality: str = "audio", parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.modality = modality.lower()
        self.setObjectName("MATLABInputPanel")

        self.setStyleSheet(f"""
            QFrame#MATLABInputPanel {{
                background-color: {DSP_THEME.BG_CARD};
                border: 1px solid {DSP_THEME.BORDER_SOLID};
                border-radius: 10px;
            }}
        """)

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 12, 16, 12)
        root.setSpacing(10)

        # ---- Row 1: Section title ----
        title_row = QHBoxLayout()
        icon = QLabel("⟿" if modality == "audio" else "▤")
        icon.setStyleSheet(f"color: {DSP_THEME.BLUE_LIGHT}; font-size: 14px;")
        title_row.addWidget(icon)

        tl = QLabel(f"DSP INPUT  —  {modality.upper()}")
        tl.setStyleSheet(
            f"color: {DSP_THEME.TEXT_SECONDARY}; font-size: 10px; font-weight: 700; letter-spacing: 1px;"
        )
        title_row.addWidget(tl)
        title_row.addStretch()

        # MATLAB availability indicator
        self._matlab_status_lbl = QLabel("Checking MATLAB…")
        self._matlab_status_lbl.setStyleSheet(
            f"color: {DSP_THEME.AMBER}; font-size: 9px; font-family: 'Consolas', monospace;"
        )
        title_row.addWidget(self._matlab_status_lbl)
        root.addLayout(title_row)

        # ---- Row 2: ML Signal Banner (only shown when pending signal available) ----
        self._ml_banner = QFrame()
        self._ml_banner.setStyleSheet(
            f"background-color: {DSP_THEME.BG_INPUT}; border: 1px solid rgba(16,185,129,0.35); "
            f"border-radius: 6px;"
        )
        ml_row = QHBoxLayout(self._ml_banner)
        ml_row.setContentsMargins(12, 7, 12, 7)
        ml_row.setSpacing(10)

        self._ml_banner_lbl = QLabel("✦  Preprocessed signal ready from ML workstation")
        self._ml_banner_lbl.setStyleSheet(
            f"color: {DSP_THEME.PERSIAN_GREEN}; font-size: 11px; font-weight: 600;"
        )
        ml_row.addWidget(self._ml_banner_lbl, stretch=1)

        self._use_ml_btn = QPushButton("▶  Run DSP on This Signal")
        self._use_ml_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._use_ml_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {DSP_THEME.PERSIAN_GREEN};
                color: #000000;
                border: none;
                border-radius: 5px;
                padding: 5px 14px;
                font-size: 11px;
                font-weight: 700;
            }}
            QPushButton:hover {{ background-color: #34D399; }}
        """)
        self._use_ml_btn.clicked.connect(self.use_ml_signal_requested.emit)
        ml_row.addWidget(self._use_ml_btn)

        self._ml_banner.setVisible(False)
        root.addWidget(self._ml_banner)

        # ---- Row 3: Action buttons ----
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        # Browse for file
        self._browse_btn = QPushButton("📂  Browse File…")
        self._browse_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._browse_btn.setStyleSheet(self._btn_style(DSP_THEME.INDIGO, DSP_THEME.BLUE_LIGHT))
        self._browse_btn.clicked.connect(self._browse_file)
        btn_row.addWidget(self._browse_btn)

        # Run MATLAB pipeline
        self._run_btn = QPushButton("⚡  Run MATLAB Pipeline")
        self._run_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._run_btn.setEnabled(False)
        self._run_btn.setStyleSheet(self._btn_style(DSP_THEME.BG_SURFACE_ALT, DSP_THEME.BORDER_SOLID, text=DSP_THEME.TEXT_SECONDARY))
        self._run_btn.clicked.connect(self._run_pipeline)
        btn_row.addWidget(self._run_btn)

        btn_row.addStretch()

        # Import .mat
        self._import_btn = QPushButton("Import .mat Result…")
        self._import_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._import_btn.setStyleSheet(self._btn_style(DSP_THEME.BG_SURFACE_ALT, DSP_THEME.BORDER_SOLID, text=DSP_THEME.TEXT_MUTED))
        self._import_btn.clicked.connect(self.import_mat_requested.emit)
        btn_row.addWidget(self._import_btn)

        root.addLayout(btn_row)

        # ---- Row 4: Selected file label ----
        self._file_lbl = QLabel("No file selected")
        self._file_lbl.setStyleSheet(
            f"color: {DSP_THEME.TEXT_MUTED}; font-size: 10px; font-family: 'Consolas', monospace;"
        )
        root.addWidget(self._file_lbl)

        # ---- Row 5: Progress bar (hidden when idle) ----
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 0)  # indeterminate
        self._progress_bar.setFixedHeight(4)
        self._progress_bar.setTextVisible(False)
        self._progress_bar.setVisible(False)
        self._progress_bar.setStyleSheet(f"""
            QProgressBar {{
                background-color: {DSP_THEME.BG_INPUT};
                border: none;
                border-radius: 2px;
            }}
            QProgressBar::chunk {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {DSP_THEME.INDIGO}, stop:1 {DSP_THEME.PERSIAN_GREEN});
                border-radius: 2px;
            }}
        """)
        root.addWidget(self._progress_bar)

        self._stage_lbl = QLabel("")
        self._stage_lbl.setStyleSheet(
            f"color: {DSP_THEME.AMBER}; font-size: 10px; font-family: 'Consolas', monospace;"
        )
        self._stage_lbl.setVisible(False)
        root.addWidget(self._stage_lbl)

        # ---- Internal state ----
        self._selected_file: str = ""
        self._update_matlab_status()

    def _btn_style(self, bg: str, border: str, text: str = "#FFFFFF") -> str:
        return f"""
            QPushButton {{
                background-color: {bg};
                color: {text};
                border: 1px solid {border};
                border-radius: 6px;
                padding: 5px 14px;
                font-size: 11px;
                font-weight: 600;
            }}
            QPushButton:hover {{
                background-color: {DSP_THEME.BG_CARD_HOVER};
                border: 1px solid {DSP_THEME.BORDER_HOVER};
                color: {DSP_THEME.TEXT_PRIMARY};
            }}
            QPushButton:disabled {{
                opacity: 0.4;
            }}
        """

    def _update_matlab_status(self):
        runner = MATLABDSPRunner()
        if runner.matlab_available:
            self._matlab_status_lbl.setText(f"● MATLAB: {Path(runner.matlab_path).name}")
            self._matlab_status_lbl.setStyleSheet(
                f"color: {DSP_THEME.PERSIAN_GREEN}; font-size: 9px; font-family: 'Consolas', monospace;"
            )
        else:
            self._matlab_status_lbl.setText("○ MATLAB: Not detected — use .mat import")
            self._matlab_status_lbl.setStyleSheet(
                f"color: {DSP_THEME.AMBER}; font-size: 9px; font-family: 'Consolas', monospace;"
            )

    def set_ml_signal_available(self, info: Optional[dict]):
        """Called when DSP state has a pending ML signal available."""
        if info:
            cls  = info.get("predicted_class", "Unknown")
            conf = info.get("confidence", 0.0)
            src  = Path(info.get("file_path", "")).name or "signal"
            self._ml_banner_lbl.setText(
                f"✦  ML classified '{src}' as  {cls}  ({conf*100:.1f}%)  —  ready for MATLAB DSP"
            )
            self._ml_banner.setVisible(True)
        else:
            self._ml_banner.setVisible(False)

    def _browse_file(self):
        if self.modality == "audio":
            filt = "Audio Files (*.wav *.flac *.mp3 *.ogg);;All Files (*.*)"
            title = "Select Audio Signal"
        else:
            filt = "Image Files (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All Files (*.*)"
            title = "Select Image Signal"

        path, _ = QFileDialog.getOpenFileName(None, title, "", filt)
        if path:
            self._selected_file = path
            self._file_lbl.setText(f"▸  {Path(path).name}")
            self._file_lbl.setStyleSheet(
                f"color: {DSP_THEME.CYAN}; font-size: 10px; font-family: 'Consolas', monospace;"
            )
            runner = MATLABDSPRunner()
            self._run_btn.setEnabled(runner.matlab_available)
            self._run_btn.setStyleSheet(
                self._btn_style(DSP_THEME.INDIGO, DSP_THEME.BLUE_LIGHT)
                if runner.matlab_available
                else self._btn_style(DSP_THEME.BG_SURFACE_ALT, DSP_THEME.BORDER_SOLID, DSP_THEME.TEXT_SECONDARY)
            )

    def _run_pipeline(self):
        if self._selected_file:
            self.run_pipeline_requested.emit(self.modality, self._selected_file)

    def set_running(self, stage: str = ""):
        self._progress_bar.setVisible(True)
        self._stage_lbl.setVisible(True)
        self._stage_lbl.setText(f"⟳  {stage}")
        self._browse_btn.setEnabled(False)
        self._run_btn.setEnabled(False)
        self._use_ml_btn.setEnabled(False)

    def set_idle(self):
        self._progress_bar.setVisible(False)
        self._stage_lbl.setVisible(False)
        self._stage_lbl.setText("")
        self._browse_btn.setEnabled(True)
        runner = MATLABDSPRunner()
        self._run_btn.setEnabled(bool(self._selected_file and runner.matlab_available))
        self._use_ml_btn.setEnabled(True)

    def set_error(self, message: str):
        self._progress_bar.setVisible(False)
        self._stage_lbl.setVisible(True)
        self._stage_lbl.setText(f"✕  {message[:120]}")
        self._stage_lbl.setStyleSheet(
            f"color: #F87171; font-size: 10px; font-family: 'Consolas', monospace;"
        )
        self.set_idle()


# ---------------------------------------------------------------------------
# Empty State Widget
# ---------------------------------------------------------------------------


class DSPEmptyStateWidget(QFrame):
    """
    Rendered when no active DSP result is loaded.
    Explains how to generate results via the Audio/Image ML pipelines and provides quick actions.
    """

    navigate_audio_ml = Signal()
    navigate_image_ml = Signal()
    load_demo_requested = Signal()

    def __init__(self, modality: str = "audio", parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {DSP_THEME.BG_CARD};
                border: 1px dashed {DSP_THEME.BORDER_SOLID};
                border-radius: 12px;
                padding: 40px;
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(14)

        icon_lbl = QLabel("∿" if modality == "audio" else "▨")
        icon_lbl.setStyleSheet(f"color: {DSP_THEME.BLUE_LIGHT}; font-size: 42px;")
        icon_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(icon_lbl)

        title_lbl = QLabel(f"No {modality.title()} DSP Analysis Available")
        title_lbl.setStyleSheet(f"color: {DSP_THEME.TEXT_PRIMARY}; font-size: 18px; font-weight: bold;")
        title_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_lbl)

        desc_lbl = QLabel(
            f"Upload and classify an {modality} signal from the machine learning workstation\n"
            "to trigger MATLAB class-aware digital signal processing routines."
        )
        desc_lbl.setStyleSheet(f"color: {DSP_THEME.TEXT_SECONDARY}; font-size: 12px; line-height: 1.5;")
        desc_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(desc_lbl)

        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(12)
        btn_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Navigation action button
        nav_btn = QPushButton(f"← Navigate to {modality.title()} ML")
        nav_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        nav_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {DSP_THEME.INDIGO};
                color: #FFFFFF;
                border: 1px solid {DSP_THEME.BLUE};
                border-radius: 6px;
                padding: 8px 18px;
                font-weight: 600;
                font-size: 12px;
            }}
            QPushButton:hover {{
                background-color: {DSP_THEME.INDIGO_LIGHT};
            }}
        """)
        if modality == "audio":
            nav_btn.clicked.connect(self.navigate_audio_ml.emit)
        else:
            nav_btn.clicked.connect(self.navigate_image_ml.emit)
        btn_layout.addWidget(nav_btn)

        # Quick demo load button
        demo_btn = QPushButton(f"Load Canonical {modality.title()} Test Sample")
        demo_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        demo_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {DSP_THEME.BG_SURFACE_ALT};
                color: {DSP_THEME.CYAN};
                border: 1px solid {DSP_THEME.BORDER_SOLID};
                border-radius: 6px;
                padding: 8px 18px;
                font-weight: 600;
                font-size: 12px;
            }}
            QPushButton:hover {{
                background-color: {DSP_THEME.BG_CARD_HOVER};
                border: 1px solid {DSP_THEME.BORDER_HOVER};
            }}
        """)
        demo_btn.clicked.connect(self.load_demo_requested.emit)
        btn_layout.addWidget(demo_btn)

        layout.addLayout(btn_layout)


# ---------------------------------------------------------------------------
# Master DSP Page Component
# ---------------------------------------------------------------------------

class DSPPage(QWidget):
    """
    Primary DSP Analysis workstation view for SPANDHAN.
    Supports both Audio and Image modalities with shared header and dynamic view routing.
    """

    navigate_requested = Signal(str)  # Emits target page key (e.g. "audio_ml", "image_ml")

    def __init__(self, initial_modality: str = "audio", locked_modality: Optional[str] = None, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.modality = initial_modality.lower()
        self.locked_modality = locked_modality.lower() if locked_modality else None
        self.state = get_dsp_state()
        self.controller = DSPController(self.state)
        self._matlab_runner: Optional[MATLABDSPRunner] = None

        self._init_ui()
        self._connect_signals()

        # If a result is already in state (from previous session), render it
        if self.modality == "audio" and self.state.has_audio_result():
            self._on_audio_result_changed(self.state.audio_result)
        elif self.modality == "image" and self.state.has_image_result():
            self._on_image_result_changed(self.state.image_result)
        else:
            # Show empty state — user must provide input
            self._show_empty_or_input()

        # Reflect any already-available ML signal
        if self.modality == "audio" and self.state.has_pending_audio():
            self.input_panel.set_ml_signal_available(self.state.pending_audio)
        elif self.modality == "image" and self.state.has_pending_image():
            self.input_panel.set_ml_signal_available(self.state.pending_image)


    def _init_ui(self):
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(24, 16, 24, 16)
        root_layout.setSpacing(14)

        # 1. Header (Breadcrumb, Title, Subtitle, Badges)
        root_layout.addLayout(self._build_header())

        # 2. Pipeline Indicator
        self.pipeline_bar = PipelineIndicatorBar(self)
        root_layout.addWidget(self.pipeline_bar)

        # 3. Compact Technical Summary Strip
        self.summary_strip = AnalysisSummaryStrip(self)
        root_layout.addWidget(self.summary_strip)

        # 4. Modality & Verification Test Switcher Bar
        root_layout.addLayout(self._build_controls_bar())

        # 5. Main Scrollable Visualization Area
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        self.scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        self.scroll_layout.setContentsMargins(0, 4, 8, 16)
        self.scroll_layout.setSpacing(14)

        # MATLAB Input Panel (always visible at top of scroll area)
        self.input_panel = MATLABInputPanel(self.modality, self.scroll_content)
        self.input_panel.run_pipeline_requested.connect(self._run_matlab_pipeline)
        self.input_panel.import_mat_requested.connect(self._import_mat_file)
        self.input_panel.use_ml_signal_requested.connect(self._use_ml_signal_for_dsp)
        self.scroll_layout.addWidget(self.input_panel)

        # Preprocessing Steps Panel (shown after ML classification or after MATLAB run)
        self.preprocessing_panel = PreprocessingStepsPanel(self.scroll_content)
        self.preprocessing_panel.setVisible(False)
        self.scroll_layout.addWidget(self.preprocessing_panel)

        # Modality Views
        self.audio_view = AudioDSPView(self.scroll_content)
        self.image_view = ImageDSPView(self.scroll_content)
        self.scroll_layout.addWidget(self.audio_view)
        self.scroll_layout.addWidget(self.image_view)

        # Empty State
        self.empty_state = DSPEmptyStateWidget(self.modality, self.scroll_content)
        self.empty_state.navigate_audio_ml.connect(lambda: self.navigate_requested.emit("audio_ml"))
        self.empty_state.navigate_image_ml.connect(lambda: self.navigate_requested.emit("image_ml"))
        self.empty_state.load_demo_requested.connect(self._load_default_sample)
        self.empty_state.setVisible(False)
        self.scroll_layout.addWidget(self.empty_state)

        # Diagnostic Section (for skipped & failed analyses)
        self.diag_section = DiagnosticSection(self.scroll_content)
        self.scroll_layout.addWidget(self.diag_section)

        self.scroll_area.setWidget(self.scroll_content)
        root_layout.addWidget(self.scroll_area, stretch=1)

    def _build_header(self) -> QHBoxLayout:
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(12)

        # Left Column: Breadcrumb + Title + Subtitle
        left_col = QVBoxLayout()
        left_col.setContentsMargins(0, 0, 0, 0)
        left_col.setSpacing(2)

        self.breadcrumb_lbl = QLabel(f"SPANDHAN  /  DSP ANALYSIS  /  {self.modality.upper()}")
        self.breadcrumb_lbl.setStyleSheet(
            f"color: {DSP_THEME.TEXT_MUTED}; font-size: 10px; font-weight: 700; letter-spacing: 1.5px;"
        )
        left_col.addWidget(self.breadcrumb_lbl)

        title_lbl = QLabel("DSP Analysis")
        title_lbl.setStyleSheet(
            f"color: {DSP_THEME.TEXT_PRIMARY}; font-size: 22px; font-weight: 800; letter-spacing: 0.5px;"
        )
        left_col.addWidget(title_lbl)

        sub_lbl = QLabel("Class-aware digital signal and image processing results generated by MATLAB.")
        sub_lbl.setStyleSheet(f"color: {DSP_THEME.TEXT_SECONDARY}; font-size: 12px;")
        left_col.addWidget(sub_lbl)

        header_layout.addLayout(left_col, stretch=1)

        # Right Column: Prominent Status Badges
        right_box = QHBoxLayout()
        right_box.setSpacing(8)

        # Modality Badge
        self.modality_badge = QLabel(self.modality.upper())
        self.modality_badge.setStyleSheet(f"""
            background-color: {DSP_THEME.BG_SURFACE_ALT};
            color: {DSP_THEME.CYAN};
            border: 1px solid rgba(56, 189, 248, 0.4);
            border-radius: 6px;
            padding: 4px 10px;
            font-size: 10px;
            font-weight: 800;
            letter-spacing: 1px;
        """)
        right_box.addWidget(self.modality_badge)

        # Signal Class Badge
        self.class_badge = QLabel("CHIRP (99.7%)")
        self.class_badge.setStyleSheet(f"""
            background-color: {DSP_THEME.BG_INPUT};
            color: {DSP_THEME.PERSIAN_GREEN};
            border: 1px solid {DSP_THEME.PERSIAN_GREEN};
            border-radius: 6px;
            padding: 4px 12px;
            font-size: 11px;
            font-weight: 800;
            letter-spacing: 0.5px;
        """)
        right_box.addWidget(self.class_badge)

        # DSP Status Badge
        self.status_badge = QLabel("DSP COMPLETE")
        self.status_badge.setStyleSheet(f"""
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 {DSP_THEME.INDIGO},
                stop:1 {DSP_THEME.PERSIAN_GREEN});
            color: #FFFFFF;
            border-radius: 6px;
            padding: 4px 12px;
            font-size: 11px;
            font-weight: 800;
            letter-spacing: 0.5px;
        """)
        right_box.addWidget(self.status_badge)

        header_layout.addLayout(right_box)
        return header_layout

    def _build_controls_bar(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        bar.setContentsMargins(0, 0, 0, 0)
        bar.setSpacing(10)

        # Modality Switcher Toggle Buttons
        lbl_mod = QLabel("Modality:")
        lbl_mod.setStyleSheet(f"color: {DSP_THEME.TEXT_MUTED}; font-size: 11px; font-weight: 600;")
        bar.addWidget(lbl_mod)

        self.btn_audio = QPushButton("Audio DSP")
        self.btn_audio.setCheckable(True)
        self.btn_audio.setChecked(self.modality == "audio")
        self.btn_audio.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_audio.clicked.connect(lambda: self.set_modality("audio"))
        bar.addWidget(self.btn_audio)

        self.btn_image = QPushButton("Image DSP")
        self.btn_image.setCheckable(True)
        self.btn_image.setChecked(self.modality == "image")
        self.btn_image.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_image.clicked.connect(lambda: self.set_modality("image"))
        bar.addWidget(self.btn_image)

        self._style_modality_buttons()

        bar.addSpacing(16)

        # Canonical Class Verification Selector
        lbl_cls = QLabel("Canonical Verification Case:")
        lbl_cls.setStyleSheet(f"color: {DSP_THEME.TEXT_MUTED}; font-size: 11px; font-weight: 600;")
        bar.addWidget(lbl_cls)

        self.class_combo = QComboBox()
        self.class_combo.addItems(["Chirp", "Sinusoidal", "White Noise", "Step", "Impulse"])
        self.class_combo.setStyleSheet(f"""
            QComboBox {{
                background-color: {DSP_THEME.BG_INPUT};
                color: {DSP_THEME.TEXT_PRIMARY};
                border: 1px solid {DSP_THEME.BORDER_SOLID};
                border-radius: 6px;
                padding: 4px 10px;
                font-size: 11px;
                font-weight: 600;
                min-width: 130px;
            }}
            QComboBox:hover {{
                border: 1px solid {DSP_THEME.BORDER_HOVER};
            }}
            QComboBox::drop-down {{
                border: none;
            }}
            QComboBox QAbstractItemView {{
                background-color: {DSP_THEME.BG_CARD};
                color: {DSP_THEME.TEXT_PRIMARY};
                selection-background-color: {DSP_THEME.INDIGO};
            }}
        """)
        self.class_combo.currentTextChanged.connect(self._on_class_combo_selected)
        bar.addWidget(self.class_combo)

        bar.addStretch()

        # Import external MATLAB .mat file button
        import_btn = QPushButton("Import MATLAB .mat Result...")
        import_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        import_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {DSP_THEME.BG_SURFACE_ALT};
                color: {DSP_THEME.TEXT_PRIMARY};
                border: 1px solid {DSP_THEME.BORDER_SOLID};
                border-radius: 6px;
                padding: 5px 14px;
                font-size: 11px;
                font-weight: 600;
            }}
            QPushButton:hover {{
                background-color: {DSP_THEME.BG_CARD_HOVER};
                border: 1px solid {DSP_THEME.BORDER_HOVER};
            }}
        """)
        import_btn.clicked.connect(self._import_mat_file)
        bar.addWidget(import_btn)

        return bar

    def _style_modality_buttons(self):
        style = f"""
            QPushButton {{
                background-color: {DSP_THEME.BG_INPUT};
                color: {DSP_THEME.TEXT_SECONDARY};
                border: 1px solid {DSP_THEME.BORDER_SOLID};
                border-radius: 6px;
                padding: 4px 14px;
                font-size: 11px;
                font-weight: 600;
            }}
            QPushButton:hover {{
                background-color: {DSP_THEME.BG_CARD_HOVER};
                color: {DSP_THEME.TEXT_PRIMARY};
            }}
            QPushButton:checked {{
                background-color: {DSP_THEME.INDIGO};
                color: #FFFFFF;
                border: 1px solid {DSP_THEME.BLUE_LIGHT};
            }}
        """
        self.btn_audio.setStyleSheet(style)
        self.btn_image.setStyleSheet(style)

    # -----------------------------------------------------------------------
    # Signal Connections
    # -----------------------------------------------------------------------

    def _connect_signals(self):
        self.state.audio_result_changed.connect(self._on_audio_result_changed)
        self.state.image_result_changed.connect(self._on_image_result_changed)
        self.state.pending_audio_signal_ready.connect(self._on_pending_audio_ready)
        self.state.pending_image_signal_ready.connect(self._on_pending_image_ready)
        if not self.locked_modality:
            self.state.modality_changed.connect(self.set_modality)

    def set_modality(self, modality: str):
        mod = modality.lower()
        if mod not in ("audio", "image"):
            return

        if self.locked_modality and mod != self.locked_modality:
            target_page = f"{mod}_dsp"
            self.navigate_requested.emit(target_page)
            self.btn_audio.setChecked(self.modality == "audio")
            self.btn_image.setChecked(self.modality == "image")
            return

        self.modality = mod
        self.breadcrumb_lbl.setText(f"SPANDHAN  /  DSP ANALYSIS  /  {self.modality.upper()}")
        self.modality_badge.setText(self.modality.upper())

        self.btn_audio.setChecked(self.modality == "audio")
        self.btn_image.setChecked(self.modality == "image")

        if self.modality == "audio":
            self.image_view.setVisible(False)
            res = self.state.audio_result
            if res:
                self.audio_view.setVisible(True)
                self.empty_state.setVisible(False)
                self.audio_view.set_result(res)
                self.summary_strip.update_audio(res)
                self._update_badges(res.signal_class, res.confidence, res.status)
                self._update_diagnostics(res.execution_log, res.failed_analyses)
                self.class_combo.blockSignals(True)
                idx = self.class_combo.findText(res.signal_class, Qt.MatchFlag.MatchFixedString)
                if idx >= 0:
                    self.class_combo.setCurrentIndex(idx)
                self.class_combo.blockSignals(False)
            else:
                self.audio_view.setVisible(False)
                self.empty_state.setVisible(True)
        else:
            self.audio_view.setVisible(False)
            res = self.state.image_result
            if res:
                self.image_view.setVisible(True)
                self.empty_state.setVisible(False)
                self.image_view.set_result(res)
                self.summary_strip.update_image(res)
                self._update_badges(res.signal_class, res.confidence, res.status)
                self._update_diagnostics(res.execution_log, res.failed_analyses)
                self.class_combo.blockSignals(True)
                idx = self.class_combo.findText(res.signal_class, Qt.MatchFlag.MatchFixedString)
                if idx >= 0:
                    self.class_combo.setCurrentIndex(idx)
                self.class_combo.blockSignals(False)
            else:
                self.image_view.setVisible(False)
                self.empty_state.setVisible(True)

    def _on_audio_result_changed(self, res: Optional[AudioDSPResult]):
        if self.modality == "audio" and res is not None:
            self.audio_view.setVisible(True)
            self.empty_state.setVisible(False)
            self.audio_view.set_result(res)
            self.summary_strip.update_audio(res)
            self._update_badges(res.signal_class, res.confidence, res.status)
            self._update_diagnostics(res.execution_log, res.failed_analyses)
            self.class_combo.blockSignals(True)
            idx = self.class_combo.findText(res.signal_class, Qt.MatchFlag.MatchFixedString)
            if idx >= 0:
                self.class_combo.setCurrentIndex(idx)
            self.class_combo.blockSignals(False)

    def _on_image_result_changed(self, res: Optional[ImageDSPResult]):
        if self.modality == "image" and res is not None:
            self.image_view.setVisible(True)
            self.empty_state.setVisible(False)
            self.image_view.set_result(res)
            self.summary_strip.update_image(res)
            self._update_badges(res.signal_class, res.confidence, res.status)
            self._update_diagnostics(res.execution_log, res.failed_analyses)
            self.class_combo.blockSignals(True)
            idx = self.class_combo.findText(res.signal_class, Qt.MatchFlag.MatchFixedString)
            if idx >= 0:
                self.class_combo.setCurrentIndex(idx)
            self.class_combo.blockSignals(False)

    def _update_badges(self, class_name: str, confidence: float, status: str):
        conf_str = f" ({confidence * 100.0:.1f}%)" if confidence > 0 else ""
        self.class_badge.setText(f"{class_name.upper()}{conf_str}")
        self.status_badge.setText(status.upper())

    def _update_diagnostics(self, logs: List[str], failed: List[str]):
        skipped = []
        failed_items = []

        for f in failed:
            failed_items.append((f, "Execution error in MATLAB DSP routine."))

        for log in logs:
            if "skipped" in log.lower():
                parts = log.split(":", 1)
                mod_name = parts[0].replace("skipped", "").strip()
                reason = parts[1].strip() if len(parts) > 1 else "Condition not met."
                skipped.append((mod_name, reason))

        self.diag_section.set_diagnostics(skipped, failed_items)

    def _on_class_combo_selected(self, class_name: str):
        if not class_name:
            return
        if self.modality == "audio":
            self.controller.load_canonical_audio_sample(class_name, switch_modality=False)
        else:
            self.controller.load_canonical_image_sample(class_name, switch_modality=False)

    def _load_default_sample(self):
        if self.modality == "audio":
            self.controller.load_canonical_audio_sample("chirp", switch_modality=False)
        else:
            self.controller.load_canonical_image_sample("sinusoidal", switch_modality=False)

    def _import_mat_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open MATLAB Pipeline Result", "", "MATLAB Files (*.mat);;All Files (*.*)"
        )
        if file_path:
            ok, msg = self.state.load_mat_file(file_path)
            if ok:
                # Show preprocessing steps for imported .mat as generic note
                steps = [("MATLAB Pipeline", "Result imported from existing .mat file")]
                fname = Path(file_path).name
                self.preprocessing_panel.set_steps(steps, source=fname)
                self.preprocessing_panel.setVisible(True)
            else:
                QMessageBox.warning(self, "Import Failed", msg)

    def _show_empty_or_input(self):
        """Show empty state when no result is loaded."""
        self.audio_view.setVisible(False)
        self.image_view.setVisible(False)
        self.empty_state.setVisible(True)

    # -----------------------------------------------------------------------
    # Pending ML Signal Handlers
    # -----------------------------------------------------------------------

    def _on_pending_audio_ready(self, info: dict):
        """Called when AudioMLPage pushes a classified signal into DSP state."""
        if self.modality == "audio":
            self.input_panel.set_ml_signal_available(info)
            # Auto-show preprocessing steps
            steps = info.get("preprocessing_steps", [])
            fname = Path(info.get("file_path", "")).name or "audio signal"
            cls   = info.get("predicted_class", "")
            conf  = info.get("confidence", 0.0)
            self.preprocessing_panel.set_steps(
                steps,
                source=f"Audio: {fname} → {cls} ({conf*100:.1f}%)"
            )
            self.preprocessing_panel.setVisible(True)

    def _on_pending_image_ready(self, info: dict):
        """Called when ImageMLPage pushes a classified image into DSP state."""
        if self.modality == "image":
            self.input_panel.set_ml_signal_available(info)
            steps = info.get("preprocessing_steps", [])
            fname = Path(info.get("file_path", "")).name or "image"
            cls   = info.get("predicted_class", "")
            conf  = info.get("confidence", 0.0)
            self.preprocessing_panel.set_steps(
                steps,
                source=f"Image: {fname} → {cls} ({conf*100:.1f}%)"
            )
            self.preprocessing_panel.setVisible(True)

    # -----------------------------------------------------------------------
    # MATLAB Pipeline Execution
    # -----------------------------------------------------------------------

    def _use_ml_signal_for_dsp(self):
        """
        Uses the already-classified signal from the ML page as input for the MATLAB
        DSP pipeline. This is the primary integration pathway.
        """
        if self.modality == "audio":
            info = self.state.pending_audio
            if not info:
                QMessageBox.information(
                    self,
                    "No Signal Available",
                    "No preprocessed audio signal from the ML workstation.\n"
                    "Please classify a signal on the Audio ML page first."
                )
                return
            self._run_matlab_pipeline("audio", info.get("file_path", ""))
        else:
            info = self.state.pending_image
            if not info:
                QMessageBox.information(
                    self,
                    "No Signal Available",
                    "No preprocessed image signal from the ML workstation.\n"
                    "Please classify an image on the Image ML page first."
                )
                return
            self._run_matlab_pipeline("image", info.get("file_path", ""))

    def _run_matlab_pipeline(self, modality: str, file_path: str):
        """
        Launches the MATLAB pipeline in a background thread.
        MATLAB computes all DSP analyses; Python only renders the result.
        """
        if not file_path:
            QMessageBox.warning(
                self, "No File Selected",
                "Please select an input file or classify a signal from the ML workstation first."
            )
            return

        self._matlab_runner = MATLABDSPRunner(self)
        self._matlab_runner.stage_changed.connect(self._on_matlab_stage)
        self._matlab_runner.finished.connect(self._on_matlab_finished)
        self._matlab_runner.failed.connect(self._on_matlab_failed)

        self.input_panel.set_running("Connecting to MATLAB…")

        if modality == "audio":
            self._matlab_runner.run_audio_pipeline(file_path)
        else:
            self._matlab_runner.run_image_pipeline(file_path)

    def _on_matlab_stage(self, stage: str):
        self.input_panel.set_running(stage)

    def _on_matlab_finished(self, mat_path: str):
        """MATLAB wrote a .mat file — load it into state and render."""
        self.input_panel.set_idle()
        ok, msg = self.state.load_mat_file(mat_path)
        if not ok:
            self.input_panel.set_error(f"Result load failed: {msg}")

    def _on_matlab_failed(self, error: str):
        self.input_panel.set_idle()
        self.input_panel.set_error(error)
        QMessageBox.critical(
            self, "MATLAB DSP Error",
            f"The MATLAB DSP pipeline failed:\n\n{error}\n\n"
            "Alternatives:\n"
            "• Run MATLAB manually and use 'Import .mat Result'\n"
            "• Load a canonical verification sample from the dropdown above"
        )


# ---------------------------------------------------------------------------
# Specialized Page Subclasses for Direct Sidebar Navigation
# ---------------------------------------------------------------------------

class AudioDSPPage(DSPPage):
    """Specialized wrapper launching directly into Audio DSP modality."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(initial_modality="audio", locked_modality="audio", parent=parent)


class ImageDSPPage(DSPPage):
    """Specialized wrapper launching directly into Image DSP modality."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(initial_modality="image", locked_modality="image", parent=parent)
