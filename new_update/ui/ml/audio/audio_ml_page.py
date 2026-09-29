"""
SPANDHAN — Audio ML Analysis Page
=================================
The primary AI + Signal Processing workstation page for 1D Audio Signal Classification.
Features:
  - Subtle breadcrumbs: SPANDHAN / MACHINE LEARNING / AUDIO
  - Drag-and-drop input card with file inspection & sample selector
  - Interactive 1D Signal Preview with zoom, pan, hover cursor, and empty placeholder
  - Large gradient "✦ ANALYZE SIGNAL" button with sequential multi-stage progress
  - ML Classification Card with large typography and confidence bar
  - Full Class Probability Distribution (Impulse, Sinusoidal, White Noise, Step, Chirp)
  - Signal Information Card with measured audio properties (sample rate, duration, samples, etc.)
  - Collapsible Signal Characteristics Card (RMS, Std Dev, Peak, Energy, Crest Factor, ZCR,
    Dominant Freq, Spectral Centroid, Spectral Bandwidth, Spectral Flatness, 3-dB Bandwidth)
  - Model Information Card with pre-trained Audio Classifier specs and status
  - Signal Class Reference Card
  - Audio ML Pipeline Architecture Indicator
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QColor, QPainter, QFont
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFileDialog,
    QFrame,
    QScrollArea,
    QComboBox,
    QProgressBar,
    QSizePolicy,
)

from ui.styles.theme import COLORS
from ui.components.cards.metric_card import CardContainer, MetricGrid
from ui.components.cards.probability_bar import ProbabilityDistributionWidget
from ui.ml.audio.waveform_preview_widget import AudioSignalPreviewWidget
from ui.ml.audio.pipeline_indicator_widget import AudioPipelineIndicatorWidget
from ui.services.audio_analysis_service import (
    AudioAnalysisWorker,
    load_and_inspect_audio,
    compute_audio_characteristics,
)
from ui.dsp.dsp_state import get_dsp_state
from ui.dsp.matlab_runner import AUDIO_PREPROCESSING_STEPS


class AudioDropUploadArea(QFrame):
    """Interactive drag-and-drop zone with dashed borders and hover states for audio files."""

    file_dropped = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setMinimumHeight(125)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._is_drag_over = False

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls and urls[0].toLocalFile():
                self._is_drag_over = True
                self.update()
                event.acceptProposedAction()
                return
        event.ignore()

    def dragLeaveEvent(self, event):
        self._is_drag_over = False
        self.update()

    def dropEvent(self, event: QDropEvent):
        self._is_drag_over = False
        self.update()
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if file_path:
                self.file_dropped.emit(file_path)
                event.acceptProposedAction()
                return

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = self.rect().adjusted(2, 2, -2, -2)

        # Background
        bg_color = QColor(COLORS.BG_INPUT) if not self._is_drag_over else QColor(17, 34, 59)
        painter.fillRect(rect, bg_color)

        # Dashed border
        border_color = QColor(COLORS.PERSIAN_GREEN) if self._is_drag_over else QColor(59, 130, 246, 75)
        border_pen = painter.pen()
        border_pen.setColor(border_color)
        border_pen.setWidth(1)
        border_pen.setStyle(Qt.PenStyle.DashLine)
        border_pen.setDashPattern([6, 5])
        painter.setPen(border_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(rect, 8, 8)


class AudioMLPage(QWidget):
    """Complete Audio ML Page for 1D Acoustic Signal Classification."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self._current_file: Optional[str] = None
        self._current_signal: Optional[np.ndarray] = None
        self._current_sr: int = 16000
        self._worker_thread: Optional[QThread] = None
        self._worker: Optional[AudioAnalysisWorker] = None

        self._init_ui()
        self._load_available_samples()

    def _init_ui(self):
        # Master layout containing a smooth scroll area
        master_layout = QVBoxLayout(self)
        master_layout.setContentsMargins(0, 0, 0, 0)
        master_layout.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        container = QWidget()
        container.setObjectName("AudioMLContent")
        container.setStyleSheet(f"QWidget#AudioMLContent {{ background-color: {COLORS.BG_PRIMARY}; }}")
        self.content_layout = QVBoxLayout(container)
        self.content_layout.setContentsMargins(28, 20, 28, 28)
        self.content_layout.setSpacing(18)

        # ---------------------------------------------------------------------
        # 1. Subtle Breadcrumbs & Page Header
        # ---------------------------------------------------------------------
        self._build_header()

        # ---------------------------------------------------------------------
        # 2. MAIN TWO-COLUMN RESPONSIVE LAYOUT
        # ---------------------------------------------------------------------
        cols_layout = QHBoxLayout()
        cols_layout.setSpacing(20)

        # =====================================================================
        # LEFT / PRIMARY COLUMN
        #   - INPUT SIGNAL
        #   - SIGNAL PREVIEW
        #   - ANALYZE SIGNAL
        #   - CLASS PROBABILITY DISTRIBUTION
        # =====================================================================
        left_col = QVBoxLayout()
        left_col.setSpacing(16)

        # 1. Input Signal Card
        self.input_card = self._build_input_signal_card()
        left_col.addWidget(self.input_card)

        # 2. Signal Preview Card (Waveform)
        self.preview_widget = AudioSignalPreviewWidget(self)
        self.preview_widget.browse_requested.connect(self._open_file_dialog)
        left_col.addWidget(self.preview_widget)

        # 3. Analyze Signal Action Bar
        self._build_analyze_action_bar(left_col)

        # 4. Class Probability Distribution
        self.prob_card = self._build_probability_card()
        left_col.addWidget(self.prob_card)

        cols_layout.addLayout(left_col, stretch=6)

        # =====================================================================
        # RIGHT / SECONDARY COLUMN
        #   - ML CLASSIFICATION
        #   - MODEL STATUS
        #   - SIGNAL INFORMATION (METADATA)
        #   - SIGNAL CHARACTERISTICS
        # =====================================================================
        right_col = QVBoxLayout()
        right_col.setSpacing(16)

        # 1. ML Classification Card
        self.classification_card = self._build_classification_card()
        right_col.addWidget(self.classification_card)

        # 2. Model Status Card
        self.model_status_card = self._build_model_status_card()
        right_col.addWidget(self.model_status_card)

        # 3. Signal Information Card (Metadata)
        self.signal_info_card = self._build_signal_info_card()
        right_col.addWidget(self.signal_info_card)

        # 4. Signal Characteristics Card (Collapsible, expanding to fill vertical space)
        self.characteristics_card = self._build_characteristics_card()
        self.characteristics_card.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        right_col.addWidget(self.characteristics_card, stretch=1)

        cols_layout.addLayout(right_col, stretch=4)

        self.content_layout.addLayout(cols_layout)

        # ---------------------------------------------------------------------
        # 3. BOTTOM: SIGNAL CLASS REFERENCE & AUDIO ML PIPELINE
        # ---------------------------------------------------------------------
        self.ref_card = self._build_reference_card()
        self.content_layout.addWidget(self.ref_card)

        self.pipeline_indicator = AudioPipelineIndicatorWidget(self)
        self.content_layout.addWidget(self.pipeline_indicator)

        scroll.setWidget(container)
        master_layout.addWidget(scroll)

    # -------------------------------------------------------------------------
    # UI Component Builders
    # -------------------------------------------------------------------------
    def _build_header(self):
        header_box = QVBoxLayout()
        header_box.setSpacing(4)

        # Subtle Breadcrumb: SPANDHAN / MACHINE LEARNING / AUDIO
        bc_layout = QHBoxLayout()
        bc_layout.setSpacing(6)
        bc_root = QLabel("SPANDHAN")
        bc_root.setStyleSheet(f"color: {COLORS.TEXT_MUTED}; font-size: 10px; font-weight: 700; letter-spacing: 1px;")
        bc_sep1 = QLabel(" / ")
        bc_sep1.setStyleSheet(f"color: #334155; font-size: 10px;")
        bc_cat = QLabel("MACHINE LEARNING")
        bc_cat.setStyleSheet(f"color: {COLORS.TEXT_MUTED}; font-size: 10px; font-weight: 600; letter-spacing: 0.8px;")
        bc_sep2 = QLabel(" / ")
        bc_sep2.setStyleSheet(f"color: #334155; font-size: 10px;")
        bc_leaf = QLabel("AUDIO")
        bc_leaf.setStyleSheet(f"color: {COLORS.TEXT_HIGHLIGHT}; font-size: 10px; font-weight: 700; letter-spacing: 0.8px;")

        bc_layout.addWidget(bc_root)
        bc_layout.addWidget(bc_sep1)
        bc_layout.addWidget(bc_cat)
        bc_layout.addWidget(bc_sep2)
        bc_layout.addWidget(bc_leaf)
        bc_layout.addStretch()
        header_box.addLayout(bc_layout)

        # Title & Subtitle
        title_lbl = QLabel("Audio Signal Classification")
        title_lbl.setStyleSheet(
            f"color: {COLORS.TEXT_PRIMARY}; font-size: 22px; font-weight: 800; "
            f"letter-spacing: -0.3px; margin-top: 2px;"
        )
        sub_lbl = QLabel("AI-assisted identification of fundamental signal types")
        sub_lbl.setStyleSheet(f"color: {COLORS.TEXT_SECONDARY}; font-size: 12px; margin-top: 1px;")

        header_box.addWidget(title_lbl)
        header_box.addWidget(sub_lbl)
        self.content_layout.addLayout(header_box)

    def _build_input_signal_card(self) -> CardContainer:
        card = CardContainer(title="Input Signal", subtitle="Select or drop 1D acoustic audio signal")

        # Drop Area
        self.drop_area = AudioDropUploadArea(self)
        self.drop_area.file_dropped.connect(self.load_file)

        drop_inner_layout = QVBoxLayout(self.drop_area)
        drop_inner_layout.setContentsMargins(12, 12, 12, 12)
        drop_inner_layout.setSpacing(4)
        drop_inner_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        icon_lbl = QLabel("〰")
        icon_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon_lbl.setStyleSheet(f"color: {COLORS.BLUE_LIGHT}; font-size: 22px;")

        self.drop_text_main = QLabel("Drop audio file here")
        self.drop_text_main.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.drop_text_main.setStyleSheet(f"color: {COLORS.TEXT_PRIMARY}; font-size: 11px; font-weight: 600;")

        self.drop_text_or = QLabel("or browse files")
        self.drop_text_or.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.drop_text_or.setStyleSheet(f"color: {COLORS.TEXT_MUTED}; font-size: 10px;")

        fmt_lbl = QLabel("WAV • MP3 • FLAC • M4A")
        fmt_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        fmt_lbl.setStyleSheet(f"color: {COLORS.TEXT_MUTED}; font-size: 9px; letter-spacing: 0.5px;")

        drop_inner_layout.addWidget(icon_lbl)
        drop_inner_layout.addWidget(self.drop_text_main)
        drop_inner_layout.addWidget(self.drop_text_or)
        drop_inner_layout.addWidget(fmt_lbl)

        card.add_widget(self.drop_area)

        # File actions: Browse & Change File
        btn_box = QHBoxLayout()
        btn_box.setSpacing(8)

        self.browse_btn = QPushButton("Browse")
        self.browse_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.browse_btn.setStyleSheet(
            f"QPushButton {{ background-color: {COLORS.BG_SURFACE_ALT}; color: {COLORS.TEXT_PRIMARY}; "
            f"border: 1px solid {COLORS.BORDER_SUBTLE}; border-radius: 6px; padding: 7px 14px; font-size: 11px; font-weight: 600; }} "
            f"QPushButton:hover {{ background-color: {COLORS.INDIGO}; border-color: {COLORS.BLUE}; }}"
        )
        self.browse_btn.clicked.connect(self._open_file_dialog)
        btn_box.addWidget(self.browse_btn)

        self.change_btn = QPushButton("Change File")
        self.change_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.change_btn.setVisible(False)
        self.change_btn.setStyleSheet(
            f"QPushButton {{ background-color: transparent; color: {COLORS.TEXT_MUTED}; "
            f"border: 1px solid {COLORS.BORDER_SUBTLE}; border-radius: 6px; padding: 7px 10px; font-size: 11px; }} "
            f"QPushButton:hover {{ color: {COLORS.TEXT_PRIMARY}; border-color: {COLORS.BORDER_HOVER}; }}"
        )
        self.change_btn.clicked.connect(self._open_file_dialog)
        btn_box.addWidget(self.change_btn)

        card.add_layout(btn_box)

        # Quick Sample Selector (Scientific convenience)
        sample_box = QHBoxLayout()
        sample_box.setSpacing(6)
        sample_lbl = QLabel("Sample:")
        sample_lbl.setStyleSheet(f"color: {COLORS.TEXT_MUTED}; font-size: 10px;")
        self.sample_combo = QComboBox()
        self.sample_combo.setStyleSheet(
            f"QComboBox {{ background-color: {COLORS.BG_INPUT}; color: {COLORS.TEXT_PRIMARY}; "
            f"border: 1px solid {COLORS.BORDER_SUBTLE}; border-radius: 5px; padding: 3px 6px; font-size: 10px; }} "
            f"QComboBox::drop-down {{ border: none; }} "
            f"QComboBox QAbstractItemView {{ background-color: {COLORS.BG_SECONDARY}; color: {COLORS.TEXT_PRIMARY}; selection-background-color: {COLORS.INDIGO}; }}"
        )
        self.sample_combo.currentIndexChanged.connect(self._on_sample_selected)
        sample_box.addWidget(sample_lbl)
        sample_box.addWidget(self.sample_combo, stretch=1)
        card.add_layout(sample_box)

        # Selected File Metadata Capsule (Populated after selection)
        self.file_capsule = QFrame()
        self.file_capsule.setVisible(False)
        self.file_capsule.setStyleSheet(
            f"background-color: #07111F; border: 1px solid {COLORS.BORDER_SUBTLE}; border-radius: 6px; padding: 6px 10px;"
        )
        fc_layout = QVBoxLayout(self.file_capsule)
        fc_layout.setContentsMargins(0, 0, 0, 0)
        fc_layout.setSpacing(3)

        self.capsule_name_lbl = QLabel("✓ sinusoidal_0003.wav")
        self.capsule_name_lbl.setStyleSheet(
            f"color: {COLORS.PERSIAN_GREEN}; font-size: 11px; font-weight: 700;"
        )
        self.capsule_meta_lbl = QLabel("WAV • 16 kHz • Mono")
        self.capsule_meta_lbl.setStyleSheet(f"color: {COLORS.TEXT_MUTED}; font-size: 10px;")

        fc_layout.addWidget(self.capsule_name_lbl)
        fc_layout.addWidget(self.capsule_meta_lbl)
        card.add_widget(self.file_capsule)

        return card

    def _build_analyze_action_bar(self, parent_layout: QVBoxLayout):
        action_layout = QVBoxLayout()
        action_layout.setSpacing(6)

        # Large Gradient Primary Action Button
        self.analyze_btn = QPushButton("✦  ANALYZE SIGNAL")
        self.analyze_btn.setProperty("class", "PrimaryButton")
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.setFixedHeight(44)
        self.analyze_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.analyze_btn.clicked.connect(self.start_analysis)
        action_layout.addWidget(self.analyze_btn)

        # Sequential Processing Status Indicator
        self.stage_status_lbl = QLabel("Awaiting input signal selection...")
        self.stage_status_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.stage_status_lbl.setStyleSheet(
            f"color: {COLORS.TEXT_MUTED}; font-size: 11px; font-weight: 500; min-height: 18px;"
        )
        action_layout.addWidget(self.stage_status_lbl)

        # Technical details accordion for errors (hidden by default)
        self.error_details_lbl = QLabel("")
        self.error_details_lbl.setWordWrap(True)
        self.error_details_lbl.setVisible(False)
        self.error_details_lbl.setStyleSheet(
            f"color: {COLORS.STATUS_ERROR}; background-color: #1A0B10; border: 1px solid #7F1D1D; "
            f"border-radius: 6px; padding: 8px; font-size: 10px; font-family: 'Consolas', monospace;"
        )
        action_layout.addWidget(self.error_details_lbl)

        parent_layout.addLayout(action_layout)

    def _build_classification_card(self) -> CardContainer:
        card = CardContainer(title="ML Classification", subtitle="AI-assisted signal type identification")

        body_box = QVBoxLayout()
        body_box.setSpacing(10)

        pred_title = QLabel("PREDICTED SIGNAL")
        pred_title.setStyleSheet(f"color: {COLORS.TEXT_MUTED}; font-size: 10px; font-weight: 700; letter-spacing: 1.2px;")
        body_box.addWidget(pred_title)

        # Big Predicted Class Typography (Visually Dominant)
        self.class_display_lbl = QLabel("AWAITING INFERENCE")
        self.class_display_lbl.setStyleSheet(
            f"color: {COLORS.TEXT_PRIMARY}; font-size: 26px; font-weight: 800; "
            f"letter-spacing: 1.0px; padding: 4px 0px;"
        )
        body_box.addWidget(self.class_display_lbl)

        # Model Confidence Section (Strictly described as Confidence, not Accuracy)
        conf_header_box = QHBoxLayout()
        conf_header_box.setSpacing(6)
        conf_title = QLabel("MODEL CONFIDENCE")
        conf_title.setStyleSheet(f"color: {COLORS.TEXT_MUTED}; font-size: 10px; font-weight: 700; letter-spacing: 1.0px;")
        self.conf_num_lbl = QLabel("—")
        self.conf_num_lbl.setStyleSheet(
            f"color: {COLORS.PERSIAN_GREEN}; font-size: 14px; font-weight: 700; font-family: 'Consolas', monospace;"
        )
        conf_header_box.addWidget(conf_title)
        conf_header_box.addStretch()
        conf_header_box.addWidget(self.conf_num_lbl)
        body_box.addLayout(conf_header_box)

        # Horizontal Confidence Progress Bar (SPANDHAN Gradient)
        self.conf_bar = QProgressBar()
        self.conf_bar.setRange(0, 1000)
        self.conf_bar.setValue(0)
        self.conf_bar.setTextVisible(False)
        self.conf_bar.setFixedHeight(8)
        self.conf_bar.setStyleSheet(f"""
            QProgressBar {{
                background-color: {COLORS.BG_INPUT};
                border: 1px solid {COLORS.BORDER_SUBTLE};
                border-radius: 4px;
            }}
            QProgressBar::chunk {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {COLORS.INDIGO},
                    stop:0.6 {COLORS.BLUE},
                    stop:1 {COLORS.PERSIAN_GREEN});
                border-radius: 4px;
            }}
        """)
        body_box.addWidget(self.conf_bar)

        card.add_layout(body_box)
        return card

    def _build_model_status_card(self) -> CardContainer:
        card = CardContainer(title="Model Status", subtitle="Inference runtime specification")

        self.model_grid = MetricGrid(self)
        self.model_grid.add_metric("status", "Status", "● Ready")
        self.model_grid.add_metric("model_name", "Model", "Audio Signal Classifier")
        self.model_grid.add_metric("arch", "Algorithm", "Random Forest Pipeline")
        self.model_grid.add_metric("features", "Feature Vector", "35 DSP Engineered Features")
        self.model_grid.add_metric("inference", "Inference", "Pre-trained")
        self.model_grid.add_metric("classes", "Classes", "5")

        card.add_widget(self.model_grid)
        return card

    def _build_signal_info_card(self) -> CardContainer:
        card = CardContainer(title="Signal Information", subtitle="Measured audio file metadata")

        self.info_grid = MetricGrid(self)
        self.info_grid.add_metric("sample_rate", "Sample Rate", "—")
        self.info_grid.add_metric("duration", "Duration", "—")
        self.info_grid.add_metric("samples", "Samples", "—")
        self.info_grid.add_metric("channels", "Channels", "—")
        self.info_grid.add_metric("format", "Format", "—")
        self.info_grid.add_metric("file_size", "File Size", "—")
        self.info_grid.add_metric("peak_amp", "Peak Amplitude", "—")

        card.add_widget(self.info_grid)
        return card

    def _build_probability_card(self) -> CardContainer:
        card = CardContainer(
            title="Class Probability Distribution",
            subtitle="Normalized multiclass probabilities over all 5 supported classes",
        )
        self.prob_widget = ProbabilityDistributionWidget(self)
        card.add_widget(self.prob_widget)
        return card

    def _build_characteristics_card(self) -> CardContainer:
        card = CardContainer(
            title="Signal Characteristics",
            subtitle="Extracted time and frequency domain acoustic metrics",
            collapsible=True,
        )
        card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # 1. Primary Metrics Grid (Two balanced 6-metric columns)
        grid_container = QWidget()
        h_layout = QHBoxLayout(grid_container)
        h_layout.setContentsMargins(0, 0, 0, 0)
        h_layout.setSpacing(20)

        # Left Column Metrics: RMS, Std Dev, Peak, Energy, Crest Factor, ZCR
        self.char_grid_left = MetricGrid(self)
        self.char_grid_left.add_metric("rms", "RMS", "—")
        self.char_grid_left.add_metric("std", "Standard Deviation", "—")
        self.char_grid_left.add_metric("peak", "Peak Amplitude", "—")
        self.char_grid_left.add_metric("energy", "Energy", "—")
        self.char_grid_left.add_metric("crest_factor", "Peak-to-RMS Ratio", "—")
        self.char_grid_left.add_metric("zcr", "Zero Crossing Rate", "—")
        h_layout.addWidget(self.char_grid_left)

        # Right Column Metrics: Dominant Freq, Spectral Centroid, Bandwidth, Flatness, 3-dB Bandwidth, Spectral Rolloff
        self.char_grid_right = MetricGrid(self)
        self.char_grid_right.add_metric("dom_freq", "Dominant Frequency", "—", "Hz")
        self.char_grid_right.add_metric("spectral_centroid", "Spectral Centroid", "—", "Hz")
        self.char_grid_right.add_metric("spectral_bandwidth", "Spectral Bandwidth", "—", "Hz")
        self.char_grid_right.add_metric("spectral_flatness", "Spectral Flatness", "—")
        self.char_grid_right.add_metric("bandwidth_3db", "3-dB Bandwidth", "—", "Hz")
        self.char_grid_right.add_metric("rolloff", "Spectral Rolloff (85%)", "—", "Hz")
        h_layout.addWidget(self.char_grid_right)

        card.add_widget(grid_container)

        # Subtle separator
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet(f"background-color: {COLORS.BORDER_SUBTLE}; max-height: 1px; margin: 6px 0px 4px 0px;")
        card.add_widget(sep)

        # 2. Acoustic Spectral Energy Profile (Covers expanded lower area)
        profile_frame = QFrame()
        profile_frame.setStyleSheet(
            f"background-color: {COLORS.BG_INPUT}; border: 1px solid {COLORS.BORDER_SUBTLE}; "
            f"border-radius: 6px; padding: 10px 12px;"
        )
        pf_layout = QVBoxLayout(profile_frame)
        pf_layout.setContentsMargins(0, 0, 0, 0)
        pf_layout.setSpacing(6)

        pf_title_row = QHBoxLayout()
        pf_title = QLabel("ACOUSTIC SPECTRAL DISTRIBUTION")
        pf_title.setStyleSheet(f"color: {COLORS.TEXT_MUTED}; font-size: 9px; font-weight: 700; letter-spacing: 0.8px;")
        self.pf_profile_summary = QLabel("Awaiting signal analysis")
        self.pf_profile_summary.setStyleSheet(f"color: {COLORS.TEXT_MUTED}; font-size: 9px; font-family: 'Consolas', monospace;")
        pf_title_row.addWidget(pf_title)
        pf_title_row.addStretch()
        pf_title_row.addWidget(self.pf_profile_summary)
        pf_layout.addLayout(pf_title_row)

        # 4 Mini Band Energy Bars
        self.band_bars: dict[str, tuple[QLabel, QProgressBar]] = {}
        bands_spec = [
            ("bass", "Sub-Bass / Bass (20 – 250 Hz)", COLORS.INDIGO, COLORS.INDIGO_LIGHT),
            ("lowmid", "Low-Midrange (250 – 1,000 Hz)", COLORS.INDIGO_LIGHT, COLORS.BLUE),
            ("mid", "Midrange / Presence (1 – 4 kHz)", COLORS.BLUE, COLORS.CYAN),
            ("high", "High Frequency (4 – 8 kHz)", COLORS.CYAN, COLORS.PERSIAN_GREEN),
        ]

        for b_key, b_label, c1, c2 in bands_spec:
            b_row = QHBoxLayout()
            b_row.setSpacing(8)

            lbl = QLabel(b_label)
            lbl.setStyleSheet(f"color: {COLORS.TEXT_SECONDARY}; font-size: 10px; min-width: 150px;")

            bar = QProgressBar()
            bar.setRange(0, 1000)
            bar.setValue(0)
            bar.setTextVisible(False)
            bar.setFixedHeight(6)
            bar.setStyleSheet(f"""
                QProgressBar {{
                    background-color: {COLORS.BG_CARD};
                    border: 1px solid {COLORS.BORDER_SUBTLE};
                    border-radius: 3px;
                }}
                QProgressBar::chunk {{
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                        stop:0 {c1}, stop:1 {c2});
                    border-radius: 3px;
                }}
            """)

            val_lbl = QLabel("—")
            val_lbl.setFixedWidth(55)
            val_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            val_lbl.setStyleSheet(f"color: {COLORS.TEXT_MUTED}; font-size: 9px; font-family: 'Consolas', monospace;")

            b_row.addWidget(lbl)
            b_row.addWidget(bar, stretch=1)
            b_row.addWidget(val_lbl)
            pf_layout.addLayout(b_row)

            self.band_bars[b_key] = (val_lbl, bar)

        card.add_widget(profile_frame)
        return card

    def _build_reference_card(self) -> CardContainer:
        card = CardContainer(title="Signal Classes", subtitle="Taxonomy of fundamental audio signal types")

        ref_box = QHBoxLayout()
        ref_box.setSpacing(10)

        classes_info = [
            ("IMPULSE", "🗲", "Localized transient Dirac pulse with instant energy concentration"),
            ("SINUSOIDAL", "∿", "Periodic single-frequency tone with pure spectral line"),
            ("WHITE NOISE", "░", "Broadband stochastic signal with uniform power across spectrum"),
            ("STEP", "⌐", "Abrupt persistent transition / DC level shift"),
            ("CHIRP", "⟿", "Time-varying frequency sweep across acoustic bandwidth"),
        ]

        for title, glyph, desc in classes_info:
            panel = QFrame()
            panel.setStyleSheet(
                f"background-color: #07111F; border: 1px solid {COLORS.BORDER_SUBTLE}; "
                f"border-radius: 6px; padding: 8px 10px;"
            )
            p_layout = QVBoxLayout(panel)
            p_layout.setContentsMargins(0, 0, 0, 0)
            p_layout.setSpacing(3)

            top_h = QHBoxLayout()
            top_h.setSpacing(4)
            g_lbl = QLabel(glyph)
            g_lbl.setStyleSheet(f"color: {COLORS.CYAN}; font-size: 12px; font-weight: bold;")
            t_lbl = QLabel(title)
            t_lbl.setStyleSheet(f"color: {COLORS.TEXT_HIGHLIGHT}; font-size: 10px; font-weight: 700;")
            top_h.addWidget(g_lbl)
            top_h.addWidget(t_lbl)
            top_h.addStretch()

            d_lbl = QLabel(desc)
            d_lbl.setWordWrap(True)
            d_lbl.setStyleSheet(f"color: {COLORS.TEXT_MUTED}; font-size: 9px;")

            p_layout.addLayout(top_h)
            p_layout.addWidget(d_lbl)
            ref_box.addWidget(panel, stretch=1)

        card.add_layout(ref_box)
        return card

    # -------------------------------------------------------------------------
    # Interactions & Logic
    # -------------------------------------------------------------------------
    def _load_available_samples(self):
        """Populate sample dropdown with real audio files from datasets."""
        root = Path(__file__).resolve().parents[3]
        samples_dir = root / "datasets" / "audio"

        self.sample_combo.blockSignals(True)
        self.sample_combo.addItem("Select sample...", "")

        if samples_dir.exists():
            for class_name in ["impulse", "sinusoidal", "white_noise", "step", "chirp"]:
                c_dir = samples_dir / class_name
                if c_dir.exists():
                    wavs = list(c_dir.glob("*.wav"))
                    if wavs:
                        # Pick a representative sample (e.g. sinusoidal_0003.wav if present)
                        chosen = wavs[0]
                        for w in wavs:
                            if "0003" in w.name:
                                chosen = w
                                break
                        display_name = f"{class_name.replace('_', ' ').capitalize()} ({chosen.name})"
                        self.sample_combo.addItem(display_name, str(chosen))

        self.sample_combo.blockSignals(False)

    def _on_sample_selected(self, index: int):
        path = self.sample_combo.currentData()
        if path and os.path.exists(path):
            self.load_file(path)

    def _open_file_dialog(self):
        root = Path(__file__).resolve().parents[3]
        initial_dir = str(root / "datasets" / "audio")
        if not os.path.exists(initial_dir):
            initial_dir = str(root)

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Audio Signal",
            initial_dir,
            "Audio Files (*.wav *.flac *.mp3 *.m4a *.ogg);;All Files (*.*)",
        )
        if file_path:
            self.load_file(file_path)

    def load_file(self, file_path: str):
        """Load audio file, update UI metadata and waveform preview, and enable analysis."""
        try:
            data, sr, metadata = load_and_inspect_audio(file_path)

            self._current_file = file_path
            self._current_signal = data
            self._current_sr = sr

            # Update Waveform Preview
            self.preview_widget.set_signal(data, sr)

            # Update File Capsule
            self.file_capsule.setVisible(True)
            self.capsule_name_lbl.setText(f"✓ {metadata['filename']}")
            self.capsule_meta_lbl.setText(
                f"{metadata['format']} • {metadata['sample_rate_str']} • {metadata['channels_str']}"
            )
            self.change_btn.setVisible(True)

            # Update Signal Info Card (Metadata)
            self.info_grid.set_value("sample_rate", metadata["sample_rate_str"])
            self.info_grid.set_value("duration", metadata["duration_str"])
            self.info_grid.set_value("samples", metadata["samples_str"])
            self.info_grid.set_value("channels", metadata["channels_str"])
            self.info_grid.set_value("format", metadata["format"])
            self.info_grid.set_value("file_size", metadata["file_size_kb"])
            self.info_grid.set_value("peak_amp", f"{metadata['peak_amplitude']:.4f}")

            # Reset previous prediction outputs
            self.class_display_lbl.setText("READY FOR ANALYSIS")
            self.class_display_lbl.setStyleSheet(
                f"color: {COLORS.TEXT_PRIMARY}; font-size: 24px; font-weight: 800; padding: 4px 0px;"
            )
            self.conf_num_lbl.setText("—")
            self.conf_bar.setValue(0)
            self.prob_widget.reset()
            self.char_grid_left.reset_all()
            self.char_grid_right.reset_all()
            self.pipeline_indicator.set_stage_progress(1)  # Stage 1: Input Loaded

            # Hide any previous error
            self.error_details_lbl.setVisible(False)

            # Enable Analyze button
            self.analyze_btn.setEnabled(True)
            self.stage_status_lbl.setText(f"Signal loaded: {metadata['filename']}. Ready for ML classification.")
            self.stage_status_lbl.setStyleSheet(f"color: {COLORS.PERSIAN_GREEN}; font-size: 11px;")

        except Exception as e:
            self.stage_status_lbl.setText(f"Error loading audio file: {e}")
            self.stage_status_lbl.setStyleSheet(f"color: {COLORS.STATUS_ERROR}; font-size: 11px;")
            self.error_details_lbl.setText(f"Failed to load '{file_path}':\n{str(e)}")
            self.error_details_lbl.setVisible(True)

    def start_analysis(self):
        """Trigger asynchronous ML inference in background worker thread."""
        if not self._current_file:
            return

        self.analyze_btn.setEnabled(False)
        self.browse_btn.setEnabled(False)
        self.change_btn.setEnabled(False)
        self.sample_combo.setEnabled(False)
        self.error_details_lbl.setVisible(False)

        # Update model status card to loading
        self.model_grid.set_value("status", "○ Loading...")

        # Setup worker thread
        self._worker_thread = QThread()
        self._worker = AudioAnalysisWorker(self._current_file)
        self._worker.moveToThread(self._worker_thread)

        self._worker_thread.started.connect(self._worker.run)
        self._worker.stage_changed.connect(self._on_stage_changed)
        self._worker.analysis_finished.connect(self._on_analysis_finished)
        self._worker.analysis_failed.connect(self._on_analysis_failed)

        # Thread cleanup
        self._worker.analysis_finished.connect(self._worker_thread.quit)
        self._worker.analysis_failed.connect(self._worker_thread.quit)
        self._worker_thread.finished.connect(self._worker.deleteLater)
        self._worker_thread.finished.connect(self._worker_thread.deleteLater)

        self._worker_thread.start()

    def _on_stage_changed(self, stage_text: str):
        self.stage_status_lbl.setText(f"✦ {stage_text}")
        self.stage_status_lbl.setStyleSheet(f"color: {COLORS.STATUS_PROCESSING}; font-size: 11px; font-weight: 600;")

        # Update pipeline stage indicator
        if "Loading" in stage_text:
            self.pipeline_indicator.set_stage_progress(1)
        elif "Preprocessing" in stage_text:
            self.pipeline_indicator.set_stage_progress(2)
        elif "ML inference" in stage_text:
            self.pipeline_indicator.set_stage_progress(4)
        elif "classification results" in stage_text:
            self.pipeline_indicator.set_stage_progress(3)

    def _on_analysis_finished(self, results: Dict[str, Any]):
        prediction = results["prediction"]
        characteristics = results["characteristics"]

        # 1. Update ML Classification Card
        pred_class = str(prediction.get("class", "UNKNOWN")).upper()
        confidence = float(prediction.get("confidence", 0.0))

        self.class_display_lbl.setText(pred_class)
        self.class_display_lbl.setStyleSheet(
            f"color: {COLORS.PERSIAN_GREEN}; font-size: 28px; font-weight: 800; "
            f"letter-spacing: 1.5px; padding: 4px 0px;"
        )

        conf_pct = confidence * 100.0
        self.conf_num_lbl.setText(f"{conf_pct:.2f}%")
        self.conf_bar.setValue(int(confidence * 1000.0))

        # 2. Update Class Probabilities Distribution
        probs = prediction.get("probabilities", {})
        self.prob_widget.set_probabilities(probs)

        # 3. Update Model Status Card
        self.model_grid.set_value("status", "● Loaded / Ready")

        # 4. Update Signal Characteristics (11 DSP metrics)
        self.char_grid_left.set_value("rms", f"{characteristics['rms']:.4f}")
        self.char_grid_left.set_value("std", f"{characteristics['std']:.4f}")
        self.char_grid_left.set_value("peak", f"{characteristics['peak']:.4f}")
        self.char_grid_left.set_value("energy", f"{characteristics['energy']:.2f}")
        self.char_grid_left.set_value("crest_factor", f"{characteristics['crest_factor']:.3f}")
        self.char_grid_left.set_value("zcr", f"{characteristics['zcr']:.4f}")

        dom_f = characteristics["dom_freq"]
        dom_f_str = f"{dom_f / 1000:.2f} kHz" if dom_f >= 1000 else f"{dom_f:.1f}"
        self.char_grid_right.set_value("dom_freq", dom_f_str, "Hz" if dom_f < 1000 else "")

        sc = characteristics["spectral_centroid"]
        sc_str = f"{sc / 1000:.2f} kHz" if sc >= 1000 else f"{sc:.1f}"
        self.char_grid_right.set_value("spectral_centroid", sc_str, "Hz" if sc < 1000 else "")

        sb = characteristics["spectral_bandwidth"]
        sb_str = f"{sb / 1000:.2f} kHz" if sb >= 1000 else f"{sb:.1f}"
        self.char_grid_right.set_value("spectral_bandwidth", sb_str, "Hz" if sb < 1000 else "")

        self.char_grid_right.set_value("spectral_flatness", f"{characteristics['spectral_flatness']:.6f}")

        bw3 = characteristics["bandwidth_3db"]
        bw3_str = f"{bw3 / 1000:.2f} kHz" if bw3 >= 1000 else f"{bw3:.1f}"
        self.char_grid_right.set_value("bandwidth_3db", bw3_str, "Hz" if bw3 < 1000 else "")

        rf = characteristics.get("spectral_rolloff", 0.0)
        rf_str = f"{rf / 1000:.2f} kHz" if rf >= 1000 else f"{rf:.1f}"
        self.char_grid_right.set_value("rolloff", rf_str, "Hz" if rf < 1000 else "")

        # Update Sub-band Profile Energy Bars
        bands = characteristics.get("bands", {})
        for b_key, (val_lbl, bar) in self.band_bars.items():
            pct = bands.get(b_key, 0.0)
            val_lbl.setText(f"{pct * 100:.1f}%")
            val_lbl.setStyleSheet(f"color: {COLORS.TEXT_PRIMARY}; font-size: 9px; font-family: 'Consolas', monospace;")
            bar.setValue(int(np.clip(pct * 1000.0, 0, 1000)))

        self.pf_profile_summary.setText(f"Peak: {dom_f:.1f} Hz  •  Centroid: {sc:.1f} Hz")
        self.pf_profile_summary.setStyleSheet(f"color: {COLORS.PERSIAN_GREEN}; font-size: 9px; font-family: 'Consolas', monospace;")

        # 5. Set Pipeline Indicator to complete
        self.pipeline_indicator.set_stage_progress(5)

        # 6. Final stage status
        self.stage_status_lbl.setText(f"✓ Analysis complete: Class = {pred_class} (Confidence: {conf_pct:.2f}%)")
        self.stage_status_lbl.setStyleSheet(f"color: {COLORS.PERSIAN_GREEN}; font-size: 11px; font-weight: 600;")

        # Re-enable controls
        self.analyze_btn.setEnabled(True)
        self.browse_btn.setEnabled(True)
        self.change_btn.setEnabled(True)
        self.sample_combo.setEnabled(True)

        # Push preprocessed signal into DSP state for the DSP Analysis page
        try:
            dsp_state = get_dsp_state()
            signal_data = results.get("signal")
            sr          = results.get("sample_rate", 16000)
            metadata    = results.get("metadata", {})
            dsp_state.set_pending_audio_signal(
                signal           = signal_data,
                sample_rate      = sr,
                file_path        = self._current_file or "",
                predicted_class  = str(prediction.get("class", "Unknown")),
                confidence       = confidence,
                preprocessing_steps = AUDIO_PREPROCESSING_STEPS,
                metadata         = metadata,
            )
        except Exception:
            pass  # DSP push is non-critical; ML result is already shown


    def _on_analysis_failed(self, error_message: str):
        self.stage_status_lbl.setText("✕ ANALYSIS FAILED: Unable to classify the selected audio signal.")
        self.stage_status_lbl.setStyleSheet(f"color: {COLORS.STATUS_ERROR}; font-size: 11px; font-weight: 600;")

        self.error_details_lbl.setText(f"Technical error during classification:\n{error_message}")
        self.error_details_lbl.setVisible(True)

        self.model_grid.set_value("status", "✕ Error")
        self.pipeline_indicator.reset()

        self.analyze_btn.setEnabled(True)
        self.browse_btn.setEnabled(True)
        self.change_btn.setEnabled(True)
        self.sample_combo.setEnabled(True)
