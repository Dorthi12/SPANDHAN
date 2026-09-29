"""
SPANDHAN — Image ML Analysis Page
==================================
The primary AI + Signal Processing workstation page for 2D Image Signal Classification.
Features:
  - Subtle breadcrumbs: SPANDHAN / MACHINE LEARNING / IMAGE
  - Drag-and-drop input card with file inspection & sample selector
  - Interactive 2D Signal Preview with zoom, pan, hover inspection, colormaps
  - Large gradient "ANALYZE SIGNAL" button with sequential multi-stage progress
  - ML Classification Card with large typography and confidence bar
  - Full Class Probability Distribution (Impulse, Sinusoidal, White Noise, Step, Chirp)
  - Signal Information Card with real image properties
  - Collapsible Signal Characteristics Card (RMS, Entropy, Spatial Freq, Gradient Energy, etc.)
  - Model Information Card with PyTorch SignalCNN architecture specs
  - Signal Class Reference Card
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
from PySide6.QtCore import Qt, QThread, Signal, QRectF
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QColor, QLinearGradient, QPainter, QFont
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
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
from ui.ml.image.image_preview_widget import ImageSignalPreviewWidget
from ui.services.image_analysis_service import (
    ImageAnalysisWorker,
    load_and_inspect_image,
    compute_image_signal_features,
)


class DropUploadArea(QFrame):
    """Interactive drag-and-drop zone with dashed borders and hover states."""

    file_dropped = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setMinimumHeight(130)
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


class ImageMLPage(QWidget):
    """Complete Audio/Image ML Page for 2D Signal Pattern Classification."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self._current_file: Optional[str] = None
        self._current_matrix: Optional[np.ndarray] = None
        self._worker_thread: Optional[QThread] = None
        self._worker: Optional[ImageAnalysisWorker] = None

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
        container.setObjectName("ImageMLContent")
        container.setStyleSheet(f"QWidget#ImageMLContent {{ background-color: {COLORS.BG_PRIMARY}; }}")
        self.content_layout = QVBoxLayout(container)
        self.content_layout.setContentsMargins(28, 20, 28, 28)
        self.content_layout.setSpacing(18)

        # ---------------------------------------------------------------------
        # 1. Subtle Breadcrumbs & Page Header
        # ---------------------------------------------------------------------
        self._build_header()

        # ---------------------------------------------------------------------
        # 2. ROW 1: INPUT SIGNAL & SIGNAL PREVIEW
        # ---------------------------------------------------------------------
        row1_layout = QHBoxLayout()
        row1_layout.setSpacing(18)

        # Left: Input Signal Card
        self.input_card = self._build_input_signal_card()
        self.input_card.setFixedWidth(340)
        row1_layout.addWidget(self.input_card)

        # Right: Signal Preview Card
        self.preview_widget = ImageSignalPreviewWidget(self)
        row1_layout.addWidget(self.preview_widget, stretch=1)

        self.content_layout.addLayout(row1_layout)

        # ---------------------------------------------------------------------
        # 3. ANALYZE BUTTON & STAGE FEEDBACK
        # ---------------------------------------------------------------------
        self._build_analyze_action_bar()

        # ---------------------------------------------------------------------
        # 4. ROW 2: ML CLASSIFICATION & SIGNAL INFORMATION
        # ---------------------------------------------------------------------
        row2_layout = QHBoxLayout()
        row2_layout.setSpacing(18)

        self.classification_card = self._build_classification_card()
        row2_layout.addWidget(self.classification_card, stretch=5)

        self.signal_info_card = self._build_signal_info_card()
        row2_layout.addWidget(self.signal_info_card, stretch=4)

        self.content_layout.addLayout(row2_layout)

        # ---------------------------------------------------------------------
        # 5. ROW 3: CLASS PROBABILITY DISTRIBUTION
        # ---------------------------------------------------------------------
        self.prob_card = self._build_probability_card()
        self.content_layout.addWidget(self.prob_card)

        # ---------------------------------------------------------------------
        # 6. ROW 4: SIGNAL CHARACTERISTICS & MODEL INFORMATION
        # ---------------------------------------------------------------------
        row4_layout = QHBoxLayout()
        row4_layout.setSpacing(18)

        self.characteristics_card = self._build_characteristics_card()
        row4_layout.addWidget(self.characteristics_card, stretch=6)

        self.model_info_card = self._build_model_info_card()
        row4_layout.addWidget(self.model_info_card, stretch=4)

        self.content_layout.addLayout(row4_layout)

        # ---------------------------------------------------------------------
        # 7. BOTTOM: SIGNAL CLASS REFERENCE
        # ---------------------------------------------------------------------
        self.ref_card = self._build_reference_card()
        self.content_layout.addWidget(self.ref_card)

        scroll.setWidget(container)
        master_layout.addWidget(scroll)

    # -------------------------------------------------------------------------
    # UI Builders
    # -------------------------------------------------------------------------
    def _build_header(self):
        header_box = QVBoxLayout()
        header_box.setSpacing(4)

        # Subtle Breadcrumb: SPANDHAN / MACHINE LEARNING / IMAGE
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
        bc_leaf = QLabel("IMAGE")
        bc_leaf.setStyleSheet(f"color: {COLORS.TEXT_HIGHLIGHT}; font-size: 10px; font-weight: 700; letter-spacing: 0.8px;")

        bc_layout.addWidget(bc_root)
        bc_layout.addWidget(bc_sep1)
        bc_layout.addWidget(bc_cat)
        bc_layout.addWidget(bc_sep2)
        bc_layout.addWidget(bc_leaf)
        bc_layout.addStretch()
        header_box.addLayout(bc_layout)

        # Title & Subtitle
        title_lbl = QLabel("Image Signal Classification")
        title_lbl.setStyleSheet(
            f"color: {COLORS.TEXT_PRIMARY}; font-size: 22px; font-weight: 800; "
            f"letter-spacing: -0.3px; margin-top: 2px;"
        )
        sub_lbl = QLabel("AI-assisted identification of fundamental 2D representation signal patterns")
        sub_lbl.setStyleSheet(f"color: {COLORS.TEXT_SECONDARY}; font-size: 12px; margin-top: 1px;")

        header_box.addWidget(title_lbl)
        header_box.addWidget(sub_lbl)
        self.content_layout.addLayout(header_box)

    def _build_input_signal_card(self) -> CardContainer:
        card = CardContainer(title="Input Signal", subtitle="Select or drop 2D signal matrix")

        # Drop Area
        self.drop_area = DropUploadArea(self)
        self.drop_area.file_dropped.connect(self.load_file)

        drop_inner_layout = QVBoxLayout(self.drop_area)
        drop_inner_layout.setContentsMargins(12, 14, 12, 14)
        drop_inner_layout.setSpacing(6)
        drop_inner_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        icon_lbl = QLabel("◫")
        icon_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon_lbl.setStyleSheet(f"color: {COLORS.BLUE_LIGHT}; font-size: 24px;")

        self.drop_text_main = QLabel("Drop audio/image signal here")
        self.drop_text_main.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.drop_text_main.setStyleSheet(f"color: {COLORS.TEXT_PRIMARY}; font-size: 11px; font-weight: 600;")

        self.drop_text_or = QLabel("or browse from workstation")
        self.drop_text_or.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.drop_text_or.setStyleSheet(f"color: {COLORS.TEXT_MUTED}; font-size: 10px;")

        fmt_lbl = QLabel("PNG • JPG • TIFF • BMP • MAT")
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

        self.browse_btn = QPushButton("Browse Files")
        self.browse_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.browse_btn.setStyleSheet(
            f"QPushButton {{ background-color: {COLORS.BG_SURFACE_ALT}; color: {COLORS.TEXT_PRIMARY}; "
            f"border: 1px solid {COLORS.BORDER_SUBTLE}; border-radius: 6px; padding: 7px 12px; font-size: 11px; font-weight: 600; }} "
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

        # File Metadata Capsule (Populated after selection)
        self.file_capsule = QFrame()
        self.file_capsule.setVisible(False)
        self.file_capsule.setStyleSheet(
            f"background-color: #07111F; border: 1px solid {COLORS.BORDER_SUBTLE}; border-radius: 6px; padding: 6px 10px;"
        )
        fc_layout = QVBoxLayout(self.file_capsule)
        fc_layout.setContentsMargins(0, 0, 0, 0)
        fc_layout.setSpacing(3)

        self.capsule_name_lbl = QLabel("✓ filename.png")
        self.capsule_name_lbl.setStyleSheet(
            f"color: {COLORS.PERSIAN_GREEN}; font-size: 11px; font-weight: 700;"
        )
        self.capsule_meta_lbl = QLabel("128 × 128  •  16,384 samples  •  Grayscale")
        self.capsule_meta_lbl.setStyleSheet(f"color: {COLORS.TEXT_MUTED}; font-size: 10px;")

        fc_layout.addWidget(self.capsule_name_lbl)
        fc_layout.addWidget(self.capsule_meta_lbl)
        card.add_widget(self.file_capsule)

        return card

    def _build_analyze_action_bar(self):
        action_layout = QVBoxLayout()
        action_layout.setSpacing(6)

        # Large Primary Action Button
        self.analyze_btn = QPushButton("✦  ANALYZE SIGNAL")
        self.analyze_btn.setProperty("class", "PrimaryButton")
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.setFixedHeight(44)
        self.analyze_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.analyze_btn.clicked.connect(self.start_analysis)
        action_layout.addWidget(self.analyze_btn)

        # Sequential Stage Feedback Status
        self.stage_status_lbl = QLabel("Awaiting input signal selection...")
        self.stage_status_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.stage_status_lbl.setStyleSheet(
            f"color: {COLORS.TEXT_MUTED}; font-size: 11px; font-weight: 500; min-height: 18px;"
        )
        action_layout.addWidget(self.stage_status_lbl)

        self.content_layout.addLayout(action_layout)

    def _build_classification_card(self) -> CardContainer:
        card = CardContainer(title="ML Classification", subtitle="Signal pattern inference outcome")

        body_box = QVBoxLayout()
        body_box.setSpacing(10)

        pred_title = QLabel("PREDICTED SIGNAL")
        pred_title.setStyleSheet(f"color: {COLORS.TEXT_MUTED}; font-size: 10px; font-weight: 700; letter-spacing: 1.2px;")
        body_box.addWidget(pred_title)

        # Big Predicted Class Typography
        self.class_display_lbl = QLabel("AWAITING INFERENCE")
        self.class_display_lbl.setStyleSheet(
            f"color: {COLORS.TEXT_PRIMARY}; font-size: 26px; font-weight: 800; "
            f"letter-spacing: 1.0px; padding: 4px 0px;"
        )
        body_box.addWidget(self.class_display_lbl)

        # Confidence Section
        conf_header_box = QHBoxLayout()
        conf_header_box.setSpacing(6)
        conf_title = QLabel("CONFIDENCE")
        conf_title.setStyleSheet(f"color: {COLORS.TEXT_MUTED}; font-size: 10px; font-weight: 700; letter-spacing: 1.0px;")
        self.conf_num_lbl = QLabel("—")
        self.conf_num_lbl.setStyleSheet(
            f"color: {COLORS.PERSIAN_GREEN}; font-size: 14px; font-weight: 700; font-family: 'Consolas', monospace;"
        )
        conf_header_box.addWidget(conf_title)
        conf_header_box.addStretch()
        conf_header_box.addWidget(self.conf_num_lbl)
        body_box.addLayout(conf_header_box)

        # Horizontal Confidence Progress Bar
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

    def _build_signal_info_card(self) -> CardContainer:
        card = CardContainer(title="Signal Information", subtitle="Measured geometry & tensor properties")

        self.info_grid = MetricGrid(self)
        self.info_grid.add_metric("file", "File", "—")
        self.info_grid.add_metric("format", "Format", "—")
        self.info_grid.add_metric("dimensions", "Dimensions", "—")
        self.info_grid.add_metric("samples", "Samples / Pixels", "—")
        self.info_grid.add_metric("channels", "Channels", "—")
        self.info_grid.add_metric("data_type", "Data Type", "—")
        self.info_grid.add_metric("intensity_range", "Intensity Range", "—")

        card.add_widget(self.info_grid)
        return card

    def _build_probability_card(self) -> CardContainer:
        card = CardContainer(
            title="Class Probability Distribution",
            subtitle="Normalized multiclass posterior probabilities over full hypothesis space",
        )
        self.prob_widget = ProbabilityDistributionWidget(self)
        card.add_widget(self.prob_widget)
        return card

    def _build_characteristics_card(self) -> CardContainer:
        card = CardContainer(
            title="Signal Characteristics",
            subtitle="Calculated 2D spatial features and morphological metrics",
            collapsible=True,
        )

        grid_container = QWidget()
        h_layout = QHBoxLayout(grid_container)
        h_layout.setContentsMargins(0, 0, 0, 0)
        h_layout.setSpacing(24)

        # Left Column Metrics
        self.char_grid_left = MetricGrid(self)
        self.char_grid_left.add_metric("mean", "Mean Intensity", "—")
        self.char_grid_left.add_metric("std", "Standard Deviation", "—")
        self.char_grid_left.add_metric("rms", "RMS Energy", "—")
        self.char_grid_left.add_metric("peak", "Peak Amplitude", "—")
        self.char_grid_left.add_metric("crest_factor", "Peak-to-RMS Ratio", "—")
        self.char_grid_left.add_metric("dyn_range", "Dynamic Range", "—", "dB")
        h_layout.addWidget(self.char_grid_left)

        # Right Column Metrics
        self.char_grid_right = MetricGrid(self)
        self.char_grid_right.add_metric("entropy", "Shannon Entropy", "—", "bits")
        self.char_grid_right.add_metric("contrast", "RMS Contrast", "—")
        self.char_grid_right.add_metric("spatial_freq", "Spatial Frequency", "—")
        self.char_grid_right.add_metric("grad_energy", "Gradient Energy", "—")
        self.char_grid_right.add_metric("sparsity", "Sparsity Ratio", "—", "%")
        self.char_grid_right.add_metric("symmetry", "Symmetry Index", "—")
        h_layout.addWidget(self.char_grid_right)

        card.add_widget(grid_container)
        return card

    def _build_model_info_card(self) -> CardContainer:
        card = CardContainer(title="Model Information", subtitle="Deep learning architecture & runtime specifications")

        self.model_grid = MetricGrid(self)
        self.model_grid.add_metric("model_name", "Model", "Image Signal Classifier")
        self.model_grid.add_metric("arch", "Architecture", "SignalCNN (PyTorch)")
        self.model_grid.add_metric("layers", "Backbone", "3× Conv-BN-ReLU-Pool + GAP")
        self.model_grid.add_metric("head", "Classifier Head", "Dense(128) → Dense(5)")
        self.model_grid.add_metric("contract", "Contract", "128×128×1 Float32")
        self.model_grid.add_metric("classes", "Classes", "5 fundamental classes")
        self.model_grid.add_metric("inference", "Inference", "Pre-trained singleton")
        self.model_grid.add_metric("status", "Status", "● Loaded")

        # Color the status value green
        card.add_widget(self.model_grid)
        return card

    def _build_reference_card(self) -> CardContainer:
        card = CardContainer(title="Signal Class Reference", subtitle="Taxonomy of supported signal types")

        ref_box = QHBoxLayout()
        ref_box.setSpacing(10)

        classes_info = [
            ("IMPULSE", "Localized point / transient Dirac excitation"),
            ("SINUSOIDAL", "Periodic spatial fringe / harmonic wave"),
            ("WHITE NOISE", "Broadband stochastic uncorrelated field"),
            ("STEP", "Abrupt 2D spatial Heaviside boundary"),
            ("CHIRP", "Spatial frequency sweep / dispersion pattern"),
        ]

        for title, desc in classes_info:
            panel = QFrame()
            panel.setStyleSheet(
                f"background-color: #07111F; border: 1px solid {COLORS.BORDER_SUBTLE}; "
                f"border-radius: 6px; padding: 8px 10px;"
            )
            p_layout = QVBoxLayout(panel)
            p_layout.setContentsMargins(0, 0, 0, 0)
            p_layout.setSpacing(3)

            t_lbl = QLabel(title)
            t_lbl.setStyleSheet(f"color: {COLORS.TEXT_HIGHLIGHT}; font-size: 10px; font-weight: 700;")
            d_lbl = QLabel(desc)
            d_lbl.setWordWrap(True)
            d_lbl.setStyleSheet(f"color: {COLORS.TEXT_MUTED}; font-size: 9px;")

            p_layout.addWidget(t_lbl)
            p_layout.addWidget(d_lbl)
            ref_box.addWidget(panel, stretch=1)

        card.add_layout(ref_box)
        return card

    # -------------------------------------------------------------------------
    # Interactions & Logic
    # -------------------------------------------------------------------------
    def _load_available_samples(self):
        """Populate the sample selector with dataset files."""
        root = Path(__file__).resolve().parents[3]
        samples_dir = root / "datasets" / "image"

        self.sample_combo.blockSignals(True)
        self.sample_combo.addItem("Select sample...", "")

        if samples_dir.exists():
            for class_name in ["impulse", "sinusoidal", "white_noise", "step", "chirp"]:
                c_dir = samples_dir / class_name
                if c_dir.exists():
                    pngs = list(c_dir.glob("*.png"))
                    if pngs:
                        first_sample = pngs[0]
                        display_name = f"{class_name.capitalize()} ({first_sample.name})"
                        self.sample_combo.addItem(display_name, str(first_sample))

        self.sample_combo.blockSignals(False)

    def _on_sample_selected(self, index: int):
        path = self.sample_combo.currentData()
        if path and os.path.exists(path):
            self.load_file(path)

    def _open_file_dialog(self):
        root = Path(__file__).resolve().parents[3]
        initial_dir = str(root / "datasets" / "image")
        if not os.path.exists(initial_dir):
            initial_dir = str(root)

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Signal Image",
            initial_dir,
            "Signal Files (*.png *.jpg *.jpeg *.bmp *.tiff *.mat);;All Files (*.*)",
        )
        if file_path:
            self.load_file(file_path)

    def load_file(self, file_path: str):
        """Load image file, update UI metadata and preview, and enable analysis."""
        try:
            arr_2d, info = load_and_inspect_image(file_path)

            self._current_file = file_path
            self._current_matrix = arr_2d

            # Update preview
            self.preview_widget.set_signal_matrix(arr_2d)

            # Update File Capsule
            self.file_capsule.setVisible(True)
            self.capsule_name_lbl.setText(f"✓ {info['filename']}")
            self.capsule_meta_lbl.setText(
                f"{info['raw_format']}  •  {info['contract_dimensions']}  •  {info['total_pixels']:,} samples  •  {info['channels']}"
            )
            self.change_btn.setVisible(True)

            # Update Signal Info Card
            self.info_grid.set_value("file", info["filename"])
            self.info_grid.set_value("format", info["raw_format"])
            self.info_grid.set_value("dimensions", info["contract_dimensions"])
            self.info_grid.set_value("samples", f"{info['total_pixels']:,}")
            self.info_grid.set_value("channels", info["channels"])
            self.info_grid.set_value("data_type", info["data_type"])
            self.info_grid.set_value("intensity_range", f"[{info['min_intensity']:.3f}, {info['max_intensity']:.3f}]")

            # Enable Analyze button
            self.analyze_btn.setEnabled(True)
            self.stage_status_lbl.setText(f"Signal loaded: {info['filename']}. Ready for ML classification.")
            self.stage_status_lbl.setStyleSheet(f"color: {COLORS.PERSIAN_GREEN}; font-size: 11px;")

        except Exception as e:
            self.stage_status_lbl.setText(f"Error loading file: {e}")
            self.stage_status_lbl.setStyleSheet(f"color: {COLORS.STATUS_ERROR}; font-size: 11px;")

    def start_analysis(self):
        """Trigger asynchronous ML inference in background thread."""
        if not self._current_file:
            return

        self.analyze_btn.setEnabled(False)
        self.browse_btn.setEnabled(False)
        self.change_btn.setEnabled(False)
        self.sample_combo.setEnabled(False)

        # Setup worker thread
        self._worker_thread = QThread()
        self._worker = ImageAnalysisWorker(self._current_file)
        self._worker.moveToThread(self._worker_thread)

        self._worker_thread.started.connect(self._worker.run)
        self._worker.stage_changed.connect(self._on_stage_changed)
        self._worker.analysis_finished.connect(self._on_analysis_finished)
        self._worker.analysis_failed.connect(self._on_analysis_failed)

        # Cleanup connections
        self._worker.analysis_finished.connect(self._worker_thread.quit)
        self._worker.analysis_failed.connect(self._worker_thread.quit)
        self._worker_thread.finished.connect(self._worker.deleteLater)
        self._worker_thread.finished.connect(self._worker_thread.deleteLater)

        self._worker_thread.start()

    def _on_stage_changed(self, stage_text: str):
        self.stage_status_lbl.setText(f"✦ {stage_text}")
        self.stage_status_lbl.setStyleSheet(f"color: {COLORS.STATUS_PROCESSING}; font-size: 11px; font-weight: 600;")

    def _on_analysis_finished(self, results: Dict[str, Any]):
        prediction = results["prediction"]
        features = results["features"]

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

        # 3. Update Signal Characteristics
        self.char_grid_left.set_value("mean", f"{features['mean']:.4f}")
        self.char_grid_left.set_value("std", f"{features['std']:.4f}")
        self.char_grid_left.set_value("rms", f"{features['rms']:.4f}")
        self.char_grid_left.set_value("peak", f"{features['peak']:.4f}")
        self.char_grid_left.set_value("crest_factor", f"{features['crest_factor']:.3f}")
        self.char_grid_left.set_value("dyn_range", f"{features['dynamic_range_db']:.1f}", "dB")

        self.char_grid_right.set_value("entropy", f"{features['entropy']:.3f}", "bits")
        self.char_grid_right.set_value("contrast", f"{features['rms_contrast']:.4f}")
        self.char_grid_right.set_value("spatial_freq", f"{features['spatial_frequency']:.4f}")
        self.char_grid_right.set_value("grad_energy", f"{features['gradient_energy']:.5f}")
        self.char_grid_right.set_value("sparsity", f"{features['sparsity_pct']:.1f}", "%")
        self.char_grid_right.set_value("symmetry", f"{features['symmetry_index']:.3f}")

        # Final stage status
        self.stage_status_lbl.setText(f"✓ Analysis complete: Class = {pred_class} (Confidence: {conf_pct:.2f}%)")
        self.stage_status_lbl.setStyleSheet(f"color: {COLORS.PERSIAN_GREEN}; font-size: 11px; font-weight: 600;")

        # Re-enable controls
        self.analyze_btn.setEnabled(True)
        self.browse_btn.setEnabled(True)
        self.change_btn.setEnabled(True)
        self.sample_combo.setEnabled(True)

    def _on_analysis_failed(self, error_message: str):
        self.stage_status_lbl.setText(f"✕ Analysis failed: {error_message}")
        self.stage_status_lbl.setStyleSheet(f"color: {COLORS.STATUS_ERROR}; font-size: 11px;")

        self.analyze_btn.setEnabled(True)
        self.browse_btn.setEnabled(True)
        self.change_btn.setEnabled(True)
        self.sample_combo.setEnabled(True)
