"""
app/pages/ml_page.py
======================
Unified ML Page — Spandhan Milestone F (Inference & Transcript First Edition).

Tab Structure
-------------
1. 🔍 Inference & Diagnostics (Default)
   - Dedicated 🎵 Audio Upload Window (.wav, .mp3, .flac, .csv, .npy)
   - Dedicated 🖼 Image Upload Window (.png, .jpg, .jpeg, .bmp, .tiff, .npy)
   - Live Classification Badge & Confidence
   - Multi-Class Probability Bar Chart
   - Full Accurate Diagnostic Transcript (Copy & Save options)

2. 📊 Model Analytics
   - Confusion Matrix & Top Feature Importances
   - Validation Metrics Summary

3. ⚙ Model Retraining (Advanced)
   - Re-generate 1000-sample dataset
   - Re-train Ensemble Classifier
"""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Optional, Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from PySide6.QtCore import Qt, QRunnable, QThreadPool, QObject, Signal, Slot
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTabWidget, QProgressBar, QTextEdit, QFileDialog,
    QGroupBox, QFormLayout, QSplitter, QFrame,
    QSizePolicy, QScrollArea, QSpinBox, QApplication,
)
from PySide6.QtGui import QFont, QColor, QDragEnterEvent, QDropEvent, QCursor

from intelligence.unified.model_io import model_exists, DEFAULT_MODEL_PATH, load_model
from intelligence.unified.dataset_builder import DEFAULT_DATASET_DIR

# ── Palette & Styling Tokens ──────────────────────────────────────────────────

_ACCENT  = "#7c6af7"
_PANEL   = "#181926"
_CARD    = "#212234"
_CARD_HOVER = "#2a2b42"
_TEXT    = "#f0f0ff"
_MUTED   = "#8e90b0"
_GREEN   = "#4caf50"
_RED     = "#f44336"
_BLUE    = "#29b6f6"
_ORANGE  = "#ff9800"
_PURPLE  = "#ab47bc"


def _lbl(text: str, bold: bool = False, size: int = 11, color: str = _TEXT) -> QLabel:
    w = QLabel(text)
    f = QFont("Segoe UI", size)
    f.setBold(bold)
    w.setFont(f)
    w.setStyleSheet(f"color:{color};")
    return w

def _sep() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    line.setStyleSheet("background: #2e3048; margin: 4px 0;")
    return line

def _grp(title: str) -> QGroupBox:
    g = QGroupBox(title)
    g.setStyleSheet(
        f"QGroupBox {{ background: {_CARD}; border: 1px solid #2e3048; border-radius: 10px; "
        f"color: {_TEXT}; font-size: 11px; font-weight: bold; margin-top: 12px; padding: 12px; }}"
        f"QGroupBox::title {{ subcontrol-origin: margin; left: 12px; padding: 0 6px; color: {_ACCENT}; }}"
    )
    return g

def _btn(text: str, color: str = _ACCENT, hover_color: str = "#9282fa") -> QPushButton:
    b = QPushButton(text)
    b.setFixedHeight(36)
    b.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
    b.setStyleSheet(
        f"QPushButton {{ background: {color}; color: #ffffff; border-radius: 8px; "
        f"font-weight: bold; font-size: 11px; padding: 0 14px; border: none; }}"
        f"QPushButton:hover {{ background: {hover_color}; }}"
        f"QPushButton:disabled {{ background: #35364d; color: {_MUTED}; }}"
    )
    return b

def _spin(lo: int, hi: int, val: int) -> QSpinBox:
    s = QSpinBox()
    s.setRange(lo, hi)
    s.setValue(val)
    s.setFixedHeight(30)
    s.setStyleSheet(
        f"QSpinBox {{ background: #131420; color: {_TEXT}; border: 1px solid #35364d; "
        f"border-radius: 6px; padding: 2px 8px; font-weight: bold; }}"
        f"QSpinBox::up-button, QSpinBox::down-button {{ width: 0px; }}"
    )
    return s


# ── Workers ───────────────────────────────────────────────────────────────────

class _Sig(QObject):
    progress = Signal(int, int, str)
    result   = Signal(object)
    error    = Signal(str)


class _InferWorker(QRunnable):
    def __init__(self, path: str) -> None:
        super().__init__()
        self.path = path
        self.signals = _Sig()

    @Slot()
    def run(self) -> None:
        try:
            from intelligence.unified.predictor import UnifiedPredictor
            predictor = UnifiedPredictor.load()
            result = predictor.predict_file(self.path)
            self.signals.result.emit(result)
        except Exception as exc:
            self.signals.error.emit(f"{exc}\n{traceback.format_exc()}")


class _DatasetWorker(QRunnable):
    def __init__(self, n_audio: int, n_image: int, seed: int) -> None:
        super().__init__()
        self.n_audio = n_audio
        self.n_image = n_image
        self.seed = seed
        self.signals = _Sig()

    @Slot()
    def run(self) -> None:
        try:
            from intelligence.unified.dataset_builder import build_unified_dataset
            self.signals.progress.emit(0, 2, "Generating dataset samples…")
            result = build_unified_dataset(
                n_audio=self.n_audio, n_image=self.n_image, seed=self.seed, save=True, verbose=False
            )
            self.signals.result.emit(result)
        except Exception as exc:
            self.signals.error.emit(f"{exc}\n{traceback.format_exc()}")


class _TrainWorker(QRunnable):
    def __init__(self, n_estimators: int, test_size: float, seed: int) -> None:
        super().__init__()
        self.n_estimators = n_estimators
        self.test_size = test_size
        self.seed = seed
        self.signals = _Sig()

    @Slot()
    def run(self) -> None:
        try:
            from intelligence.unified.trainer import train_unified_model
            def _prog(step, total, msg):
                self.signals.progress.emit(step, total, msg)

            bundle = train_unified_model(
                n_estimators=self.n_estimators, test_size=self.test_size,
                random_state=self.seed, save=True, verbose=False, progress_callback=_prog,
            )
            self.signals.result.emit(bundle)
        except Exception as exc:
            self.signals.error.emit(f"{exc}\n{traceback.format_exc()}")


# ── Dedicated Upload Window Widget ────────────────────────────────────────────

class DedicatedUploadWidget(QFrame):
    file_selected = Signal(str)

    def __init__(self, title: str, icon_str: str, file_filter: str, accent_color: str, parent=None):
        super().__init__(parent)
        self.file_filter = file_filter
        self.setAcceptDrops(True)
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.setStyleSheet(
            f"QFrame {{ background: {_CARD}; border: 2px dashed #3a3b5c; border-radius: 10px; padding: 10px; }}"
            f"QFrame:hover {{ border-color: {accent_color}; background: {_CARD_HOVER}; }}"
        )
        lay = QVBoxLayout(self)
        lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.setSpacing(6)

        ic = _lbl(icon_str, size=22)
        ic.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        t = _lbl(title, bold=True, size=11, color=_TEXT)
        t.setAlignment(Qt.AlignmentFlag.AlignCenter)

        sub = _lbl("Drag & drop file or click to browse", size=9, color=_MUTED)
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.path_lbl = _lbl("No file selected", size=9, color=_MUTED)
        self.path_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.path_lbl.setWordWrap(True)

        lay.addWidget(ic)
        lay.addWidget(t)
        lay.addWidget(sub)
        lay.addWidget(self.path_lbl)

    def mousePressEvent(self, event):
        path, _ = QFileDialog.getOpenFileName(self, "Select File", "", self.file_filter)
        if path:
            self.set_file(path)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            self.set_file(path)

    def set_file(self, path: str):
        self.path_lbl.setText(Path(path).name)
        self.path_lbl.setStyleSheet(f"color:{_GREEN};font-weight:bold;")
        self.file_selected.emit(path)


# ── Matplotlib Canvases ───────────────────────────────────────────────────────

class _ProbCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self._fig, self._ax = plt.subplots(figsize=(5.5, 3), facecolor=_PANEL)
        super().__init__(self._fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._ax.set_facecolor(_CARD)
        self.draw()

    def update_chart(self, probs: dict[str, float], predicted: str) -> None:
        ax = self._ax
        ax.cla()
        ax.set_facecolor(_CARD)
        names = list(probs.keys())
        vals  = [probs[n] for n in names]
        colors = [_ACCENT if n == predicted else "#393a56" for n in names]
        bars = ax.barh(names, vals, color=colors, height=0.55)
        
        for bar, v in zip(bars, vals):
            ax.text(min(v + 0.015, 0.95), bar.get_y() + bar.get_height() / 2,
                    f"{v:.1%}", va="center", color=_TEXT, fontsize=8, fontweight="bold")
        
        ax.set_xlim(0, 1.08)
        ax.set_xlabel("Probability Confidence", color=_MUTED, fontsize=8, labelpad=4)
        ax.tick_params(colors=_TEXT, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333452")
        self._fig.tight_layout(pad=0.6)
        self.draw()


class _CmCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self._fig, self._ax = plt.subplots(figsize=(5, 4), facecolor=_PANEL)
        super().__init__(self._fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.draw()

    def update_chart(self, cm: np.ndarray, class_names: list[str]) -> None:
        ax = self._ax
        ax.cla()
        ax.set_facecolor(_CARD)
        im = ax.imshow(cm, interpolation="nearest", cmap="Purples")
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=40, ha="right", color=_TEXT, fontsize=8)
        ax.set_yticklabels(class_names, color=_TEXT, fontsize=8)
        ax.set_xlabel("Predicted Label", color=_MUTED, fontsize=9)
        ax.set_ylabel("True Label", color=_MUTED, fontsize=9)
        ax.set_title("Confusion Matrix", color=_TEXT, fontsize=10, pad=6, fontweight="bold")
        thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "#d0d0e8", fontsize=8, fontweight="bold")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333452")
        self._fig.tight_layout()
        self.draw()


class _FiCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self._fig, self._ax = plt.subplots(figsize=(5, 4), facecolor=_PANEL)
        super().__init__(self._fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.draw()

    def update_chart(self, importances: np.ndarray, names: list[str], top: int = 12) -> None:
        ax = self._ax
        ax.cla()
        ax.set_facecolor(_CARD)
        idxs = importances.argsort()[-top:]
        vals = importances[idxs]
        lbls = [names[i] if i < len(names) else f"f{i}" for i in idxs]
        ax.barh(lbls, vals, color=_BLUE, height=0.55)
        ax.set_xlabel("Importance Score", color=_MUTED, fontsize=9)
        ax.set_title(f"Top {top} Predictive Features", color=_TEXT, fontsize=10, pad=6, fontweight="bold")
        ax.tick_params(colors=_TEXT, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333452")
        self._fig.tight_layout()
        self.draw()


# ── Tab 1: Primary Inference & Diagnostics ────────────────────────────────────

class _InferenceTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._pool = QThreadPool.globalInstance()
        self._active_path = ""
        self._last_result = None
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(12)
        root.setContentsMargins(16, 16, 16, 16)
        self.setStyleSheet(f"background:{_PANEL};")

        # ── Dedicated Upload Windows Row ──────────────────────────
        upload_row = QHBoxLayout()
        upload_row.setSpacing(12)

        self._audio_drop = DedicatedUploadWidget(
            "Audio Signal Window", "🎵",
            "Audio Files (*.wav *.csv *.npy *.mp3 *.flac)", _BLUE
        )
        self._image_drop = DedicatedUploadWidget(
            "Image Media Window", "🖼",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff *.npy)", _PURPLE
        )

        self._audio_drop.file_selected.connect(self._on_file_selected)
        self._image_drop.file_selected.connect(self._on_file_selected)

        upload_row.addWidget(self._audio_drop)
        upload_row.addWidget(self._image_drop)
        root.addLayout(upload_row)

        # ── Action Button & Status Bar ───────────────────────────
        action_row = QHBoxLayout()
        self._predict_btn = _btn("🔍 Run Intelligence Analysis & Generate Transcript", _ACCENT, "#8c7dfc")
        self._predict_btn.setFixedHeight(40)
        self._predict_btn.setEnabled(False)
        self._predict_btn.clicked.connect(self._on_predict)
        
        self._badge = QLabel("Loaded Pre-Trained Model")
        self._badge.setStyleSheet(
            f"background: {_CARD}; color: {_GREEN}; border-radius: 8px; "
            f"padding: 6px 14px; font-weight: bold; font-size: 11px;"
        )
        
        action_row.addWidget(self._predict_btn, stretch=3)
        action_row.addWidget(self._badge, stretch=1)
        root.addLayout(action_row)

        # ── Results & Transcript Area (Splitter) ─────────────────
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(4)
        splitter.setStyleSheet("QSplitter::handle { background: #2a2b42; }")

        # Left: Probability Distribution Chart
        left = QWidget()
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(0, 0, 0, 0)
        left_lay.addWidget(_lbl("Multi-Class Confidence Distribution", bold=True, size=11))
        self._prob_canvas = _ProbCanvas()
        left_lay.addWidget(self._prob_canvas)
        splitter.addWidget(left)

        # Right: Full Accurate Diagnostic Transcript
        right = QWidget()
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(0, 0, 0, 0)
        right_lay.setSpacing(6)

        t_header = QHBoxLayout()
        t_header.addWidget(_lbl("Accurate Diagnostic Transcript", bold=True, size=11))
        t_header.addStretch(1)

        self._copy_btn = _btn("📋 Copy", "#333452", "#444568")
        self._copy_btn.setFixedHeight(28)
        self._copy_btn.clicked.connect(self._on_copy_transcript)
        
        self._save_btn = _btn("💾 Save (.txt)", "#333452", "#444568")
        self._save_btn.setFixedHeight(28)
        self._save_btn.clicked.connect(self._on_save_transcript)

        t_header.addWidget(self._copy_btn)
        t_header.addWidget(self._save_btn)
        right_lay.addLayout(t_header)

        self._transcript_box = QTextEdit()
        self._transcript_box.setReadOnly(True)
        self._transcript_box.setStyleSheet(
            f"QTextEdit {{ background: #11121c; color: {_TEXT}; border: 1px solid #292a40; "
            f"border-radius: 8px; font-family: Consolas, monospace; font-size: 10px; padding: 10px; }}"
        )
        self._transcript_box.setPlaceholderText("Upload an audio or image file above and click 'Run Intelligence Analysis' to generate transcript…")
        right_lay.addWidget(self._transcript_box)
        splitter.addWidget(right)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        root.addWidget(splitter)

    @Slot(str)
    def _on_file_selected(self, path: str):
        self._active_path = path
        self._predict_btn.setEnabled(True)
        self._badge.setText(f"File Ready: {Path(path).name}")
        self._badge.setStyleSheet(f"background:{_CARD};color:{_BLUE};border-radius:8px;font-weight:bold;")

    def _on_predict(self):
        if not model_exists():
            self._badge.setText("Model File Missing!")
            self._badge.setStyleSheet(f"background:{_RED};color:#fff;border-radius:8px;")
            return

        self._predict_btn.setEnabled(False)
        self._badge.setText("Analyzing...")
        self._badge.setStyleSheet(f"background:{_ACCENT};color:#fff;border-radius:8px;")

        w = _InferWorker(self._active_path)
        w.signals.result.connect(self._on_infer_done)
        w.signals.error.connect(self._on_infer_error)
        self._pool.start(w)

    @Slot(object)
    def _on_infer_done(self, result):
        self._predict_btn.setEnabled(True)
        self._last_result = result
        
        if not result.success:
            self._badge.setText("Analysis Error")
            self._badge.setStyleSheet(f"background:{_RED};color:#fff;border-radius:8px;")
            self._transcript_box.setText(f"Error analyzing file: {result.error}")
            return

        clean_type = result.noise_type.replace("_", " ").title()
        self._badge.setText(f"Result: {clean_type} ({result.confidence:.1%})")
        self._badge.setStyleSheet(f"background:{_GREEN};color:#fff;border-radius:8px;font-weight:bold;")

        self._prob_canvas.update_chart(result.all_probabilities, result.noise_type)

        # Generate accurate transcript
        transcript_text = result.generate_transcript(self._active_path)
        self._transcript_box.setText(transcript_text)

    @Slot(str)
    def _on_infer_error(self, msg):
        self._predict_btn.setEnabled(True)
        self._badge.setText("Error")
        self._transcript_box.setText(f"Error: {msg}")

    def _on_copy_transcript(self):
        txt = self._transcript_box.toPlainText()
        if txt:
            QApplication.clipboard().setText(txt)
            self._copy_btn.setText("✔ Copied!")
            QApplication.processEvents()

    def _on_save_transcript(self):
        txt = self._transcript_box.toPlainText()
        if not txt:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Transcript", "spandhan_diagnostic_transcript.txt", "Text Files (*.txt)")
        if path:
            with open(path, "w") as f:
                f.write(txt)


# ── Tab 2: Model Info & Analytics ─────────────────────────────────────────────

class _AnalyticsTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(12)
        root.setContentsMargins(16, 16, 16, 16)
        self.setStyleSheet(f"background:{_PANEL};")

        top_row = QHBoxLayout()
        top_row.addWidget(_lbl("Model Analytics & Performance Metrics", bold=True, size=12))
        top_row.addStretch(1)
        
        ref_btn = _btn("🔄 Refresh Stats", _ACCENT)
        ref_btn.setFixedWidth(130)
        ref_btn.clicked.connect(self.refresh)
        top_row.addWidget(ref_btn)
        root.addLayout(top_row)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(4)
        splitter.setStyleSheet("QSplitter::handle { background: #2a2b42; }")

        cm_grp = _grp("Confusion Matrix")
        cm_lay = QVBoxLayout(cm_grp)
        self._cm_canvas = _CmCanvas()
        cm_lay.addWidget(self._cm_canvas)
        splitter.addWidget(cm_grp)

        fi_grp = _grp("Feature Importances")
        fi_lay = QVBoxLayout(fi_grp)
        self._fi_canvas = _FiCanvas()
        fi_lay.addWidget(self._fi_canvas)
        splitter.addWidget(fi_grp)

        root.addWidget(splitter)

        cr_grp = _grp("Classification Metrics Summary")
        cr_lay = QVBoxLayout(cr_grp)
        self._cr_text = QTextEdit()
        self._cr_text.setReadOnly(True)
        self._cr_text.setStyleSheet(
            f"QTextEdit {{ background: #11121c; color: {_TEXT}; border: 1px solid #292a40; "
            f"border-radius: 6px; font-family: Consolas, monospace; font-size: 10px; padding: 8px; }}"
        )
        self._cr_text.setFixedHeight(110)
        cr_lay.addWidget(self._cr_text)
        root.addWidget(cr_grp)

        self.refresh()

    def refresh(self):
        if not model_exists():
            self._cr_text.setText("No trained model bundle found.")
            return
        try:
            from intelligence.unified.feature_extractor import UNIFIED_FEATURE_NAMES
            bundle = load_model()
            if bundle.confusion_matrix is not None:
                self._cm_canvas.update_chart(bundle.confusion_matrix, bundle.class_names)
            if bundle.feature_importances is not None:
                self._fi_canvas.update_chart(bundle.feature_importances, UNIFIED_FEATURE_NAMES)

            report = (
                f"Validation Accuracy : {bundle.val_accuracy:.4f}\n"
                f"Train Samples       : {bundle.n_train}\n"
                f"Validation Samples  : {bundle.n_val}\n"
                f"Classes Trained     : {', '.join(bundle.class_names)}\n\n"
                f"{bundle.classification_report}"
            )
            self._cr_text.setText(report)
        except Exception as exc:
            self._cr_text.setText(f"Error loading analytics: {exc}")


# ── Tab 3: Model Retraining (Advanced) ────────────────────────────────────────

class _RetrainTab(QWidget):
    model_trained = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pool = QThreadPool.globalInstance()
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(12)
        root.setContentsMargins(16, 16, 16, 16)
        self.setStyleSheet(f"background:{_PANEL};")

        root.addWidget(_lbl("Advanced Model Re-Training & Dataset Generation", bold=True, size=12))

        settings_row = QHBoxLayout()
        ds_grp = _grp("Dataset Generator Options")
        ds_form = QFormLayout(ds_grp)
        self._n_audio = _spin(50, 2000, 500)
        self._n_image = _spin(50, 2000, 500)
        self._ds_seed = _spin(0, 9999, 42)
        ds_form.addRow(_lbl("Audio Samples:", size=9), self._n_audio)
        ds_form.addRow(_lbl("Image Samples:", size=9), self._n_image)
        ds_form.addRow(_lbl("Seed:", size=9), self._ds_seed)
        settings_row.addWidget(ds_grp)

        tr_grp = _grp("Ensemble Classifier Options")
        tr_form = QFormLayout(tr_grp)
        self._n_est = _spin(50, 500, 150)
        self._tr_seed = _spin(0, 9999, 42)
        tr_form.addRow(_lbl("Estimators:", size=9), self._n_est)
        tr_form.addRow(_lbl("Seed:", size=9), self._tr_seed)
        settings_row.addWidget(tr_grp)
        root.addLayout(settings_row)

        btn_row = QHBoxLayout()
        self._ds_btn  = _btn("⚙ Re-Generate Dataset", _ACCENT)
        self._tr_btn  = _btn("🚂 Re-Train Ensemble Classifier", "#2e7d32")
        self._all_btn = _btn("⚡ Re-Build + Train All", "#7b1fa2")
        
        self._ds_btn.clicked.connect(self._on_create_dataset)
        self._tr_btn.clicked.connect(self._on_train)
        self._all_btn.clicked.connect(self._on_create_and_train)
        
        for b in (self._ds_btn, self._tr_btn, self._all_btn):
            btn_row.addWidget(b)
        root.addLayout(btn_row)

        self._progress = QProgressBar()
        self._progress.setRange(0, 6)
        self._progress.setValue(0)
        self._progress.setFixedHeight(8)
        self._progress.setStyleSheet(
            f"QProgressBar {{ background: #131420; border-radius: 4px; border: none; }}"
            f"QProgressBar::chunk {{ background: {_ACCENT}; border-radius: 4px; }}"
        )
        root.addWidget(self._progress)

        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setStyleSheet(
            f"QTextEdit {{ background: #11121c; color: {_TEXT}; border: 1px solid #292a40; "
            f"border-radius: 8px; font-family: Consolas, monospace; font-size: 10px; padding: 8px; }}"
        )
        self._log.setFixedHeight(120)
        root.addWidget(self._log)

    def _log_line(self, msg: str):
        self._log.append(msg)
        self._log.verticalScrollBar().setValue(self._log.verticalScrollBar().maximum())

    def _on_create_dataset(self):
        self._log_line("▶ Generating dataset…")
        w = _DatasetWorker(self._n_audio.value(), self._n_image.value(), self._ds_seed.value())
        w.signals.progress.connect(self._on_progress)
        w.signals.result.connect(lambda res: self._log_line(f"✔ Dataset ready: {res.n_total} samples"))
        self._pool.start(w)

    def _on_train(self):
        self._log_line("▶ Training ensemble model…")
        w = _TrainWorker(self._n_est.value(), 0.2, self._tr_seed.value())
        w.signals.progress.connect(self._on_progress)
        w.signals.result.connect(self._on_train_done)
        self._pool.start(w)

    def _on_create_and_train(self):
        self._log_line("▶ Generating dataset + training model…")
        w = _DatasetWorker(self._n_audio.value(), self._n_image.value(), self._ds_seed.value())
        w.signals.progress.connect(self._on_progress)
        w.signals.result.connect(self._on_dataset_then_train)
        self._pool.start(w)

    @Slot(object)
    def _on_dataset_then_train(self, res):
        w = _TrainWorker(self._n_est.value(), 0.2, self._tr_seed.value())
        w.signals.progress.connect(self._on_progress)
        w.signals.result.connect(self._on_train_done)
        self._pool.start(w)

    @Slot(object)
    def _on_train_done(self, bundle):
        self._log_line(f"✔ Training complete! Val accuracy: {bundle.val_accuracy:.1%}")
        self.model_trained.emit(bundle)

    @Slot(int, int, str)
    def _on_progress(self, step, total, msg):
        self._progress.setMaximum(max(total, 1))
        self._progress.setValue(step)
        self._log_line(f"  [{step}/{total}] {msg}")


# ── Main MLPage ───────────────────────────────────────────────────────────────

class MLPage(QWidget):
    """
    Unified ML page: Inference & Diagnostics (Default) | Analytics | Retraining.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        self.setStyleSheet(f"background:{_PANEL};")
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        header = QFrame()
        header.setFixedHeight(54)
        header.setStyleSheet(f"background: #12131f; border-bottom: 1px solid #2a2b42;")
        h_lay = QHBoxLayout(header)
        h_lay.setContentsMargins(16, 0, 16, 0)

        title = _lbl("⚡ Spandhan Signal & Image Noise Classifier", bold=True, size=13, color=_TEXT)
        h_lay.addWidget(title)
        h_lay.addStretch(1)

        pill = QLabel(" Pre-Trained Model Ready ")
        pill.setStyleSheet(
            f"background: #2a2b42; color: {_GREEN}; border-radius: 12px; "
            f"padding: 4px 12px; font-weight: bold; font-size: 10px;"
        )
        h_lay.addWidget(pill)
        root.addWidget(header)

        self._tabs = QTabWidget()
        self._tabs.setStyleSheet(
            f"QTabWidget::pane {{ border: none; background: {_PANEL}; }}"
            f"QTabBar::tab {{ background: #12131f; color: {_MUTED}; padding: 10px 22px; "
            f"font-size: 11px; font-weight: bold; border-bottom: 2px solid transparent; margin-right: 2px; }}"
            f"QTabBar::tab:selected {{ background: {_PANEL}; color: {_ACCENT}; border-bottom: 2px solid {_ACCENT}; }}"
            f"QTabBar::tab:hover {{ color: {_TEXT}; }}"
        )

        self._infer_tab   = _InferenceTab()
        self._analytics_tab = _AnalyticsTab()
        self._retrain_tab   = _RetrainTab()

        self._retrain_tab.model_trained.connect(self._analytics_tab.refresh)

        # Tab Order: Inference is #1 default tab!
        self._tabs.addTab(self._infer_tab,     "🔍  Inference & Diagnostics")
        self._tabs.addTab(self._analytics_tab, "📊  Model Analytics")
        self._tabs.addTab(self._retrain_tab,   "⚙  Re-Train Model (Advanced)")

        root.addWidget(self._tabs)
