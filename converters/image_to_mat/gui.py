"""
converters/image_to_mat/gui.py
==============================
Standalone Drag-and-Drop Desktop GUI for converting images into MATLAB .mat files.
Provides real-time image preview, signal waveform visualization, and one-click
export to Spandhan's input workflow.
"""

from __future__ import annotations

import os
import sys
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QDoubleSpinBox, QFileDialog,
    QFrame, QGroupBox, QSplitter, QMessageBox, QScrollArea,
)
from PySide6.QtGui import QFont, QColor, QDragEnterEvent, QDropEvent, QCursor, QIcon

from converters.image_to_mat.converter import (
    load_image_as_array,
    convert_image_to_signal,
    save_signal_to_mat,
    CONVERSION_MODES,
    SUPPORTED_IMAGE_EXTENSIONS,
)

# Visual styling tokens
_BG      = "#0F172A"
_PANEL   = "#1E293B"
_CARD    = "#253349"
_ACCENT  = "#38BDF8"
_ACCENT_HOVER = "#60A5FA"
_PURPLE  = "#818CF8"
_GREEN   = "#4ADE80"
_TEXT    = "#F8FAFC"
_MUTED   = "#94A3B8"
_BORDER  = "#334155"


def _lbl(text: str, bold: bool = False, size: int = 11, color: str = _TEXT) -> QLabel:
    w = QLabel(text)
    f = QFont("Segoe UI", size)
    f.setBold(bold)
    w.setFont(f)
    w.setStyleSheet(f"color: {color};")
    return w


class DropZoneWidget(QFrame):
    """Interactive drag and drop zone for image files."""
    image_selected = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {_CARD};
                border: 2px dashed {_BORDER};
                border-radius: 12px;
                padding: 16px;
            }}
            QFrame:hover {{
                border-color: {_ACCENT};
                background-color: #1E3A8A30;
            }}
        """)
        lay = QVBoxLayout(self)
        lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.setSpacing(6)

        icon = _lbl("🖼️", size=26)
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)

        title = _lbl("Drag & Drop Image Here", bold=True, size=12, color=_TEXT)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        sub = _lbl("or click to browse (.png, .jpg, .bmp, .tiff, .npy)", size=9, color=_MUTED)
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.file_lbl = _lbl("No file selected", size=9, color=_MUTED)
        self.file_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

        lay.addWidget(icon)
        lay.addWidget(title)
        lay.addWidget(sub)
        lay.addWidget(self.file_lbl)

    def mousePressEvent(self, event):
        filter_str = "Image Files (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp *.npy);;All Files (*.*)"
        path, _ = QFileDialog.getOpenFileName(self, "Select Image File", "", filter_str)
        if path:
            self.set_file(path)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            if os.path.isfile(path):
                self.set_file(path)

    def set_file(self, path: str):
        self.file_lbl.setText(Path(path).name)
        self.file_lbl.setStyleSheet(f"color: {_GREEN}; font-weight: bold;")
        self.image_selected.emit(path)


class ImageCanvas(FigureCanvas):
    """Canvas to preview the 2D image."""
    def __init__(self, parent=None):
        self._fig, self._ax = plt.subplots(figsize=(4, 3.2), facecolor=_PANEL)
        self._fig.subplots_adjust(left=0.04, right=0.96, top=0.92, bottom=0.04)
        super().__init__(self._fig)
        self.setParent(parent)
        self._ax.set_facecolor(_CARD)
        self._ax.axis("off")
        self._ax.set_title("Image Preview", color=_MUTED, fontsize=10, pad=6)
        self.draw()

    def display_image(self, img_array: np.ndarray, title: str = ""):
        self._ax.clear()
        self._ax.imshow(img_array, cmap="gray", aspect="auto", vmin=0.0, vmax=1.0)
        self._ax.axis("off")
        self._ax.set_title(title or f"Image ({img_array.shape[0]} × {img_array.shape[1]})", color=_TEXT, fontsize=10, pad=6)
        self.draw()


class SignalCanvas(FigureCanvas):
    """Canvas to preview the converted 1D signal waveform."""
    def __init__(self, parent=None):
        self._fig, self._ax = plt.subplots(figsize=(4, 3.2), facecolor=_PANEL)
        self._fig.subplots_adjust(left=0.12, right=0.96, top=0.90, bottom=0.18)
        super().__init__(self._fig)
        self.setParent(parent)
        self._ax.set_facecolor(_CARD)
        self._ax.tick_params(colors=_MUTED, labelsize=8)
        for spine in self._ax.spines.values():
            spine.set_color(_BORDER)
        self._ax.set_title("1D Converted Signal Waveform", color=_MUTED, fontsize=10, pad=6)
        self._ax.set_xlabel("Sample Index", color=_MUTED, fontsize=8)
        self._ax.set_ylabel("Amplitude [0, 1]", color=_MUTED, fontsize=8)
        self.draw()

    def display_signal(self, signal: np.ndarray, fs: float):
        self._ax.clear()
        self._ax.set_facecolor(_CARD)
        self._ax.tick_params(colors=_MUTED, labelsize=8)
        for spine in self._ax.spines.values():
            spine.set_color(_BORDER)

        n = len(signal)
        # Downsample for snappy UI display if array is huge
        if n > 20000:
            step = n // 10000
            x = np.arange(0, n, step)
            y = signal[::step]
        else:
            x = np.arange(n)
            y = signal

        self._ax.plot(x, y, color=_ACCENT, linewidth=1.0, alpha=0.9)
        self._ax.set_title(f"1D Signal ({n:,} samples, {n/fs:.2f}s @ {fs:.0f}Hz)", color=_TEXT, fontsize=10, pad=6)
        self._ax.set_xlabel("Sample Index", color=_MUTED, fontsize=8)
        self._ax.set_ylabel("Normalized Value", color=_MUTED, fontsize=8)
        self._ax.grid(True, color=_BORDER, alpha=0.4, linestyle="--")
        self._fig.tight_layout()
        self.draw()


class ImageToMatWindow(QMainWindow):
    """Main application window for Image-to-.MAT conversion tool."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Spandhan — Image to .MAT Signal Converter")
        self.resize(1080, 680)
        self.setMinimumSize(850, 520)
        self.setStyleSheet(f"""
            QMainWindow {{ background-color: {_BG}; }}
            QWidget {{ color: {_TEXT}; font-family: 'Segoe UI', Arial, sans-serif; }}
            QComboBox, QDoubleSpinBox {{
                background-color: {_CARD};
                border: 1px solid {_BORDER};
                border-radius: 6px;
                padding: 4px 8px;
                color: {_TEXT};
                font-weight: bold;
            }}
            QComboBox::drop-down {{ border: none; }}
            QPushButton#Primary {{
                background-color: {_ACCENT};
                color: #0F172A;
                font-weight: bold;
                border-radius: 8px;
                padding: 8px 16px;
                font-size: 12px;
            }}
            QPushButton#Primary:hover {{ background-color: {_ACCENT_HOVER}; }}
            QPushButton#Secondary {{
                background-color: {_CARD};
                color: {_TEXT};
                border: 1px solid {_BORDER};
                border-radius: 8px;
                padding: 6px 12px;
                font-size: 11px;
            }}
            QPushButton#Secondary:hover {{ background-color: {_BORDER}; }}
        """)

        self.current_image_path: Optional[str] = None
        self.current_image_array: Optional[np.ndarray] = None
        self.current_signal_1d: Optional[np.ndarray] = None
        self.last_saved_mat_path: Optional[Path] = None

        self._init_ui()

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_lay = QVBoxLayout(central)
        root_lay.setContentsMargins(16, 16, 16, 16)
        root_lay.setSpacing(12)

        # Header Bar
        header = QHBoxLayout()
        title_box = QVBoxLayout()
        title_box.setSpacing(2)
        h1 = _lbl("Image to .MAT Signal Converter", bold=True, size=15, color=_ACCENT)
        h2 = _lbl("Transform 2D image scans into 1D MATLAB .mat signals ready for Spandhan DSP workflows.", size=10, color=_MUTED)
        title_box.addWidget(h1)
        title_box.addWidget(h2)
        header.addLayout(title_box)
        header.addStretch()
        root_lay.addLayout(header)

        # Main Split Content
        split_layout = QHBoxLayout()
        split_layout.setSpacing(14)

        # Left Column: Configuration Controls
        left_panel = QFrame()
        left_panel.setStyleSheet(f"background-color: {_PANEL}; border-radius: 12px; padding: 6px;")
        left_lay = QVBoxLayout(left_panel)
        left_lay.setSpacing(12)
        left_panel.setFixedWidth(360)

        # 1. Drop zone
        self.drop_zone = DropZoneWidget()
        self.drop_zone.image_selected.connect(self._on_image_selected)
        left_lay.addWidget(self.drop_zone)

        # 2. Options Form
        opt_group = QGroupBox("Conversion Settings")
        opt_group.setStyleSheet(f"QGroupBox {{ color: {_ACCENT}; font-weight: bold; border: 1px solid {_BORDER}; border-radius: 8px; margin-top: 10px; padding: 10px; }}")
        form_lay = QVBoxLayout(opt_group)
        form_lay.setSpacing(8)

        # Mode
        form_lay.addWidget(_lbl("Signal Extraction Mode:", bold=True, size=10, color=_MUTED))
        self.mode_combo = QComboBox()
        for key, desc in CONVERSION_MODES.items():
            self.mode_combo.addItem(desc, key)
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        form_lay.addWidget(self.mode_combo)

        # Sampling Rate
        form_lay.addWidget(_lbl("Sampling Rate (Hz):", bold=True, size=10, color=_MUTED))
        self.fs_spin = QDoubleSpinBox()
        self.fs_spin.setRange(1.0, 1000000.0)
        self.fs_spin.setValue(1000.0)
        self.fs_spin.setSuffix(" Hz")
        self.fs_spin.valueChanged.connect(self._on_fs_changed)
        form_lay.addWidget(self.fs_spin)

        left_lay.addWidget(opt_group)

        # 3. Action Buttons
        self.convert_btn = QPushButton("💾 Convert & Save .MAT File")
        self.convert_btn.setObjectName("Primary")
        self.convert_btn.setFixedHeight(40)
        self.convert_btn.setEnabled(False)
        self.convert_btn.clicked.connect(self._on_convert_clicked)
        left_lay.addWidget(self.convert_btn)

        self.open_folder_btn = QPushButton("📁 Open Output Folder")
        self.open_folder_btn.setObjectName("Secondary")
        self.open_folder_btn.setFixedHeight(32)
        self.open_folder_btn.setEnabled(False)
        self.open_folder_btn.clicked.connect(self._on_open_folder_clicked)
        left_lay.addWidget(self.open_folder_btn)

        # Status Label
        self.status_lbl = _lbl("Ready. Select or drop an image to begin.", size=9, color=_MUTED)
        self.status_lbl.setWordWrap(True)
        left_lay.addWidget(self.status_lbl)
        left_lay.addStretch()

        split_layout.addWidget(left_panel)

        # Right Column: Visualizer Canvases (Image + Signal)
        right_panel = QFrame()
        right_panel.setStyleSheet(f"background-color: {_PANEL}; border-radius: 12px; padding: 6px;")
        right_lay = QVBoxLayout(right_panel)
        right_lay.setSpacing(10)

        # Canvas split
        self.image_canvas = ImageCanvas()
        self.signal_canvas = SignalCanvas()
        right_lay.addWidget(self.image_canvas, stretch=1)
        right_lay.addWidget(self.signal_canvas, stretch=1)

        split_layout.addWidget(right_panel, stretch=1)
        root_lay.addLayout(split_layout)

    def _on_image_selected(self, file_path: str):
        try:
            self.current_image_path = file_path
            self.current_image_array = load_image_as_array(file_path, grayscale=True, normalize=True)
            self.image_canvas.display_image(self.current_image_array, title=f"Image: {Path(file_path).name}")
            self._update_signal()
            self.convert_btn.setEnabled(True)
            self.status_lbl.setText(f"Loaded: {Path(file_path).name} ({self.current_image_array.shape[0]}x{self.current_image_array.shape[1]})")
            self.status_lbl.setStyleSheet(f"color: {_GREEN}; font-size: 9px;")
        except Exception as exc:
            QMessageBox.critical(self, "Load Error", f"Failed to load image:\n{exc}")
            self.status_lbl.setText(f"Error: {exc}")
            self.status_lbl.setStyleSheet(f"color: #F87171; font-size: 9px;")

    def _on_mode_changed(self):
        if self.current_image_array is not None:
            self._update_signal()

    def _on_fs_changed(self):
        if self.current_signal_1d is not None:
            self.signal_canvas.display_signal(self.current_signal_1d, self.fs_spin.value())

    def _update_signal(self):
        if self.current_image_array is None:
            return
        mode = self.mode_combo.currentData()
        self.current_signal_1d = convert_image_to_signal(self.current_image_array, mode=mode)
        self.signal_canvas.display_signal(self.current_signal_1d, self.fs_spin.value())

    def _on_convert_clicked(self):
        if self.current_image_path is None or self.current_signal_1d is None:
            return

        mode = self.mode_combo.currentData()
        src_path = Path(self.current_image_path)
        default_out_dir = Path("converters/image_to_mat/output_mat").resolve()
        default_out_dir.mkdir(parents=True, exist_ok=True)
        default_out_name = f"{src_path.stem}_{mode}.mat"
        default_full_path = default_out_dir / default_out_name

        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Converted .MAT File",
            str(default_full_path),
            "MAT Files (*.mat)"
        )

        if not save_path:
            return

        try:
            meta = {
                "source_image": src_path.name,
                "conversion_mode": mode,
                "original_shape": np.array(self.current_image_array.shape, dtype=np.int32),
                "num_samples": int(len(self.current_signal_1d)),
            }
            saved_path = save_signal_to_mat(
                signal=self.current_signal_1d,
                output_mat_path=save_path,
                sampling_rate=self.fs_spin.value(),
                metadata=meta,
            )
            self.last_saved_mat_path = saved_path
            self.open_folder_btn.setEnabled(True)
            self.status_lbl.setText(f"✓ Saved successfully:\n{saved_path.name}")
            self.status_lbl.setStyleSheet(f"color: {_GREEN}; font-weight: bold; font-size: 9px;")
            QMessageBox.information(
                self,
                "Conversion Complete",
                f"Successfully exported .mat file!\n\nFile: {saved_path.name}\nSamples: {len(self.current_signal_1d):,}\nPath: {saved_path}\n\nYou can now drag & drop this .mat file into Spandhan's Input Workspace."
            )
        except Exception as exc:
            QMessageBox.critical(self, "Save Error", f"Failed to save .mat file:\n{exc}")

    def _on_open_folder_clicked(self):
        target_dir = self.last_saved_mat_path.parent if self.last_saved_mat_path else Path("converters/image_to_mat/output_mat").resolve()
        if sys.platform == "win32":
            os.startfile(str(target_dir))
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(target_dir)])
        else:
            subprocess.Popen(["xdg-open", str(target_dir)])


def main():
    app = QApplication(sys.argv)
    window = ImageToMatWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
