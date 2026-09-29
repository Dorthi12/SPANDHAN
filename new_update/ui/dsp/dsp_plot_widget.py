"""
SPANDHAN — Interactive Scientific Plot Widget
==============================================
Embeds high-performance Matplotlib FigureCanvasQTAgg with:
  - Custom dark scientific toolbar (Zoom, Pan, Reset, Grid, Fullscreen, Export)
  - Real-time coordinate hover readout with physical units
  - Scientific dark theme matching SPANDHAN (#081426, #1E293B, #94A3B8)
  - Anti-aliasing, responsive sizing, and high-DPI scaling
"""

from __future__ import annotations

import os
from typing import Callable, Optional, Tuple, Any
from pathlib import Path

from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QColor, QFont, QIcon, QPainter, QCursor
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFrame,
    QSizePolicy,
    QFileDialog,
)

import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT

from ui.dsp.dsp_theme import DSP_THEME, apply_scientific_plot_style


class MicroPlotButton(QPushButton):
    """Compact technical toolbar button with icon and hover tooltips."""

    def __init__(self, text: str, tooltip: str, checkable: bool = False, parent: Optional[QWidget] = None):
        super().__init__(text, parent)
        self.setToolTip(tooltip)
        self.setCheckable(checkable)
        self.setFixedSize(26, 24)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._update_style()

    def setChecked(self, checked: bool):
        super().setChecked(checked)
        self._update_style()

    def _update_style(self):
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {DSP_THEME.BG_SURFACE_ALT};
                color: {DSP_THEME.TEXT_SECONDARY};
                border: 1px solid {DSP_THEME.BORDER_SOLID};
                border-radius: 4px;
                font-size: 11px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {DSP_THEME.BG_CARD_HOVER};
                color: {DSP_THEME.TEXT_PRIMARY};
                border: 1px solid {DSP_THEME.BORDER_HOVER};
            }}
            QPushButton:checked {{
                background-color: {DSP_THEME.INDIGO};
                color: #FFFFFF;
                border: 1px solid {DSP_THEME.BLUE};
            }}
            QPushButton:pressed {{
                background-color: {DSP_THEME.BG_PRIMARY};
            }}
        """)


class DSPScientificPlotWidget(QWidget):
    """
    Primary scientific plot canvas component for SPANDHAN DSP.
    Integrates Matplotlib with custom dark toolbar controls and coordinate inspection.
    """

    fullscreen_requested = Signal(object)  # Emits self or figure for full-screen inspection

    def __init__(
        self,
        width: float = 6.0,
        height: float = 3.5,
        dpi: int = 100,
        enable_toolbar: bool = True,
        coord_formatter: Optional[Callable[[float, float], str]] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumHeight(240)

        self._coord_formatter = coord_formatter
        self._grid_visible = True

        # Initialize Matplotlib Figure & Canvas
        self.figure = Figure(figsize=(width, height), dpi=dpi, facecolor=DSP_THEME.BG_PLOT)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.canvas.setStyleSheet(f"background-color: {DSP_THEME.BG_PLOT};")

        # Standard Matplotlib Navigation Toolbar (hidden, used under the hood for pan/zoom actions)
        self._nav_toolbar = NavigationToolbar2QT(self.canvas, self)
        self._nav_toolbar.setVisible(False)

        # Connect hover motion to coordinate readout
        self.canvas.mpl_connect("motion_notify_event", self._on_mouse_move)

        # Build Widget Layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(4)

        if enable_toolbar:
            toolbar_layout = self._build_toolbar()
            main_layout.addLayout(toolbar_layout)

        main_layout.addWidget(self.canvas, stretch=1)

    def _build_toolbar(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        bar.setContentsMargins(6, 2, 6, 2)
        bar.setSpacing(5)

        # Coordinate hover readout
        self.coord_label = QLabel("Cursor: -- , --")
        self.coord_label.setStyleSheet(
            f"color: {DSP_THEME.TEXT_MUTED}; font-family: 'Consolas', monospace; font-size: 10px;"
        )
        bar.addWidget(self.coord_label)
        bar.addStretch()

        # Reset view button
        self.reset_btn = MicroPlotButton("↺", "Reset View / Autoscale (Home)")
        self.reset_btn.clicked.connect(self.reset_view)
        bar.addWidget(self.reset_btn)

        # Pan button
        self.pan_btn = MicroPlotButton("✥", "Pan / Drag Axes", checkable=True)
        self.pan_btn.clicked.connect(self._toggle_pan)
        bar.addWidget(self.pan_btn)

        # Zoom button
        self.zoom_btn = MicroPlotButton("🔍", "Box Zoom", checkable=True)
        self.zoom_btn.clicked.connect(self._toggle_zoom)
        bar.addWidget(self.zoom_btn)

        # Grid toggle button
        self.grid_btn = MicroPlotButton("◫", "Toggle Grid", checkable=True)
        self.grid_btn.setChecked(True)
        self.grid_btn.clicked.connect(self._toggle_grid)
        bar.addWidget(self.grid_btn)

        # Export button
        self.save_btn = MicroPlotButton("💾", "Export High-Res PNG")
        self.save_btn.clicked.connect(self.export_image)
        bar.addWidget(self.save_btn)

        # Fullscreen expand button
        self.expand_btn = MicroPlotButton("⛶", "Maximize / Fullscreen Viewer")
        self.expand_btn.clicked.connect(lambda: self.fullscreen_requested.emit(self))
        bar.addWidget(self.expand_btn)

        return bar

    # -----------------------------------------------------------------------
    # Interactive Actions
    # -----------------------------------------------------------------------

    def reset_view(self):
        """Restores original plot boundaries."""
        self._nav_toolbar.home()
        self.canvas.draw_idle()

    def _toggle_pan(self):
        if self.pan_btn.isChecked():
            self.zoom_btn.setChecked(False)
            if self._nav_toolbar.mode != "pan/zoom":
                self._nav_toolbar.pan()
        else:
            if self._nav_toolbar.mode == "pan/zoom":
                self._nav_toolbar.pan()

    def _toggle_zoom(self):
        if self.zoom_btn.isChecked():
            self.pan_btn.setChecked(False)
            if self._nav_toolbar.mode != "zoom rect":
                self._nav_toolbar.zoom()
        else:
            if self._nav_toolbar.mode == "zoom rect":
                self._nav_toolbar.zoom()

    def _toggle_grid(self):
        self._grid_visible = self.grid_btn.isChecked()
        for ax in self.figure.get_axes():
            ax.grid(self._grid_visible, linestyle="--", linewidth=0.5, color=DSP_THEME.GRID_COLOR, alpha=0.7)
        self.canvas.draw_idle()

    def export_image(self):
        """Saves high-res publication figure."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Scientific Plot", "dsp_visualization.png", "PNG Image (*.png);;PDF Vector (*.pdf)"
        )
        if file_path:
            self.figure.savefig(file_path, dpi=300, facecolor=self.figure.get_facecolor(), bbox_inches="tight")

    def _on_mouse_move(self, event):
        """Updates the hover readout with scientific coordinates."""
        if event.inaxes and event.xdata is not None and event.ydata is not None:
            if self._coord_formatter:
                text = self._coord_formatter(event.xdata, event.ydata)
            else:
                text = f"X: {event.xdata:.3g}  |  Y: {event.ydata:.3g}"
            self.coord_label.setText(text)
        else:
            self.coord_label.setText("Cursor: -- , --")

    def set_coord_formatter(self, formatter: Callable[[float, float], str]):
        self._coord_formatter = formatter

    def apply_style(self, grid: bool = True):
        """Applies SPANDHAN dark scientific styling to figure and all axes."""
        apply_scientific_plot_style(self.figure, grid=grid)
        self.canvas.draw_idle()

    def clear(self):
        """Clears all figure axes."""
        self.figure.clf()
        self.canvas.draw_idle()
