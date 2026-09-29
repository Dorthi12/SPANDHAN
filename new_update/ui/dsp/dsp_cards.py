"""
SPANDHAN — Scientific Visualization Cards & Diagnostic Containers
===================================================================
Provides modular, glassmorphic scientific cards for DSP visualizations:
  - Header with title, subtitle, module badge, and action buttons (collapse, fullscreen)
  - Interactive plot/matrix body
  - Technical metadata strip displaying MATLAB-computed physical metrics
  - Diagnostic container for skipped and failed analyses
  - Fullscreen modal dialog for high-resolution inspection
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any
from PySide6.QtCore import Qt, QRectF, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QCursor
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QPushButton,
    QFrame,
    QDialog,
    QSizePolicy,
    QScrollArea,
)

from ui.styles.theme import COLORS
from ui.dsp.dsp_theme import DSP_THEME, get_chip_stylesheet
from ui.dsp.dsp_plot_widget import DSPScientificPlotWidget


class MetadataMetricChip(QFrame):
    """Compact technical chip showing a single parameter/metric and its physical unit."""

    def __init__(self, label: str, value: str, unit: str = "", parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {DSP_THEME.BG_INPUT};
                border: 1px solid {DSP_THEME.BORDER_SOLID};
                border-radius: 5px;
                padding: 3px 8px;
            }}
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 3, 6, 3)
        layout.setSpacing(6)

        lbl = QLabel(label.upper())
        lbl.setStyleSheet(f"color: {DSP_THEME.TEXT_MUTED}; font-size: 9px; font-weight: 700; letter-spacing: 0.5px;")

        val_str = f"{value} {unit}".strip()
        val = QLabel(val_str)
        val.setStyleSheet(f"color: {DSP_THEME.TEXT_PRIMARY}; font-family: 'Consolas', monospace; font-size: 11px; font-weight: 600;")

        layout.addWidget(lbl)
        layout.addWidget(val)


class ScientificVisualizationCard(QFrame):
    """
    Standard scientific visualization container for DSP analysis.
    Hosts title, description, badge, plot widget, collapse animation, and metadata metrics.
    """

    def __init__(
        self,
        title: str,
        subtitle: str = "",
        module_tag: str = "",
        span_full_width: bool = False,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.setObjectName("ScientificVisualizationCard")
        self.span_full_width = span_full_width
        self._is_collapsed = False

        self.setStyleSheet(f"""
            QFrame#ScientificVisualizationCard {{
                background-color: {DSP_THEME.BG_CARD};
                border: 1px solid {DSP_THEME.BORDER_SUBTLE};
                border-radius: 10px;
            }}
            QFrame#ScientificVisualizationCard:hover {{
                border: 1px solid {DSP_THEME.BORDER_HOVER};
            }}
        """)

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(14, 12, 14, 14)
        self.main_layout.setSpacing(10)

        # 1. Header Layout
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(8)

        # Title & Subtitle column
        title_col = QVBoxLayout()
        title_col.setContentsMargins(0, 0, 0, 0)
        title_col.setSpacing(1)

        title_row = QHBoxLayout()
        title_row.setContentsMargins(0, 0, 0, 0)
        title_row.setSpacing(8)

        self.title_lbl = QLabel(title.upper())
        self.title_lbl.setStyleSheet(
            f"color: {DSP_THEME.TEXT_PRIMARY}; font-size: 12px; font-weight: 700; letter-spacing: 0.8px;"
        )
        title_row.addWidget(self.title_lbl)

        if module_tag:
            tag_lbl = QLabel(module_tag.upper())
            tag_lbl.setStyleSheet(f"""
                background-color: {DSP_THEME.BG_SURFACE_ALT};
                color: {DSP_THEME.CYAN};
                border: 1px solid rgba(56, 189, 248, 0.3);
                border-radius: 4px;
                padding: 1px 6px;
                font-size: 9px;
                font-weight: 700;
            """)
            title_row.addWidget(tag_lbl)

        title_row.addStretch()
        title_col.addLayout(title_row)

        if subtitle:
            self.sub_lbl = QLabel(subtitle)
            self.sub_lbl.setStyleSheet(f"color: {DSP_THEME.TEXT_MUTED}; font-size: 10px;")
            title_col.addWidget(self.sub_lbl)

        header.addLayout(title_col, stretch=1)

        # Right Header Actions: Expand/Fullscreen and Collapse
        self.collapse_btn = QPushButton("▼")
        self.collapse_btn.setFixedSize(24, 24)
        self.collapse_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.collapse_btn.setToolTip("Collapse / Expand Card")
        self.collapse_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {DSP_THEME.BG_SURFACE_ALT};
                color: {DSP_THEME.TEXT_SECONDARY};
                border: 1px solid {DSP_THEME.BORDER_SOLID};
                border-radius: 4px;
                font-size: 9px;
            }}
            QPushButton:hover {{
                background-color: {DSP_THEME.INDIGO};
                color: #FFFFFF;
            }}
        """)
        self.collapse_btn.clicked.connect(self.toggle_collapsed)
        header.addWidget(self.collapse_btn)

        self.main_layout.addLayout(header)

        # 2. Content Widget (Plot/Visual Area)
        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(8)

        self.plot_container = QWidget()
        self.plot_layout = QVBoxLayout(self.plot_container)
        self.plot_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.addWidget(self.plot_container, stretch=1)

        # 3. Bottom Metadata Strip
        self.meta_strip = QFrame()
        self.meta_strip.setStyleSheet(f"""
            QFrame {{
                background-color: {DSP_THEME.BG_PLOT};
                border-top: 1px solid {DSP_THEME.BORDER_SOLID};
                border-radius: 0px 0px 8px 8px;
            }}
        """)
        self.meta_layout = QHBoxLayout(self.meta_strip)
        self.meta_layout.setContentsMargins(8, 6, 8, 6)
        self.meta_layout.setSpacing(8)
        self.content_layout.addWidget(self.meta_strip)

        self.main_layout.addWidget(self.content_widget, stretch=1)

    def set_plot_widget(self, plot_widget: QWidget):
        """Attaches the main scientific plot widget to the card."""
        while self.plot_layout.count():
            item = self.plot_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.plot_layout.addWidget(plot_widget)

    def set_metrics(self, metrics: List[Tuple[str, str, str]]):
        """
        Populates the bottom metadata strip with (label, value, unit) tuples.
        """
        while self.meta_layout.count():
            item = self.meta_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not metrics:
            self.meta_strip.setVisible(False)
            return

        self.meta_strip.setVisible(True)
        for label, val, unit in metrics:
            chip = MetadataMetricChip(label, val, unit, self.meta_strip)
            self.meta_layout.addWidget(chip)
        self.meta_layout.addStretch()

    def toggle_collapsed(self):
        """Expands or collapses the visualization content."""
        self._is_collapsed = not self._is_collapsed
        self.content_widget.setVisible(not self._is_collapsed)
        self.collapse_btn.setText("▲" if self._is_collapsed else "▼")


class SkippedOrFailedItem(QFrame):
    """Entry displaying an analysis that was either intentionally skipped or failed."""

    def __init__(self, module_name: str, status_type: str, reason: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        is_failed = status_type.lower() == "failed"

        border_color = "rgba(239, 68, 68, 0.4)" if is_failed else "rgba(59, 130, 246, 0.2)"
        bg_color = "rgba(239, 68, 68, 0.08)" if is_failed else DSP_THEME.BG_INPUT
        status_color = DSP_THEME.ROSE if is_failed else DSP_THEME.TEXT_MUTED
        status_text = "FAILED" if is_failed else "NOT EXECUTED"

        self.setStyleSheet(f"""
            QFrame {{
                background-color: {bg_color};
                border: 1px solid {border_color};
                border-radius: 6px;
                padding: 6px 10px;
            }}
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(10)

        # Module name
        name_lbl = QLabel(module_name)
        name_lbl.setStyleSheet(f"color: {DSP_THEME.TEXT_PRIMARY}; font-size: 11px; font-weight: 700;")
        layout.addWidget(name_lbl)

        # Status badge
        badge = QLabel(status_text)
        badge.setStyleSheet(f"""
            color: {status_color};
            background-color: {DSP_THEME.BG_SURFACE_ALT};
            border: 1px solid {status_color};
            border-radius: 3px;
            font-size: 9px;
            font-weight: 700;
            padding: 1px 5px;
        """)
        layout.addWidget(badge)

        # Reason
        reason_lbl = QLabel(f"Reason: {reason}")
        reason_lbl.setStyleSheet(f"color: {DSP_THEME.TEXT_SECONDARY}; font-size: 10px;")
        layout.addWidget(reason_lbl, stretch=1)


class DiagnosticSection(QFrame):
    """Collapsible container for skipped and failed analysis diagnostics."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("DiagnosticSection")
        self.setStyleSheet(f"""
            QFrame#DiagnosticSection {{
                background-color: {DSP_THEME.BG_CARD};
                border: 1px solid {DSP_THEME.BORDER_SUBTLE};
                border-radius: 9px;
            }}
        """)

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(14, 12, 14, 14)
        self.main_layout.setSpacing(8)

        # Header
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        self.title_lbl = QLabel("DSP DIAGNOSTICS & SKIPPED ANALYSES")
        self.title_lbl.setStyleSheet(
            f"color: {DSP_THEME.TEXT_SECONDARY}; font-size: 11px; font-weight: 700; letter-spacing: 0.8px;"
        )
        header.addWidget(self.title_lbl)
        header.addStretch()

        self.count_badge = QLabel("0 Items")
        self.count_badge.setStyleSheet(f"color: {DSP_THEME.TEXT_MUTED}; font-size: 10px; font-weight: 600;")
        header.addWidget(self.count_badge)

        self.main_layout.addLayout(header)

        # List of items
        self.items_container = QWidget()
        self.items_layout = QVBoxLayout(self.items_container)
        self.items_layout.setContentsMargins(0, 0, 0, 0)
        self.items_layout.setSpacing(6)
        self.main_layout.addWidget(self.items_container)

    def set_diagnostics(self, skipped: List[Tuple[str, str]], failed: List[Tuple[str, str]]):
        """
        Updates the diagnostic entries.
        skipped: list of (module_name, reason)
        failed: list of (module_name, error_message)
        """
        while self.items_layout.count():
            item = self.items_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        total = len(skipped) + len(failed)
        if total == 0:
            self.setVisible(False)
            return

        self.setVisible(True)
        self.count_badge.setText(f"{total} item{'s' if total > 1 else ''}")

        for name, err in failed:
            self.items_layout.addWidget(SkippedOrFailedItem(name, "failed", err, self))

        for name, reason in skipped:
            self.items_layout.addWidget(SkippedOrFailedItem(name, "skipped", reason, self))


class FullscreenPlotDialog(QDialog):
    """High-resolution modal window for fullscreen plot exploration."""

    def __init__(self, title: str, plot_generator_func, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle(f"SPANDHAN — {title} (Fullscreen Inspection)")
        self.resize(1100, 720)
        self.setStyleSheet(f"background-color: {DSP_THEME.BG_PRIMARY};")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        # Top bar
        top_bar = QHBoxLayout()
        title_lbl = QLabel(title)
        title_lbl.setStyleSheet(f"color: {DSP_THEME.TEXT_PRIMARY}; font-size: 14px; font-weight: bold;")
        top_bar.addWidget(title_lbl)
        top_bar.addStretch()

        close_btn = QPushButton("✕ Close")
        close_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {DSP_THEME.BG_SURFACE_ALT};
                color: {DSP_THEME.TEXT_PRIMARY};
                border: 1px solid {DSP_THEME.BORDER_SOLID};
                border-radius: 6px;
                padding: 6px 16px;
                font-weight: 600;
            }}
            QPushButton:hover {{
                background-color: {DSP_THEME.BG_CARD_HOVER};
                border: 1px solid {DSP_THEME.BORDER_HOVER};
            }}
        """)
        close_btn.clicked.connect(self.accept)
        top_bar.addWidget(close_btn)
        layout.addLayout(top_bar)

        # Expanded Plot Widget
        self.plot_widget = DSPScientificPlotWidget(width=10.0, height=6.0, dpi=120, parent=self)
        layout.addWidget(self.plot_widget, stretch=1)

        # Call generation function
        plot_generator_func(self.plot_widget)
