"""
SPANDHAN — Reusable Metric & Card Components
=============================================
Provides styled glassmorphic containers, metric rows, and technical key-value cards.
"""

from typing import Optional
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QFrame,
    QGridLayout,
    QPushButton,
)

from ui.styles.theme import COLORS


class CardContainer(QFrame):
    """
    Standard glassmorphic card for SPANDHAN workstation.
    Includes title, optional subtitle, optional header action, and content area.
    """

    def __init__(
        self,
        title: str = "",
        subtitle: str = "",
        collapsible: bool = False,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.setObjectName("SpandhanCard")
        self.setProperty("class", "SpandhanCard")
        self.setStyleSheet(
            f"QFrame#SpandhanCard {{ background-color: {COLORS.BG_CARD}; "
            f"border: 1px solid {COLORS.BORDER_SUBTLE}; "
            f"border-radius: 9px; }} "
            f"QFrame#SpandhanCard:hover {{ border: 1px solid {COLORS.BORDER_HOVER}; }}"
        )

        self._collapsible = collapsible
        self._is_collapsed = False

        self._main_layout = QVBoxLayout(self)
        self._main_layout.setContentsMargins(16, 14, 16, 16)
        self._main_layout.setSpacing(12)

        # Header section
        if title:
            self._header_layout = QHBoxLayout()
            self._header_layout.setContentsMargins(0, 0, 0, 0)
            self._header_layout.setSpacing(8)

            title_col = QVBoxLayout()
            title_col.setContentsMargins(0, 0, 0, 0)
            title_col.setSpacing(1)

            self.title_lbl = QLabel(title.upper())
            self.title_lbl.setStyleSheet(
                f"color: {COLORS.TEXT_PRIMARY}; font-size: 11px; font-weight: 700; "
                f"letter-spacing: 1.0px;"
            )
            title_col.addWidget(self.title_lbl)

            if subtitle:
                self.sub_lbl = QLabel(subtitle)
                self.sub_lbl.setStyleSheet(
                    f"color: {COLORS.TEXT_MUTED}; font-size: 10px; font-weight: 500;"
                )
                title_col.addWidget(self.sub_lbl)

            self._header_layout.addLayout(title_col)
            self._header_layout.addStretch()

            if collapsible:
                self.collapse_btn = QPushButton("▼")
                self.collapse_btn.setFixedSize(24, 24)
                self.collapse_btn.setCursor(Qt.CursorShape.PointingHandCursor)
                self.collapse_btn.setStyleSheet(
                    f"QPushButton {{ background-color: {COLORS.BG_SURFACE_ALT}; "
                    f"color: {COLORS.TEXT_SECONDARY}; border: 1px solid {COLORS.BORDER_SUBTLE}; "
                    f"border-radius: 4px; font-size: 10px; }} "
                    f"QPushButton:hover {{ background-color: {COLORS.INDIGO}; color: #FFF; }}"
                )
                self.collapse_btn.clicked.connect(self.toggle_collapsed)
                self._header_layout.addWidget(self.collapse_btn)

            self._main_layout.addLayout(self._header_layout)

        # Body container
        self.body_widget = QWidget()
        self.body_layout = QVBoxLayout(self.body_widget)
        self.body_layout.setContentsMargins(0, 0, 0, 0)
        self.body_layout.setSpacing(8)
        self._main_layout.addWidget(self.body_widget, stretch=1)

    def add_widget(self, widget: QWidget):
        self.body_layout.addWidget(widget)

    def add_layout(self, layout):
        self.body_layout.addLayout(layout)

    def toggle_collapsed(self):
        if not self._collapsible:
            return
        self._is_collapsed = not self._is_collapsed
        self.body_widget.setVisible(not self._is_collapsed)
        self.collapse_btn.setText("▶" if self._is_collapsed else "▼")


class MetricGrid(QWidget):
    """
    Two-column technical key-value grid for signal metrics and features.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._layout = QGridLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setHorizontalSpacing(24)
        self._layout.setVerticalSpacing(8)
        self._row = 0
        self._value_labels: dict[str, QLabel] = {}

    def add_metric(self, key: str, label: str, default_value: str = "—", unit: str = ""):
        # Key label (left)
        lbl_key = QLabel(label)
        lbl_key.setStyleSheet(f"color: {COLORS.TEXT_SECONDARY}; font-size: 11px; font-weight: 500;")

        # Value label (right)
        val_text = f"{default_value} {unit}".strip() if unit else default_value
        lbl_val = QLabel(val_text)
        lbl_val.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        lbl_val.setStyleSheet(
            f"color: {COLORS.TEXT_PRIMARY}; font-size: 11px; font-weight: 600; "
            f"font-family: 'Consolas', 'Courier New', monospace;"
        )

        self._layout.addWidget(lbl_key, self._row, 0)
        self._layout.addWidget(lbl_val, self._row, 1)
        self._value_labels[key] = lbl_val
        self._row += 1

    def set_value(self, key: str, value: str, unit: str = ""):
        if key in self._value_labels:
            val_text = f"{value} {unit}".strip() if unit else value
            self._value_labels[key].setText(val_text)

    def reset_all(self, default_value: str = "—"):
        for lbl in self._value_labels.values():
            lbl.setText(default_value)
