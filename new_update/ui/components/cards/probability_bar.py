"""
SPANDHAN — Class Probability Distribution Widget
=================================================
Renders the multiclass probability bars for:
  - Impulse
  - Sinusoidal
  - White Noise
  - Step
  - Chirp
Highlights the winner with an Indigo-Blue-Persian Green gradient while keeping
all five classes visible with numeric readouts.
"""

from typing import Optional
from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QPainter, QColor, QLinearGradient, QFont
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
)

from ui.styles.theme import COLORS


class ProbabilityBarItem(QWidget):
    """Single horizontal probability bar with label, value, and gradient progress."""

    def __init__(
        self,
        class_name: str,
        display_name: str,
        probability: float = 0.0,
        is_highest: bool = False,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.class_name = class_name
        self.display_name = display_name
        self._probability = max(0.0, min(1.0, float(probability)))
        self._is_highest = is_highest

        self.setFixedHeight(30)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_probability(self, prob: float, is_highest: bool = False):
        self._probability = max(0.0, min(1.0, float(prob)))
        self._is_highest = is_highest
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = float(self.width())
        h = float(self.height())

        # Layout boundaries
        label_width = 110.0
        val_width = 55.0
        bar_left = label_width + 10.0
        bar_width = max(10.0, w - bar_left - val_width - 15.0)
        bar_height = 8.0
        bar_top = (h - bar_height) / 2.0

        # 1. Class Name Text
        font = painter.font()
        font.setPointSize(10)
        font.setWeight(QFont.Weight.Bold if self._is_highest else QFont.Weight.Medium)
        painter.setFont(font)

        if self._is_highest:
            painter.setPen(QColor(COLORS.TEXT_PRIMARY))
        else:
            painter.setPen(QColor(COLORS.TEXT_SECONDARY))

        painter.drawText(0, int(h / 2.0 + 4), self.display_name)

        # 2. Background track
        track_rect = QRectF(bar_left, bar_top, bar_width, bar_height)
        painter.setBrush(QColor("#081426"))
        painter.setPen(QColor("#1E293B"))
        painter.drawRoundedRect(track_rect, 4, 4)

        # 3. Active Fill Bar
        fill_w = bar_width * self._probability
        if fill_w > 1.5:
            fill_rect = QRectF(bar_left, bar_top, fill_w, bar_height)
            if self._is_highest:
                grad = QLinearGradient(bar_left, 0, bar_left + bar_width, 0)
                grad.setColorAt(0.0, QColor(COLORS.INDIGO_LIGHT))
                grad.setColorAt(0.5, QColor(COLORS.BLUE))
                grad.setColorAt(1.0, QColor(COLORS.PERSIAN_GREEN))
                painter.setBrush(grad)
                painter.setPen(Qt.PenStyle.NoPen)
            else:
                grad = QLinearGradient(bar_left, 0, bar_left + bar_width, 0)
                grad.setColorAt(0.0, QColor("#334155"))
                grad.setColorAt(1.0, QColor("#475569"))
                painter.setBrush(grad)
                painter.setPen(Qt.PenStyle.NoPen)

            painter.drawRoundedRect(fill_rect, 4, 4)

        # 4. Numeric Probability Text (e.g. 0.96 or 96.0%)
        font.setPointSize(9)
        font.setFamily("Consolas")
        painter.setFont(font)

        if self._is_highest:
            painter.setPen(QColor(COLORS.PERSIAN_GREEN))
        else:
            painter.setPen(QColor(COLORS.TEXT_MUTED))

        val_str = f"{self._probability:.4f}"
        val_x = bar_left + bar_width + 12.0
        painter.drawText(int(val_x), int(h / 2.0 + 4), val_str)


class ProbabilityDistributionWidget(QWidget):
    """Container managing all 5 class probability bars."""

    CLASSES = [
        ("impulse", "Impulse"),
        ("sinusoidal", "Sinusoidal"),
        ("white_noise", "White Noise"),
        ("step", "Step"),
        ("chirp", "Chirp"),
    ]

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(6)

        self._items: dict[str, ProbabilityBarItem] = {}

        for key, display_name in self.CLASSES:
            item = ProbabilityBarItem(
                class_name=key,
                display_name=display_name,
                probability=0.0,
                is_highest=False,
                parent=self,
            )
            self._items[key] = item
            layout.addWidget(item)

    def set_probabilities(self, probs: dict[str, float]):
        """
        Update the probabilities.
        probs dict format: {'impulse': 0.002, 'sinusoidal': 0.015, ...}
        """
        if not probs:
            self.reset()
            return

        highest_key = max(probs.items(), key=lambda x: x[1])[0] if probs else ""

        for key, item in self._items.items():
            prob = float(probs.get(key, 0.0))
            is_winner = (key == highest_key and prob > 0.0)
            item.set_probability(prob, is_highest=is_winner)

    def reset(self):
        for item in self._items.values():
            item.set_probability(0.0, is_highest=False)
