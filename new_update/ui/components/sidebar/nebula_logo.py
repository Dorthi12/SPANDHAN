"""
SPANDHAN — Glowing Nebula Logo Widget
======================================
Renders an abstract cosmic nebula emblem with layered radial/conical gradients,
electric blue, violet, and deep indigo tones, soft outer glow, and refined typography.
"""

from PySide6.QtCore import Qt, QRectF, QPointF
from PySide6.QtGui import (
    QPainter,
    QColor,
    QRadialGradient,
    QConicalGradient,
    QPainterPath,
    QFont,
    QPen,
)
from PySide6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QLabel

from ui.styles.theme import COLORS


class NebulaEmblem(QWidget):
    """Custom-painted scientific glowing nebula symbol."""

    def __init__(self, parent: QWidget | None = None, size: int = 40):
        super().__init__(parent)
        self._size = size
        self.setFixedSize(size, size)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        w = float(self.width())
        h = float(self.height())
        center = QPointF(w / 2.0, h / 2.0)
        radius = min(w, h) / 2.0

        # 1. Outer faint celestial glow
        outer_grad = QRadialGradient(center, radius)
        outer_grad.setColorAt(0.0, QColor(99, 102, 241, 140))    # Indigo glow
        outer_grad.setColorAt(0.4, QColor(59, 130, 246, 70))     # Electric blue
        outer_grad.setColorAt(0.8, QColor(139, 92, 246, 25))    # Violet whisper
        outer_grad.setColorAt(1.0, QColor(7, 17, 31, 0))        # Fade to navy
        painter.setBrush(outer_grad)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(center, radius, radius)

        # 2. Asymmetric swirling nebula clouds
        painter.save()
        painter.translate(center)

        # Cloud lobe 1 (Electric Blue)
        painter.rotate(-25)
        lobe1_path = QPainterPath()
        lobe1_path.addEllipse(QRectF(-radius * 0.55, -radius * 0.40, radius * 1.1, radius * 0.8))
        g1 = QRadialGradient(QPointF(-radius * 0.15, -radius * 0.1), radius * 0.6)
        g1.setColorAt(0.0, QColor(56, 189, 248, 190))   # Cyan electric core
        g1.setColorAt(0.5, QColor(59, 130, 246, 120))   # Blue
        g1.setColorAt(1.0, QColor(67, 56, 202, 0))      # Indigo fade
        painter.setBrush(g1)
        painter.drawPath(lobe1_path)

        # Cloud lobe 2 (Violet / Magenta-purple)
        painter.rotate(65)
        lobe2_path = QPainterPath()
        lobe2_path.addEllipse(QRectF(-radius * 0.40, -radius * 0.55, radius * 0.85, radius * 1.05))
        g2 = QRadialGradient(QPointF(radius * 0.1, radius * 0.15), radius * 0.55)
        g2.setColorAt(0.0, QColor(168, 85, 247, 180))   # Vivid purple
        g2.setColorAt(0.6, QColor(129, 140, 248, 90))   # Indigo-violet
        g2.setColorAt(1.0, QColor(15, 23, 42, 0))
        painter.setBrush(g2)
        painter.drawPath(lobe2_path)

        # 3. Dynamic spiral energy filaments
        painter.rotate(-40)
        pen_glow = QPen(QColor(56, 189, 248, 160), 1.6)
        painter.setPen(pen_glow)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawArc(QRectF(-radius * 0.65, -radius * 0.65, radius * 1.3, radius * 1.3), 30 * 16, 160 * 16)

        pen_glow_inner = QPen(QColor(16, 185, 129, 150), 1.2)  # Persian green accent ring
        painter.setPen(pen_glow_inner)
        painter.drawArc(QRectF(-radius * 0.45, -radius * 0.45, radius * 0.9, radius * 0.9), 190 * 16, 130 * 16)

        painter.restore()

        # 4. Dense brilliant stellar nucleus
        core_grad = QRadialGradient(center, radius * 0.28)
        core_grad.setColorAt(0.0, QColor(255, 255, 255, 240))  # White hot core
        core_grad.setColorAt(0.3, QColor(224, 231, 255, 200))  # Pale electric
        core_grad.setColorAt(0.7, QColor(99, 102, 241, 140))   # Deep indigo
        core_grad.setColorAt(1.0, QColor(59, 130, 246, 0))
        painter.setBrush(core_grad)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(center, radius * 0.30, radius * 0.30)

        # 5. Fine starlight diffraction points
        painter.setPen(QPen(QColor(255, 255, 255, 220), 1.0))
        painter.drawPoint(QPointF(center.x() + radius * 0.38, center.y() - radius * 0.25))
        painter.setPen(QPen(QColor(56, 189, 248, 200), 1.2))
        painter.drawPoint(QPointF(center.x() - radius * 0.35, center.y() + radius * 0.32))


class NebulaLogoWidget(QWidget):
    """Top-left branding header containing the Nebula emblem and technical typography."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setFixedHeight(64)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(12)

        # Emblem
        self.emblem = NebulaEmblem(self, size=40)
        layout.addWidget(self.emblem)

        # Text labels
        text_layout = QVBoxLayout()
        text_layout.setContentsMargins(0, 0, 0, 0)
        text_layout.setSpacing(2)

        self.title_label = QLabel("SPANDHAN")
        self.title_label.setStyleSheet(
            f"color: {COLORS.TEXT_PRIMARY};"
            "font-size: 15px;"
            "font-weight: 800;"
            "letter-spacing: 1.5px;"
        )

        self.sub_label = QLabel("SIGNAL INTELLIGENCE")
        self.sub_label.setStyleSheet(
            f"color: {COLORS.PERSIAN_GREEN};"
            "font-size: 9px;"
            "font-weight: 700;"
            "letter-spacing: 1.2px;"
        )

        text_layout.addWidget(self.title_label)
        text_layout.addWidget(self.sub_label)
        layout.addLayout(text_layout)
        layout.addStretch()
