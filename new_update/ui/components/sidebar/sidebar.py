"""
SPANDHAN — Persistent Sidebar Navigation
=========================================
Strictly implements the scientific workstation navigation specification:
- ANALYSIS: Preprocessing, Audio ML, Image ML, Audio DSP, Image DSP
- RESULTS: Analysis Results
- SYSTEM: About
No Dashboard, No Settings.
Features active indicator bar, indigo-green gradient highlight, and clean hover states.
"""

from typing import Optional
from PySide6.QtCore import Qt, Signal, QRectF
from PySide6.QtGui import QPainter, QColor, QLinearGradient, QFont
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QPushButton,
    QFrame,
    QSizePolicy,
)

from ui.styles.theme import COLORS
from ui.components.sidebar.nebula_logo import NebulaLogoWidget


class NavItemButton(QPushButton):
    """Custom sidebar navigation button with active accent indicator and gradient."""

    def __init__(self, key: str, label: str, icon_symbol: str = "◆", parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.key = key
        self.label_text = label
        self.icon_symbol = icon_symbol
        self._is_active = False

        self.setFixedHeight(38)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_active(self, active: bool):
        self._is_active = active
        self.update()

    def is_active(self) -> bool:
        return self._is_active

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = self.rect()

        if self._is_active:
            # Rounded rectangular active background with subtle indigo->green gradient
            grad = QLinearGradient(0, 0, rect.width(), 0)
            grad.setColorAt(0.0, QColor(67, 56, 202, 70))      # Indigo translucent
            grad.setColorAt(0.7, QColor(59, 130, 246, 45))     # Blue
            grad.setColorAt(1.0, QColor(16, 185, 129, 30))     # Persian green faint
            painter.setBrush(grad)
            painter.setPen(QColor(59, 130, 246, 60))
            painter.drawRoundedRect(rect.adjusted(2, 2, -2, -2), 6, 6)

            # Left-side active indicator bar (Persian Green to Blue)
            bar_grad = QLinearGradient(0, 0, 0, rect.height())
            bar_grad.setColorAt(0.0, QColor(67, 56, 202))
            bar_grad.setColorAt(0.5, QColor(59, 130, 246))
            bar_grad.setColorAt(1.0, QColor(16, 185, 129))
            painter.setBrush(bar_grad)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(QRectF(2, 6, 3.5, rect.height() - 12), 2, 2)

            # Active text color
            text_color = QColor(COLORS.TEXT_PRIMARY)
            icon_color = QColor(COLORS.PERSIAN_GREEN)
            font_weight = QFont.Weight.DemiBold
        else:
            if self.underMouse():
                # Hover state
                painter.setBrush(QColor(17, 34, 59, 110))
                painter.setPen(QColor(59, 130, 246, 40))
                painter.drawRoundedRect(rect.adjusted(2, 2, -2, -2), 6, 6)
                text_color = QColor("#E2E8F0")
                icon_color = QColor(COLORS.BLUE_LIGHT)
                font_weight = QFont.Weight.Normal
            else:
                text_color = QColor(COLORS.TEXT_SECONDARY)
                icon_color = QColor("#475569")
                font_weight = QFont.Weight.Normal

        # Draw icon glyph
        font = painter.font()
        font.setPointSize(9)
        painter.setFont(font)
        painter.setPen(icon_color)
        painter.drawText(16, 23, self.icon_symbol)

        # Draw label
        font.setPointSize(10)
        font.setWeight(font_weight)
        painter.setFont(font)
        painter.setPen(text_color)
        painter.drawText(34, 23, self.label_text)


class Sidebar(QWidget):
    """
    Fixed Left Sidebar for the SPANDHAN Application.
    Emits `page_changed(key: str)` when navigation items are clicked.
    """

    page_changed = Signal(str)

    def __init__(self, active_page: str = "image_ml", parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setFixedWidth(240)
        self.setObjectName("SpandhanSidebar")
        self.setStyleSheet(
            f"QWidget#SpandhanSidebar {{ background-color: {COLORS.BG_SECONDARY}; "
            f"border-right: 1px solid {COLORS.BORDER_SUBTLE}; }}"
        )

        self._buttons: dict[str, NavItemButton] = {}
        self._current_page = active_page

        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 12, 10, 16)
        layout.setSpacing(4)

        # Top branding: Glowing Nebula Logo
        self.logo_widget = NebulaLogoWidget(self)
        layout.addWidget(self.logo_widget)

        # Thin divider below logo
        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setStyleSheet(f"background-color: {COLORS.BORDER_SUBTLE}; max-height: 1px; margin: 6px 4px 10px 4px;")
        layout.addWidget(divider)

        # ---------------------------------------------------------------------
        # SECTION: ANALYSIS
        # ---------------------------------------------------------------------
        layout.addWidget(self._create_section_header("ANALYSIS"))

        self._add_nav_item("preprocessing", "Preprocessing", "◫", layout)
        self._add_nav_item("audio_ml", "Audio ML", "〰", layout)
        self._add_nav_item("image_ml", "Image ML", "▦", layout)
        self._add_nav_item("audio_dsp", "Audio DSP", "∿", layout)
        self._add_nav_item("image_dsp", "Image DSP", "▨", layout)

        layout.addSpacing(14)

        # ---------------------------------------------------------------------
        # SECTION: RESULTS
        # ---------------------------------------------------------------------
        layout.addWidget(self._create_section_header("RESULTS"))
        self._add_nav_item("results", "Analysis Results", "◷", layout)

        layout.addSpacing(14)

        # ---------------------------------------------------------------------
        # SECTION: SYSTEM
        # ---------------------------------------------------------------------
        layout.addWidget(self._create_section_header("SYSTEM"))
        self._add_nav_item("about", "About", "ⓘ", layout)

        layout.addStretch()

        # Version & Architecture Footer
        footer_frame = QFrame()
        footer_layout = QVBoxLayout(footer_frame)
        footer_layout.setContentsMargins(8, 8, 8, 4)
        footer_layout.setSpacing(2)

        sys_label = QLabel("SPANDHAN Workstation")
        sys_label.setStyleSheet(f"color: {COLORS.TEXT_MUTED}; font-size: 10px; font-weight: 600;")
        arch_label = QLabel("PyTorch CNN + MATLAB DSP")
        arch_label.setStyleSheet(f"color: {COLORS.TEXT_MUTED}; font-size: 9px;")

        footer_layout.addWidget(sys_label)
        footer_layout.addWidget(arch_label)
        layout.addWidget(footer_frame)

        # Set initial active page
        self.set_active_page(self._current_page)

    def _create_section_header(self, title: str) -> QLabel:
        lbl = QLabel(title)
        lbl.setStyleSheet(
            f"color: {COLORS.TEXT_MUTED};"
            "font-size: 9px;"
            "font-weight: 700;"
            "letter-spacing: 1.5px;"
            "padding: 6px 8px 3px 8px;"
        )
        return lbl

    def _add_nav_item(self, key: str, label: str, icon_symbol: str, layout: QVBoxLayout):
        btn = NavItemButton(key=key, label=label, icon_symbol=icon_symbol, parent=self)
        btn.clicked.connect(lambda: self._on_button_clicked(key))
        self._buttons[key] = btn
        layout.addWidget(btn)

    def _on_button_clicked(self, key: str):
        self.set_active_page(key)
        self.page_changed.emit(key)

    def set_active_page(self, key: str):
        self._current_page = key
        for k, btn in self._buttons.items():
            btn.set_active(k == key)
