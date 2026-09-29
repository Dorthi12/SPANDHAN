"""
SPANDHAN — Generic Section Placeholder Page
===========================================
Displays a clean, themed placeholder card for sections awaiting future implementation.
"""

from typing import Optional
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QFrame,
    QSizePolicy,
)

from ui.styles.theme import COLORS
from ui.components.cards.metric_card import CardContainer


class PlaceholderPage(QWidget):
    """Clean scientific placeholder view for pages to be implemented in subsequent milestones."""

    def __init__(
        self,
        breadcrumb: str,
        title: str,
        subtitle: str,
        description: str,
        icon_symbol: str = "⚙",
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 24, 28, 28)
        layout.setSpacing(18)

        # Breadcrumbs
        bc_layout = QHBoxLayout()
        bc_lbl = QLabel(breadcrumb)
        bc_lbl.setStyleSheet(f"color: {COLORS.TEXT_MUTED}; font-size: 10px; font-weight: 700; letter-spacing: 1px;")
        bc_layout.addWidget(bc_lbl)
        bc_layout.addStretch()
        layout.addLayout(bc_layout)

        # Title
        t_lbl = QLabel(title)
        t_lbl.setStyleSheet(
            f"color: {COLORS.TEXT_PRIMARY}; font-size: 22px; font-weight: 800; "
            f"letter-spacing: -0.3px;"
        )
        s_lbl = QLabel(subtitle)
        s_lbl.setStyleSheet(f"color: {COLORS.TEXT_SECONDARY}; font-size: 12px;")

        layout.addWidget(t_lbl)
        layout.addWidget(s_lbl)

        # Information Card
        card = CardContainer(title="MODULE STATUS", subtitle="Pipeline Architecture")
        card_inner = QFrame()
        card_inner.setStyleSheet(
            f"background-color: #07111F; border: 1px solid {COLORS.BORDER_SUBTLE}; "
            f"border-radius: 8px; padding: 24px;"
        )
        ci_layout = QVBoxLayout(card_inner)
        ci_layout.setSpacing(12)
        ci_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        ic = QLabel(icon_symbol)
        ic.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ic.setStyleSheet(f"color: {COLORS.INDIGO_LIGHT}; font-size: 32px;")

        desc_lbl = QLabel(description)
        desc_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        desc_lbl.setStyleSheet(f"color: {COLORS.TEXT_SECONDARY}; font-size: 13px; line-height: 1.5;")
        desc_lbl.setWordWrap(True)

        status_lbl = QLabel("● Module registered in SPANDHAN shell")
        status_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        status_lbl.setStyleSheet(f"color: {COLORS.PERSIAN_GREEN}; font-size: 11px; font-weight: 600;")

        ci_layout.addWidget(ic)
        ci_layout.addWidget(desc_lbl)
        ci_layout.addWidget(status_lbl)

        card.add_widget(card_inner)
        layout.addWidget(card)
        layout.addStretch()
