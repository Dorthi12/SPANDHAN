"""
SPANDHAN — Design System & Visual Theme
========================================
Defines the dark navy foundation, indigo & Persian green accents,
glassmorphic styling, and reusable Qt stylesheets (QSS) for SPANDHAN.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ThemeColors:
    # Foundations
    BG_PRIMARY: str = "#07111F"       # Midnight dark navy
    BG_SECONDARY: str = "#0B1930"     # Deep navy sidebar/container
    BG_CARD: str = "#0D1B2E"          # Glassmorphic card surface
    BG_CARD_HOVER: str = "#11223B"    # Card hover
    BG_INPUT: str = "#081426"         # Input / well container
    BG_SURFACE_ALT: str = "#13233E"   # Raised surface

    # Accents
    INDIGO: str = "#4338CA"           # Core indigo
    INDIGO_LIGHT: str = "#6366F1"     # Bright indigo
    INDIGO_GLOW: str = "rgba(67, 56, 202, 0.35)"

    BLUE: str = "#3B82F6"             # Electric blue
    BLUE_LIGHT: str = "#60A5FA"
    BLUE_GLOW: str = "rgba(59, 130, 246, 0.30)"

    PERSIAN_GREEN: str = "#10B981"    # Persian green / emerald
    GREEN_LIGHT: str = "#34D399"
    GREEN_GLOW: str = "rgba(16, 185, 129, 0.35)"

    CYAN: str = "#06B6D4"             # Signal cyan accent
    VIOLET: str = "#8B5CF6"           # Nebula violet accent

    # Typography
    TEXT_PRIMARY: str = "#F8FAFC"     # Slate 50 - High contrast text
    TEXT_SECONDARY: str = "#94A3B8"   # Slate 400 - Supporting labels
    TEXT_MUTED: str = "#64748B"       # Slate 500 - Metadata & timestamps
    TEXT_HIGHLIGHT: str = "#38BDF8"   # Sky blue highlight

    # Borders & Dividers
    BORDER_SUBTLE: str = "rgba(59, 130, 246, 0.15)"
    BORDER_HOVER: str = "rgba(59, 130, 246, 0.35)"
    BORDER_ACTIVE: str = "#3B82F6"
    BORDER_SOLID: str = "#1E293B"

    # Status indicators
    STATUS_READY: str = "#10B981"     # Persian green
    STATUS_PROCESSING: str = "#38BDF8"# Cyan/Sky
    STATUS_WARNING: str = "#F59E0B"   # Amber
    STATUS_ERROR: str = "#EF4444"     # Rose red


COLORS = ThemeColors()

MAIN_STYLESHEET = f"""
/* -------------------------------------------------------------
   SPANDHAN Global Application Stylesheet
   ------------------------------------------------------------- */

QMainWindow, QWidget#CentralWidget {{
    background-color: {COLORS.BG_PRIMARY};
    font-family: "Segoe UI";
    font-size: 13px;
}}

/* Scroll Areas & Scrollbars */
QScrollArea {{
    background: transparent;
    border: none;
}}

QScrollBar:vertical {{
    background-color: {COLORS.BG_PRIMARY};
    width: 8px;
    margin: 0px;
    border-radius: 4px;
}}

QScrollBar::handle:vertical {{
    background-color: #1E293B;
    min-height: 24px;
    border-radius: 4px;
}}

QScrollBar::handle:vertical:hover {{
    background-color: {COLORS.INDIGO};
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}

QScrollBar:horizontal {{
    background-color: {COLORS.BG_PRIMARY};
    height: 8px;
    border-radius: 4px;
}}

QScrollBar::handle:horizontal {{
    background-color: #1E293B;
    min-width: 24px;
    border-radius: 4px;
}}

QScrollBar::handle:horizontal:hover {{
    background-color: {COLORS.INDIGO};
}}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0px;
}}

/* Card Container Styles */
QFrame.SpandhanCard {{
    background-color: {COLORS.BG_CARD};
    border: 1px solid {COLORS.BORDER_SUBTLE};
    border-radius: 10px;
}}

QFrame.SpandhanCard:hover {{
    border: 1px solid {COLORS.BORDER_HOVER};
}}

QFrame.SpandhanCardInner {{
    background-color: {COLORS.BG_INPUT};
    border: 1px solid {COLORS.BORDER_SUBTLE};
    border-radius: 8px;
}}

/* Buttons */
QPushButton.PrimaryButton {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 {COLORS.INDIGO},
        stop:0.55 {COLORS.BLUE},
        stop:1 {COLORS.PERSIAN_GREEN});
    color: #FFFFFF;
    font-weight: 600;
    font-size: 13px;
    padding: 10px 22px;
    border-radius: 8px;
    border: none;
    letter-spacing: 0.5px;
}}

QPushButton.PrimaryButton:hover {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 {COLORS.INDIGO_LIGHT},
        stop:0.55 {COLORS.BLUE_LIGHT},
        stop:1 {COLORS.GREEN_LIGHT});
}}

QPushButton.PrimaryButton:pressed {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #3730A3,
        stop:0.55 #2563EB,
        stop:1 #059669);
}}

QPushButton.PrimaryButton:disabled {{
    background-color: #1E293B;
    color: #64748B;
    border: 1px solid #334155;
}}

QPushButton.SecondaryButton {{
    background-color: {COLORS.BG_SURFACE_ALT};
    color: {COLORS.TEXT_PRIMARY};
    border: 1px solid {COLORS.BORDER_SUBTLE};
    border-radius: 7px;
    padding: 7px 15px;
    font-size: 12px;
    font-weight: 500;
}}

QPushButton.SecondaryButton:hover {{
    background-color: {COLORS.BG_CARD_HOVER};
    border: 1px solid {COLORS.BORDER_HOVER};
    color: #FFFFFF;
}}

QPushButton.SecondaryButton:pressed {{
    background-color: {COLORS.BG_PRIMARY};
}}

/* Tooltips */
QToolTip {{
    background-color: {COLORS.BG_SECONDARY};
    color: {COLORS.TEXT_PRIMARY};
    border: 1px solid {COLORS.BORDER_HOVER};
    border-radius: 5px;
    padding: 5px 8px;
    font-size: 11px;
}}

/* Status Bar */
QStatusBar {{
    background-color: {COLORS.BG_SECONDARY};
    border-top: 1px solid {COLORS.BORDER_SUBTLE};
    color: {COLORS.TEXT_SECONDARY};
    font-size: 11px;
    min-height: 28px;
}}
"""
