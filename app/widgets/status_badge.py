"""
Status Badge Widget for displaying state pills (Ready, Processing, Complete, Warning, Error).
"""

from PySide6.QtWidgets import QLabel
from PySide6.QtCore import Qt


class StatusBadge(QLabel):
    """
    Colored Pill Badge indicating status.
    """
    COLOR_MAP = {
        "Ready": ("#065F46", "#34D399", "● Ready"),
        "Complete": ("#065F46", "#10B981", "✓ Complete"),
        "Processing": ("#1E3A8A", "#60A5FA", "⟳ Processing..."),
        "Warning": ("#78350F", "#FBBF24", "⚠ Warning"),
        "Error": ("#7F1D1D", "#F87171", "✕ Error"),
        "Not started": ("#1E293B", "#94A3B8", "○ Not Started"),
        "Loaded": ("#065F46", "#34D399", "● ML Model: Loaded"),
        "Not Loaded": ("#334155", "#94A3B8", "○ ML Model: Not Loaded"),
    }

    def __init__(self, status: str = "Ready", parent=None):
        super().__init__(parent)
        self.set_status(status)

    def set_status(self, status: str):
        bg, fg, text = self.COLOR_MAP.get(status, ("#1E293B", "#94A3B8", status))
        self.setText(text)
        self.setStyleSheet(f"""
            QLabel {{
                background-color: {bg};
                color: {fg};
                border: 1px solid {fg}40;
                border-radius: 12px;
                padding: 3px 10px;
                font-size: 11px;
                font-weight: 600;
            }}
        """)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
