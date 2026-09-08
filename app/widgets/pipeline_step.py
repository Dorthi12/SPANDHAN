"""
PipelineStep Widget for displaying pipeline progression steps on the Dashboard.
"""

from PySide6.QtWidgets import QFrame, QVBoxLayout, QLabel
from PySide6.QtCore import Qt, Signal


class PipelineStepCard(QFrame):
    """
    Clickable visual progression step card for the analysis pipeline.
    """
    clicked = Signal(str)

    COLOR_MAP = {
        "Complete": ("#065F46", "#10B981", "✓ Complete"),
        "Ready": ("#1E3A8A", "#60A5FA", "● Ready"),
        "Not started": ("#1E293B", "#64748B", "○ Not Started"),
        "Warning": ("#78350F", "#FBBF24", "⚠ Warning"),
    }

    def __init__(self, step_name: str, step_desc: str = "", status: str = "Not started", parent=None):
        super().__init__(parent)
        self.step_name = step_name
        self.setObjectName("CardPanel")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(4)
        
        self.name_label = QLabel(step_name)
        self.name_label.setStyleSheet("font-size: 12px; font-weight: 700; color: #F8FAFC;")
        
        self.desc_label = QLabel(step_desc)
        self.desc_label.setStyleSheet("font-size: 10px; color: #94A3B8;")
        
        self.status_label = QLabel()
        self.set_status(status)
        
        layout.addWidget(self.name_label)
        layout.addWidget(self.desc_label)
        layout.addWidget(self.status_label)

    def set_status(self, status: str):
        bg, fg, text = self.COLOR_MAP.get(status, ("#1E293B", "#64748B", status))
        self.status_label.setText(text)
        self.status_label.setStyleSheet(f"font-size: 10px; font-weight: 600; color: {fg}; margin-top: 2px;")
        self.setStyleSheet(f"""
            QFrame#CardPanel {{
                background-color: #1E293B;
                border: 1px solid {fg}60;
                border-radius: 6px;
            }}
            QFrame#CardPanel:hover {{
                border-color: {fg};
                background-color: #334155;
            }}
        """)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.step_name)
        super().mousePressEvent(event)
