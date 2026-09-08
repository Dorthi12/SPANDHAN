"""
MetricCard Widget for displaying quick engineering metrics (Title, Value, Unit, Subtitle).
"""

from PySide6.QtWidgets import QFrame, QVBoxLayout, QLabel
from PySide6.QtCore import Qt


class MetricCard(QFrame):
    """
    Compact engineering metric card.
    """
    def __init__(self, title: str, value: str = "N/A", unit: str = "", subtitle: str = "", accent_color: str = "#38BDF8", parent=None):
        super().__init__(parent)
        self.setObjectName("MetricCard")
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(2)

        self.title_label = QLabel(title)
        self.title_label.setObjectName("MetricTitle")
        
        self.value_label = QLabel(f"{value} {unit}".strip())
        self.value_label.setObjectName("MetricValue")
        if accent_color:
            self.value_label.setStyleSheet(f"color: {accent_color};")
            
        self.sub_label = QLabel(subtitle)
        self.sub_label.setObjectName("MetricSub")
        
        layout.addWidget(self.title_label)
        layout.addWidget(self.value_label)
        if subtitle:
            layout.addWidget(self.sub_label)

    def update_metric(self, value: str, unit: str = "", subtitle: str = ""):
        val_text = f"{value} {unit}".strip() if value is not None else "N/A"
        self.value_label.setText(val_text)
        if subtitle:
            self.sub_label.setText(subtitle)
