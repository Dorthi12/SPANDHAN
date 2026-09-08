"""
SectionHeader Widget for consistent title & contextual description across views.
"""

from PySide6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton
from PySide6.QtCore import Qt


class SectionHeader(QWidget):
    """
    Header component with Title, Subtitle/Description, and optional Action Buttons.
    """
    def __init__(self, title: str, subtitle: str = "", action_title: str = "", action_slot=None, parent=None):
        super().__init__(parent)
        
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 8)
        
        text_layout = QVBoxLayout()
        text_layout.setSpacing(2)
        
        self.title_label = QLabel(title)
        self.title_label.setStyleSheet("font-size: 15px; font-weight: 700; color: #F8FAFC;")
        
        self.subtitle_label = QLabel(subtitle)
        self.subtitle_label.setStyleSheet("font-size: 11px; color: #94A3B8;")
        
        text_layout.addWidget(self.title_label)
        if subtitle:
            text_layout.addWidget(self.subtitle_label)
            
        main_layout.addLayout(text_layout)
        main_layout.addStretch()
        
        if action_title and action_slot:
            self.action_btn = QPushButton(action_title)
            self.action_btn.clicked.connect(action_slot)
            main_layout.addWidget(self.action_btn)
