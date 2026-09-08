"""
Application State Widgets: EmptyState, LoadingState, ErrorState.
"""

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton
from PySide6.QtCore import Qt


class EmptyStateWidget(QWidget):
    """
    Informative empty state card displayed when no signal is loaded.
    """
    def __init__(self, title: str = "No Signal Loaded", message: str = "Load or generate a digital signal to begin analysis.", action_title: str = "Load Signal", action_slot=None, parent=None):
        super().__init__(parent)
        
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setContentsMargins(30, 40, 30, 40)
        layout.setSpacing(10)
        
        icon_lbl = QLabel("📡")
        icon_lbl.setStyleSheet("font-size: 40px;")
        icon_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        title_lbl = QLabel(title)
        title_lbl.setStyleSheet("font-size: 15px; font-weight: 700; color: #F8FAFC;")
        title_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        msg_lbl = QLabel(message)
        msg_lbl.setStyleSheet("font-size: 12px; color: #94A3B8; max-width: 400px;")
        msg_lbl.setWordWrap(True)
        msg_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        layout.addWidget(icon_lbl)
        layout.addWidget(title_lbl)
        layout.addWidget(msg_lbl)
        
        if action_title and action_slot:
            btn = QPushButton(action_title)
            btn.setObjectName("PrimaryButton")
            btn.clicked.connect(action_slot)
            layout.addWidget(btn, alignment=Qt.AlignmentFlag.AlignCenter)


class LoadingStateWidget(QWidget):
    """
    Processing/loading spinner state.
    """
    def __init__(self, message: str = "Computing DSP Analysis...", parent=None):
        super().__init__(parent)
        
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(10)
        
        icon_lbl = QLabel("⟳")
        icon_lbl.setStyleSheet("font-size: 36px; color: #60A5FA;")
        icon_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        msg_lbl = QLabel(message)
        msg_lbl.setStyleSheet("font-size: 13px; font-weight: 600; color: #E2E8F0;")
        msg_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        layout.addWidget(icon_lbl)
        layout.addWidget(msg_lbl)


class ErrorStateWidget(QWidget):
    """
    User-friendly error notification widget with troubleshooting steps.
    Never exposes raw tracebacks to normal users.
    """
    def __init__(self, operation: str, problem: str, action: str, parent=None):
        super().__init__(parent)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(8)
        
        self.setStyleSheet("background-color: #450A0A; border: 1px solid #991B1B; border-radius: 8px;")
        
        title_lbl = QLabel(f"✕ {operation} Failed")
        title_lbl.setStyleSheet("font-size: 14px; font-weight: 700; color: #FCA5A5;")
        
        prob_lbl = QLabel(f"<b>Problem:</b> {problem}")
        prob_lbl.setStyleSheet("font-size: 12px; color: #FECACA;")
        prob_lbl.setWordWrap(True)
        
        act_lbl = QLabel(f"<b>Suggested Action:</b> {action}")
        act_lbl.setStyleSheet("font-size: 12px; color: #F87171;")
        act_lbl.setWordWrap(True)
        
        layout.addWidget(title_lbl)
        layout.addWidget(prob_lbl)
        layout.addWidget(act_lbl)
