"""
TopBar Widget for Spandhan persistent header.
"""

from PySide6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton
from PySide6.QtCore import Qt, Signal
from app.widgets.status_badge import StatusBadge
from core.session import session_manager


class TopBar(QWidget):
    """
    Persistent Header Bar showing Page Title, Current Signal Info, Status Badge, and Global Actions.
    """
    open_signal_requested = Signal()
    save_session_requested = Signal()
    export_requested = Signal()
    run_analysis_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("TopBarWidget")
        
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(16, 8, 16, 8)
        
        # Left Title section
        title_box = QVBoxLayout()
        title_box.setSpacing(2)
        
        self.title_label = QLabel("Dashboard")
        self.title_label.setObjectName("PageTitleLabel")
        
        self.desc_label = QLabel("Multi-domain digital signal analysis and diagnostics")
        self.desc_label.setObjectName("PageDescLabel")
        
        title_box.addWidget(self.title_label)
        title_box.addWidget(self.desc_label)
        
        main_layout.addLayout(title_box)
        main_layout.addStretch()

        # Center/Right Signal Status badge
        self.signal_badge = QLabel("Signal: None Loaded")
        self.signal_badge.setObjectName("SignalNameBadge")
        
        self.status_badge = StatusBadge("Ready")
        
        main_layout.addWidget(self.signal_badge)
        main_layout.addWidget(self.status_badge)
        main_layout.addSpacing(16)

        # Global Action Buttons
        self.open_btn = QPushButton("Open Signal")
        self.open_btn.clicked.connect(self.open_signal_requested.emit)
        
        self.run_btn = QPushButton("Run Analysis")
        self.run_btn.setObjectName("PrimaryButton")
        self.run_btn.clicked.connect(self.run_analysis_requested.emit)
        
        self.export_btn = QPushButton("Export")
        self.export_btn.clicked.connect(self.export_requested.emit)
        
        main_layout.addWidget(self.open_btn)
        main_layout.addWidget(self.run_btn)
        main_layout.addWidget(self.export_btn)

        # Connect session updates
        session_manager.session_updated.connect(self.update_signal_info)

    def set_page_info(self, title: str, description: str):
        self.title_label.setText(title)
        self.desc_label.setText(description)

    def update_signal_info(self):
        raw = session_manager.session.raw
        if raw is not None and len(raw.signal) > 0:
            self.signal_badge.setText(f"Signal: {raw.filename} [{raw.sampling_rate:.0f} Hz | {raw.duration:.2f}s | {raw.num_samples} samples]")
            has_results = bool(session_manager.session.spectral or session_manager.session.noise)
            self.status_badge.set_status("Complete" if has_results else "Ready")
        else:
            self.signal_badge.setText("Signal: None Loaded")
            self.status_badge.set_status("Ready")
