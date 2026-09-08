"""
Sidebar Widget for Spandhan Navigation.
"""

from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame
from PySide6.QtCore import Qt, Signal
from app.widgets.status_badge import StatusBadge
from core.session import session_manager
from core.config import VERSION


class Sidebar(QWidget):
    """
    Persistent Left Navigation Sidebar with Logo, Page Items, and System/Model Status.
    """
    page_changed = Signal(str)

    NAV_ITEMS = [
        ("Dashboard", "📊"),
        ("Input", "📥"),
        ("Preprocessing", "⚡"),
        ("DSP Analysis", "📈"),
        ("High-Resolution", "🎯"),
        ("Noise Analysis", "🔍"),
        ("Features", "🧮"),
        ("ML Studio", "🧠"),
        ("Domain Analysis", "⚙️"),
        ("Visualization", "👁️"),
        ("Reports", "📄"),
        ("Settings", "🛠️"),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("SidebarWidget")
        
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(12, 16, 12, 16)
        main_layout.setSpacing(4)
        
        # Logo section
        logo_label = QLabel("SPANDHAN")
        logo_label.setObjectName("SidebarLogoLabel")
        logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        subtitle_label = QLabel("DSP & DIAGNOSTICS PLATFORM")
        subtitle_label.setObjectName("SidebarSubtitleLabel")
        subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        main_layout.addWidget(logo_label)
        main_layout.addWidget(subtitle_label)
        
        # Divider
        divider = QFrame()
        divider.setStyleSheet("background-color: #1E293B; max-height: 1px; margin: 12px 0px;")
        main_layout.addWidget(divider)
        
        # Navigation buttons
        self.buttons = {}
        for name, icon in self.NAV_ITEMS:
            btn = QPushButton(f"  {icon}  {name}")
            btn.setObjectName("NavButton")
            btn.setProperty("active", "false")
            btn.clicked.connect(lambda checked=False, n=name: self.on_nav_click(n))
            main_layout.addWidget(btn)
            self.buttons[name] = btn

        main_layout.addStretch()

        # Bottom Status Section
        bottom_box = QFrame()
        bottom_box.setStyleSheet("background-color: #1E293B; border-radius: 6px; padding: 10px;")
        bottom_layout = QVBoxLayout(bottom_box)
        bottom_layout.setContentsMargins(8, 8, 8, 8)
        bottom_layout.setSpacing(6)
        
        self.model_status_badge = StatusBadge("Loaded" if session_manager.is_ml_model_loaded else "Not Loaded")
        
        sys_status = QLabel("System: Ready")
        sys_status.setStyleSheet("font-size: 10px; color: #94A3B8;")
        
        version_label = QLabel(f"Version {VERSION}")
        version_label.setStyleSheet("font-size: 10px; color: #64748B;")
        version_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        
        bottom_layout.addWidget(self.model_status_badge)
        
        ver_row = QHBoxLayout()
        ver_row.addWidget(sys_status)
        ver_row.addStretch()
        ver_row.addWidget(version_label)
        bottom_layout.addLayout(ver_row)
        
        main_layout.addWidget(bottom_box)

        # Default active page
        self.set_active_page("Dashboard")

    def on_nav_click(self, page_name: str):
        self.set_active_page(page_name)
        self.page_changed.emit(page_name)

    def set_active_page(self, page_name: str):
        for name, btn in self.buttons.items():
            is_active = (name == page_name)
            btn.setProperty("active", "true" if is_active else "false")
            btn.style().unpolish(btn)
            btn.style().polish(btn)

    def update_model_status(self):
        session_manager.check_ml_model_status()
        status_str = "Loaded" if session_manager.is_ml_model_loaded else "Not Loaded"
        self.model_status_badge.set_status(status_str)
