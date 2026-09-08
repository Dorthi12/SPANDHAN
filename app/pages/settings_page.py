"""
Settings Page module.
Provides application settings for general preferences, analysis defaults, visualization options, and ML model configuration.
"""

from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QFrame, QLabel, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox, QFormLayout, QScrollArea, QLineEdit, QCheckBox, QGroupBox
from PySide6.QtCore import Qt

from app.widgets.section_header import SectionHeader
from app.widgets.status_badge import StatusBadge
from core.session import session_manager
from core.config import DEFAULT_MODEL_PATH, VERSION


class SettingsPage(QWidget):
    """
    Application Settings View.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        
        container = QWidget()
        scroll.setWidget(container)
        
        main_layout = QVBoxLayout(container)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(16)
        
        # Header
        self.header = SectionHeader(
            "Application Settings",
            "Configure visual themes, analysis default parameters, and ML classification model paths."
        )
        main_layout.addWidget(self.header)

        # 1. General & Visual Preferences
        gen_card = QFrame()
        gen_card.setObjectName("CardPanel")
        gen_layout = QVBoxLayout(gen_card)
        
        gen_title = QLabel("GENERAL PREFERENCES")
        gen_title.setStyleSheet("font-size: 13px; font-weight: 700; color: #38BDF8;")
        gen_layout.addWidget(gen_title)
        
        form1 = QFormLayout()
        self.theme_cb = QComboBox()
        self.theme_cb.addItems(["Dark Engineering Workstation (Default)", "Light Scientific"])
        form1.addRow("Visual Theme:", self.theme_cb)
        
        self.domain_cb = QComboBox()
        self.domain_cb.addItems(["General", "Audio", "ECG", "MCSA"])
        form1.addRow("Default Domain:", self.domain_cb)
        
        self.autosave_cb = QCheckBox("Enable Session Autosave")
        self.autosave_cb.setChecked(True)
        form1.addRow("Autosave:", self.autosave_cb)
        
        gen_layout.addLayout(form1)
        main_layout.addWidget(gen_card)

        # 2. ML Classifier Model Management
        model_card = QFrame()
        model_card.setObjectName("CardPanel")
        model_layout = QVBoxLayout(model_card)
        
        model_header = QHBoxLayout()
        m_title = QLabel("ML NOISE CLASSIFICATION MODEL")
        m_title.setStyleSheet("font-size: 13px; font-weight: 700; color: #38BDF8;")
        self.model_status = StatusBadge("Loaded" if session_manager.is_ml_model_loaded else "Not Loaded")
        
        model_header.addWidget(m_title)
        model_header.addStretch()
        model_header.addWidget(self.model_status)
        model_layout.addLayout(model_header)
        
        form2 = QFormLayout()
        self.model_path_input = QLineEdit(DEFAULT_MODEL_PATH)
        form2.addRow("Model Path:", self.model_path_input)
        
        model_layout.addLayout(form2)
        
        reload_btn = QPushButton("Reload ML Model")
        reload_btn.setObjectName("PrimaryButton")
        reload_btn.clicked.connect(self.reload_model)
        model_layout.addWidget(reload_btn, alignment=Qt.AlignmentFlag.AlignLeft)
        
        main_layout.addWidget(model_card)

        # 3. Advanced Configuration (Expandable GroupBox)
        adv_group = QGroupBox("ADVANCED ANALYSIS DEFAULTS (EXPANDABLE)")
        adv_layout = QFormLayout(adv_group)
        adv_layout.setSpacing(10)
        
        self.default_nfft_spin = QSpinBox()
        self.default_nfft_spin.setRange(256, 16384)
        self.default_nfft_spin.setValue(4096)
        adv_layout.addRow("Default NFFT Size:", self.default_nfft_spin)
        
        self.default_order_spin = QSpinBox()
        self.default_order_spin.setRange(2, 100)
        self.default_order_spin.setValue(20)
        adv_layout.addRow("Default Subspace Order:", self.default_order_spin)
        
        self.max_threads_spin = QSpinBox()
        self.max_threads_spin.setRange(1, 32)
        self.max_threads_spin.setValue(4)
        adv_layout.addRow("Max Thread Pool Workers:", self.max_threads_spin)
        
        main_layout.addWidget(adv_group)

        # Save Button
        save_btn = QPushButton("Save Settings")
        save_btn.setObjectName("SuccessButton")
        main_layout.addWidget(save_btn, alignment=Qt.AlignmentFlag.AlignRight)
        
        main_layout.addStretch()

        # Root layout
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.addWidget(scroll)

    def reload_model(self):
        session_manager.check_ml_model_status()
        st = "Loaded" if session_manager.is_ml_model_loaded else "Not Loaded"
        self.model_status.set_status(st)
