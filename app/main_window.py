"""
Spandhan Main Application Window module.
Integrates persistent Sidebar, TopBar, and QStackedWidget for all 11 pages.
"""

from PySide6.QtWidgets import QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QStackedWidget, QFileDialog
from PySide6.QtCore import Qt, QThreadPool

from app.styles import DARK_THEME_QSS
from app.widgets.sidebar import Sidebar
from app.widgets.topbar import TopBar

from app.pages.dashboard import DashboardPage
from app.pages.input_page import InputPage
from app.pages.preprocessing_page import PreprocessingPage
from app.pages.dsp_page import DspPage
from app.pages.high_res_page import HighResPage
from app.pages.noise_page import NoisePage
from app.pages.features_page import FeaturesPage
from app.pages.domain_page import DomainPage
from app.pages.visualization_page import VisualizationPage
from app.pages.report_page import ReportPage
from app.pages.settings_page import SettingsPage
from app.pages.ml_page import MLPage

from core.session import session_manager
from core.worker import Worker


class MainWindow(QMainWindow):
    """
    Main Window Shell for Spandhan Desktop Platform.
    """
    PAGE_INFO = {
        "Dashboard": ("Signal Analysis Dashboard", "Multi-domain digital signal analysis and diagnostics overview"),
        "Input": ("Signal Input Workspace", "Import digital signals from files or generate precision test signals"),
        "Preprocessing": ("Signal Preprocessing", "Condition signal using zero-phase filtering, detrending, and DC removal"),
        "DSP Analysis": ("DSP Analysis Workspace", "Spectral decomposition, time-frequency distributions, and wavelets"),
        "High-Resolution": ("High-Resolution Spectral Analysis", "Sub-resolution frequency estimation using MUSIC and ESPRIT"),
        "Noise Analysis": ("Noise Assessment & Diagnostics", "Evaluate SNR, noise statistics, and ML-based classification"),
        "Features": ("Feature Extraction Vector", "Structured 20-feature diagnostic vector"),
        "ML Studio": ("ML Studio", "Generate dataset · Train noise classifier · Upload signal · Get all-class predictions"),
        "Domain Analysis": ("Domain Analysis Workspace", "Specialized analysis for Audio, ECG R-peaks, and MCSA motor sidebands"),
        "Visualization": ("Visualization Workspace", "High-resolution interactive plotting workspace"),
        "Reports": ("Analysis Report Generator", "Generate, preview, and export comprehensive DSP & diagnostic reports"),
        "Settings": ("Application Settings", "Configure visual themes, analysis default parameters, and ML models"),
    }

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Spandhan - Multi-Domain Digital Signal Analysis & Diagnostics Platform")
        self.resize(1400, 900)
        self.setMinimumSize(1100, 700)
        
        # Apply dark theme
        self.setStyleSheet(DARK_THEME_QSS)

        # Main Central Widget Layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        root_layout = QHBoxLayout(central_widget)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # 1. Sidebar Navigation
        self.sidebar = Sidebar()
        self.sidebar.page_changed.connect(self.switch_page)
        root_layout.addWidget(self.sidebar)

        # Right Content Column (TopBar + Stacked Pages)
        right_column = QWidget()
        right_layout = QVBoxLayout(right_column)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        # 2. TopBar
        self.topbar = TopBar()
        self.topbar.open_signal_requested.connect(self.open_signal_dialog)
        self.topbar.run_analysis_requested.connect(self.run_full_pipeline)
        self.topbar.export_requested.connect(lambda: self.switch_page("Reports"))
        right_layout.addWidget(self.topbar)

        # 3. Stacked Pages Widget
        self.stacked_widget = QStackedWidget()
        
        self.pages = {
            "Dashboard": DashboardPage(),
            "Input": InputPage(),
            "Preprocessing": PreprocessingPage(),
            "DSP Analysis": DspPage(),
            "High-Resolution": HighResPage(),
            "Noise Analysis": NoisePage(),
            "Features": FeaturesPage(),
            "ML Studio": MLPage(),
            "Domain Analysis": DomainPage(),
            "Visualization": VisualizationPage(),
            "Reports": ReportPage(),
            "Settings": SettingsPage(),
        }

        # Connect Dashboard card navigation requests
        self.pages["Dashboard"].navigate_requested.connect(self.switch_page)

        for name, page_widget in self.pages.items():
            self.stacked_widget.addWidget(page_widget)

        right_layout.addWidget(self.stacked_widget)
        root_layout.addWidget(right_column)

        # Threadpool for background tasks
        self.threadpool = QThreadPool()

        # Initial Page
        self.switch_page("Dashboard")

    def switch_page(self, page_name: str):
        if page_name in self.pages:
            self.stacked_widget.setCurrentWidget(self.pages[page_name])
            self.sidebar.set_active_page(page_name)
            title, desc = self.PAGE_INFO.get(page_name, (page_name, ""))
            self.topbar.set_page_info(title, desc)

    def open_signal_dialog(self):
        self.switch_page("Input")

    def run_full_pipeline(self):
        """Execute complete analysis pipeline across DSP, Noise, High-Res, and Features."""
        sig = session_manager.session.active_signal
        if sig is None or len(sig) == 0:
            return
            
        def _pipeline_task():
            # Force trigger calculations across pages
            self.pages["DSP Analysis"].run_active_dsp()
            self.pages["High-Resolution"].run_high_res_analysis()
            self.pages["Noise Analysis"].run_noise_analysis()
            self.pages["Features"].extract_features()
            self.pages["Domain Analysis"].run_active_domain()

        worker = Worker(_pipeline_task)
        self.threadpool.start(worker)
