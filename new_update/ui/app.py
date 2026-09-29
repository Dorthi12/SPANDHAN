"""
SPANDHAN — Application Shell (MainWindow)
=========================================
Central workstation window integrating:
  - Persistent scientific left sidebar (Logo, Sections: ANALYSIS, RESULTS, SYSTEM)
  - Stacked page router hosting Image ML and modular future tabs
  - Bottom system status bar (Python ML, PyTorch CNN, MATLAB DSP indicators)
"""

import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from typing import Optional
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QStackedWidget,
    QApplication,
)

from ui.styles.theme import COLORS, MAIN_STYLESHEET
from ui.components.sidebar.sidebar import Sidebar
from ui.components.common.status_bar import SpandhanStatusBar
from ui.pages.image_ml.image_ml_page import ImageMLPage
from ui.pages.audio_ml.audio_ml_page import AudioMLPage
from ui.pages.preprocessing.preprocessing_page import PreprocessingPage
from ui.pages.audio_dsp.audio_dsp_page import AudioDSPPage
from ui.pages.image_dsp.image_dsp_page import ImageDSPPage
from ui.pages.results.results_page import ResultsPage
from ui.pages.about.about_page import AboutPage


class SpandhanMainWindow(QMainWindow):
    """Primary desktop workstation window for SPANDHAN."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self.setWindowTitle("SPANDHAN — Signal Intelligence Workstation")
        self.resize(1280, 860)
        self.setMinimumSize(1024, 700)

        # Apply global stylesheet
        self.setStyleSheet(MAIN_STYLESHEET)

        self._init_shell()

    def _init_shell(self):
        central_widget = QWidget(self)
        central_widget.setObjectName("CentralWidget")
        self.setCentralWidget(central_widget)

        master_layout = QVBoxLayout(central_widget)
        master_layout.setContentsMargins(0, 0, 0, 0)
        master_layout.setSpacing(0)

        # Main horizontal split: Sidebar + Content
        body_layout = QHBoxLayout()
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(0)

        # 1. Persistent Sidebar (Active: audio_ml)
        self.sidebar = Sidebar(active_page="audio_ml", parent=self)
        self.sidebar.page_changed.connect(self._navigate_to_page)
        body_layout.addWidget(self.sidebar)

        # 2. Page Router (Stacked Widget)
        self.page_stack = QStackedWidget(self)
        self._page_indices: dict[str, int] = {}

        # Instantiate pages
        self._register_page("preprocessing", PreprocessingPage(self))
        self._register_page("audio_ml", AudioMLPage(self))
        self._register_page("image_ml", ImageMLPage(self))
        self._register_page("audio_dsp", AudioDSPPage(self))
        self._register_page("image_dsp", ImageDSPPage(self))
        self._register_page("results", ResultsPage(self))
        self._register_page("about", AboutPage(self))

        body_layout.addWidget(self.page_stack, stretch=1)
        master_layout.addLayout(body_layout, stretch=1)

        # 3. Bottom Persistent Status Bar
        self.status_bar_widget = SpandhanStatusBar(self)
        master_layout.addWidget(self.status_bar_widget)

        # Set default active page to Audio ML
        self._navigate_to_page("audio_ml")

    def _register_page(self, key: str, page_widget: QWidget):
        idx = self.page_stack.addWidget(page_widget)
        self._page_indices[key] = idx
        if hasattr(page_widget, "navigate_requested"):
            page_widget.navigate_requested.connect(self._navigate_to_page)

    def _navigate_to_page(self, key: str):
        if key in self._page_indices:
            self.page_stack.setCurrentIndex(self._page_indices[key])
            if hasattr(self, "sidebar") and self.sidebar:
                self.sidebar.set_active_page(key)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SpandhanMainWindow()
    window.show()
    sys.exit(app.exec())
