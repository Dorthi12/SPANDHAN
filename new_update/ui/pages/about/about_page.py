"""About Page Module."""
from ui.pages.common.placeholder_page import PlaceholderPage

class AboutPage(PlaceholderPage):
    def __init__(self, parent=None):
        super().__init__(
            breadcrumb="SPANDHAN  /  SYSTEM  /  ABOUT",
            title="About SPANDHAN Signal Intelligence",
            subtitle="Signal Processing & Artificial Neural Detection / Harmonized Analysis Network",
            description="Version 2.0 Engineering Workstation combining MATLAB Digital Signal Processing with PyTorch Deep Neural Networks.",
            icon_symbol="ⓘ",
            parent=parent,
        )
