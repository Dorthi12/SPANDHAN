"""Preprocessing Page Module."""
from ui.pages.common.placeholder_page import PlaceholderPage

class PreprocessingPage(PlaceholderPage):
    def __init__(self, parent=None):
        super().__init__(
            breadcrumb="SPANDHAN  /  ANALYSIS  /  PREPROCESSING",
            title="Signal Preprocessing Pipeline",
            subtitle="MATLAB conditioning, resampling, and normalization contracts",
            description="Coordinates pre-inference conditioning for 1D audio time-series and 2D spatial matrices.",
            icon_symbol="◫",
            parent=parent,
        )
