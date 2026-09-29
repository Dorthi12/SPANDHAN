"""Results Page Module."""
from ui.pages.common.placeholder_page import PlaceholderPage

class ResultsPage(PlaceholderPage):
    def __init__(self, parent=None):
        super().__init__(
            breadcrumb="SPANDHAN  /  RESULTS  /  ANALYSIS RESULTS",
            title="Comprehensive Analysis Results",
            subtitle="Combined DSP and ML inference summary reports",
            description="Aggregates classified signal states, confidence distributions, and DSP filtering outputs.",
            icon_symbol="◷",
            parent=parent,
        )
