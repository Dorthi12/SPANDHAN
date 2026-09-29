"""SPANDHAN Application Pages Package."""
from ui.pages.image_ml.image_ml_page import ImageMLPage
from ui.pages.audio_ml.audio_ml_page import AudioMLPage
from ui.pages.preprocessing.preprocessing_page import PreprocessingPage
from ui.pages.audio_dsp.audio_dsp_page import AudioDSPPage
from ui.pages.image_dsp.image_dsp_page import ImageDSPPage
from ui.pages.results.results_page import ResultsPage
from ui.pages.about.about_page import AboutPage

__all__ = [
    "ImageMLPage",
    "AudioMLPage",
    "PreprocessingPage",
    "AudioDSPPage",
    "ImageDSPPage",
    "ResultsPage",
    "AboutPage",
]
