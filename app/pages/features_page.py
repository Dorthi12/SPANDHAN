"""
Feature Extraction Page module.
Extracts and presents the engineered 20-feature vector for time-domain, spectral, and noise/event characteristics.
"""

import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout, QFrame, QLabel, QScrollArea
from PySide6.QtCore import Qt

from app.widgets.section_header import SectionHeader
from app.widgets.results_table import ResultsTableWidget
from core.session import session_manager
from intelligence.noise.features import extract_noise_features, NOISE_FEATURE_NAMES


class FeaturesPage(QWidget):
    """
    Feature Extraction Vector View.
    """
    FEATURE_METADATA = {
        # Time Domain
        "rms": ("RMS", "", "TIME DOMAIN", "Root Mean Square amplitude"),
        "variance": ("Variance", "", "TIME DOMAIN", "Signal statistical variance"),
        "std": ("Standard Deviation", "", "TIME DOMAIN", "Signal standard deviation"),
        "kurtosis": ("Kurtosis", "", "TIME DOMAIN", "Measure of tailedness / impulsiveness"),
        "skewness": ("Skewness", "", "TIME DOMAIN", "Asymmetry of amplitude distribution"),
        "crest_factor": ("Crest Factor", "", "TIME DOMAIN", "Ratio of peak to RMS amplitude"),
        "zero_crossing_rate": ("Zero Crossing Rate", "crossings/sample", "TIME DOMAIN", "Rate of sign changes"),
        # Spectral
        "spectral_centroid": ("Spectral Centroid", "Hz", "SPECTRAL", "Center of mass of power spectrum"),
        "spectral_flatness": ("Spectral Flatness", "", "SPECTRAL", "Wiener entropy measure of noise-likeness"),
        "spectral_entropy": ("Spectral Entropy", "", "SPECTRAL", "Spectral power distribution entropy"),
        "spectral_rolloff": ("Spectral Rolloff", "Hz", "SPECTRAL", "Frequency below which 85% energy lies"),
        "spectral_variance": ("Spectral Variance", "", "SPECTRAL", "Variance of PSD values"),
        "low_band_energy": ("Low-band Energy", "J", "SPECTRAL", "PSD energy below 10% Nyquist"),
        "mid_band_energy": ("Mid-band Energy", "J", "SPECTRAL", "PSD energy 10–50% Nyquist"),
        "high_band_energy": ("High-band Energy", "J", "SPECTRAL", "PSD energy above 50% Nyquist"),
        "mains_band_energy": ("Mains-band Energy", "J", "SPECTRAL", "PSD energy around 50/60 Hz mains"),
        # Noise / Event
        "snr_db": ("Signal-to-Noise Ratio (SNR)", "dB", "NOISE / EVENT", "Estimated ratio of signal to noise power"),
        "impulse_count": ("Impulse Count", "events", "NOISE / EVENT", "Grouped sample exceedances > 3σ"),
        "peak_count_rate": ("Peak Count Rate", "peaks/sample", "NOISE / EVENT", "Local peaks exceeding 2σ"),
        "energy_ratio_first_half": ("Energy Ratio (First Half)", "", "NOISE / EVENT", "Ratio of first half energy to second half"),
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(14)
        
        # Header
        self.header = SectionHeader(
            "Feature Extraction Vector",
            "Structured 20-feature diagnostic vector combining time-domain, spectral shape, and noise metrics."
        )
        main_layout.addWidget(self.header)

        # Table Card
        table_card = QFrame()
        table_card.setObjectName("CardPanel")
        tbl_layout = QVBoxLayout(table_card)
        tbl_layout.setContentsMargins(14, 14, 14, 14)
        
        self.results_table = ResultsTableWidget(["Category", "Feature Name", "Value", "Unit", "Description"])
        tbl_layout.addWidget(self.results_table)
        
        main_layout.addWidget(table_card)

        # Session connection
        session_manager.session_updated.connect(self.extract_features)
        self.extract_features()

    def extract_features(self):
        sig = session_manager.session.active_signal
        fs = session_manager.session.active_sampling_rate
        if sig is None or len(sig) == 0:
            self.results_table.set_data([])
            return

        try:
            feat = extract_noise_features(sig, fs)
            session_manager.session.features = feat
            
            rows = []
            for fname in NOISE_FEATURE_NAMES:
                val = feat.get(fname, 0.0)
                meta = self.FEATURE_METADATA.get(fname, (fname, "", "OTHER", ""))
                
                val_str = f"{val:.6e}" if abs(val) < 0.001 or abs(val) > 10000 else f"{val:.4f}"
                rows.append([meta[2], meta[0], val_str, meta[1], meta[3]])
                
            self.results_table.set_data(rows)
        except Exception as e:
            print("Feature extraction error:", e)
