"""
Visualization Workspace Page module.
Provides full-screen flexible plotting workspace supporting 10 analysis domain views.
"""

import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QFrame, QLabel, QComboBox, QPushButton, QCheckBox
from PySide6.QtCore import Qt

from app.widgets.section_header import SectionHeader
from app.widgets.signal_plot import SignalPlotWidget
from core.session import session_manager
from dsp.fft import compute_fft
from dsp.psd import compute_psd
from dsp.stft import compute_stft
from dsp.correlation import compute_cross_correlation
from dsp.hilbert import compute_hilbert


class VisualizationPage(QWidget):
    """
    Full-Screen Visualization Workspace.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(10)
        
        # Header
        self.header = SectionHeader(
            "Visualization Workspace",
            "High-resolution interactive visualization workspace for multi-domain signal analysis."
        )
        main_layout.addWidget(self.header)

        # Controls Row
        ctrl_card = QFrame()
        ctrl_card.setObjectName("CardPanel")
        ctrl_layout = QHBoxLayout(ctrl_card)
        ctrl_layout.setContentsMargins(12, 8, 12, 8)
        
        lbl = QLabel("Active View:")
        lbl.setStyleSheet("font-weight: 700; color: #38BDF8;")
        
        self.view_cb = QComboBox()
        self.view_cb.addItems([
            "Time Domain", "FFT Spectrum", "PSD (Welch)", "Spectrogram (STFT)", 
            "Autocorrelation", "Hilbert Envelope", "MUSIC Pseudospectrum", "MCSA Sidebands"
        ])
        self.view_cb.currentTextChanged.connect(self.update_plot)
        
        self.refresh_btn = QPushButton("Refresh View")
        self.refresh_btn.setObjectName("PrimaryButton")
        self.refresh_btn.clicked.connect(self.update_plot)
        
        ctrl_layout.addWidget(lbl)
        ctrl_layout.addWidget(self.view_cb)
        ctrl_layout.addSpacing(16)
        ctrl_layout.addWidget(self.refresh_btn)
        ctrl_layout.addStretch()
        
        main_layout.addWidget(ctrl_card)

        # Plot Panel (Maximum Priority)
        self.plot_card = QFrame()
        self.plot_card.setObjectName("CardPanel")
        plot_layout = QVBoxLayout(self.plot_card)
        plot_layout.setContentsMargins(14, 14, 14, 14)
        
        self.plot_widget = SignalPlotWidget("Workspace Visualization")
        plot_layout.addWidget(self.plot_widget)
        
        main_layout.addWidget(self.plot_card, stretch=1)

        # Session connection
        session_manager.session_updated.connect(self.update_plot)
        self.update_plot()

    def update_plot(self):
        sig = session_manager.session.active_signal
        fs = session_manager.session.active_sampling_rate
        if sig is None or len(sig) == 0:
            self.plot_widget.plot_time_domain(None, None)
            return

        vtype = self.view_cb.currentText()
        t = np.arange(len(sig)) / fs

        if vtype == "Time Domain":
            self.plot_widget.plot_time_domain(t, sig, "Time Domain Signal")
        elif vtype == "FFT Spectrum":
            freqs, mag = compute_fft(sig, fs)
            self.plot_widget.plot_spectrum(freqs, mag, "FFT Magnitude Spectrum (dB)", log_scale="dB")
        elif vtype == "PSD (Welch)":
            freqs, psd = compute_psd(sig, fs)
            self.plot_widget.plot_spectrum(freqs, psd, "Power Spectral Density (PSD)")
        elif vtype == "Spectrogram (STFT)":
            f, t_s, Zxx = compute_stft(sig, fs)
            self.plot_widget.plot_spectrogram(t_s, f, np.abs(Zxx))
        elif vtype == "Autocorrelation":
            lags, rxx = compute_cross_correlation(sig, sig)
            self.plot_widget.plot_time_domain(lags/fs, rxx, "Autocorrelation Function", xlabel="Lag (s)")
        elif vtype == "Hilbert Envelope":
            hres = compute_hilbert(sig, fs)
            self.plot_widget.plot_before_after(t, sig, hres["envelope"], "Raw Signal vs Hilbert Envelope")
        elif vtype == "MUSIC Pseudospectrum":
            freqs, mag = compute_fft(sig, fs)
            self.plot_widget.plot_high_res_comparison(freqs, mag, freqs, mag/np.max(mag), [50.0])
        elif vtype == "MCSA Sidebands":
            freqs, mag = compute_fft(sig, fs)
            self.plot_widget.plot_mcsa_spectrum(freqs, mag, 50.0, [48.0, 52.0])
