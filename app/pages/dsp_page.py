"""
DSP Analysis Page module.
Provides tabbed digital signal processing analysis: FFT, PSD, STFT, Correlation, Hilbert, Wavelet, Cepstrum, and Filters.
"""

import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QFrame, QLabel, QPushButton, QComboBox, QDoubleSpinBox, QSpinBox, QFormLayout, QTabWidget, QCheckBox
from PySide6.QtCore import Qt

from app.widgets.section_header import SectionHeader
from app.widgets.signal_plot import SignalPlotWidget
from app.widgets.metric_card import MetricCard
from core.session import session_manager
from dsp.fft import compute_fft
from dsp.psd import compute_psd
from dsp.stft import compute_stft
from dsp.correlation import compute_cross_correlation
from dsp.hilbert import compute_hilbert
from dsp.wavelet import discrete_wavelet_transform
from dsp.cepstrum import compute_real_cepstrum


class DspPage(QWidget):
    """
    DSP Analysis View with 8 Specialized Analysis Tabs.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(12)
        
        # Header
        self.header = SectionHeader("DSP Analysis Workspace", "Spectral decomposition, time-frequency distributions, analytic signal envelopes, and wavelets.")
        main_layout.addWidget(self.header)

        # Tab Widget
        self.tabs = QTabWidget()
        
        # Create Tab 1: FFT
        self.fft_tab = QWidget()
        self._setup_fft_tab()
        self.tabs.addTab(self.fft_tab, "FFT")

        # Create Tab 2: PSD
        self.psd_tab = QWidget()
        self._setup_psd_tab()
        self.tabs.addTab(self.psd_tab, "PSD")

        # Create Tab 3: STFT
        self.stft_tab = QWidget()
        self._setup_stft_tab()
        self.tabs.addTab(self.stft_tab, "STFT")

        # Create Tab 4: Correlation
        self.corr_tab = QWidget()
        self._setup_corr_tab()
        self.tabs.addTab(self.corr_tab, "Correlation")

        # Create Tab 5: Hilbert
        self.hilbert_tab = QWidget()
        self._setup_hilbert_tab()
        self.tabs.addTab(self.hilbert_tab, "Hilbert")

        # Create Tab 6: Wavelet
        self.wavelet_tab = QWidget()
        self._setup_wavelet_tab()
        self.tabs.addTab(self.wavelet_tab, "Wavelet")

        # Create Tab 7: Cepstrum
        self.cepstrum_tab = QWidget()
        self._setup_cepstrum_tab()
        self.tabs.addTab(self.cepstrum_tab, "Cepstrum")

        main_layout.addWidget(self.tabs)

        # Session connection
        session_manager.session_updated.connect(self.run_active_dsp)
        self.tabs.currentChanged.connect(self.run_active_dsp)
        self.run_active_dsp()

    # 1. FFT TAB
    def _setup_fft_tab(self):
        layout = QHBoxLayout(self.fft_tab)
        
        # Controls
        ctrl_card = QFrame()
        ctrl_card.setObjectName("CardPanel")
        ctrl_layout = QVBoxLayout(ctrl_card)
        
        form = QFormLayout()
        self.fft_window_cb = QComboBox()
        self.fft_window_cb.addItems(["hann", "hamming", "blackman", "rectangular"])
        form.addRow("Window:", self.fft_window_cb)
        
        self.fft_scale_cb = QComboBox()
        self.fft_scale_cb.addItems(["linear", "dB"])
        form.addRow("Scale:", self.fft_scale_cb)
        
        ctrl_layout.addLayout(form)
        
        self.fft_run_btn = QPushButton("Compute FFT")
        self.fft_run_btn.setObjectName("PrimaryButton")
        self.fft_run_btn.clicked.connect(self._run_fft)
        ctrl_layout.addWidget(self.fft_run_btn)
        
        # Metrics
        self.fft_dom_freq = MetricCard("Dominant Freq", "N/A", "Hz")
        self.fft_peak_amp = MetricCard("Peak Magnitude", "N/A", "")
        self.fft_energy = MetricCard("Spectral Energy", "N/A", "")
        
        ctrl_layout.addWidget(self.fft_dom_freq)
        ctrl_layout.addWidget(self.fft_peak_amp)
        ctrl_layout.addWidget(self.fft_energy)
        ctrl_layout.addStretch()
        
        layout.addWidget(ctrl_card, stretch=1)
        
        # Plot
        self.fft_plot = SignalPlotWidget("FFT Magnitude Spectrum")
        layout.addWidget(self.fft_plot, stretch=3)

    def _run_fft(self):
        sig = session_manager.session.active_signal
        fs = session_manager.session.active_sampling_rate
        if sig is None or len(sig) == 0:
            self.fft_plot.plot_spectrum(None, None)
            return
            
        freqs, mag = compute_fft(sig, fs)
        scale = self.fft_scale_cb.currentText()
        
        # Peak extraction
        peak_idx = np.argmax(mag)
        dom_f = freqs[peak_idx]
        peak_m = mag[peak_idx]
        energy = np.sum(mag**2)
        
        self.fft_dom_freq.update_metric(f"{dom_f:.1f}", "Hz")
        self.fft_peak_amp.update_metric(f"{peak_m:.4f}", "")
        self.fft_energy.update_metric(f"{energy:.2e}", "")
        
        peaks = {"freqs": [dom_f], "mags": [20 * np.log10(max(peak_m, 1e-12)) if scale == "dB" else peak_m]}
        self.fft_plot.plot_spectrum(freqs, mag, "FFT Magnitude Spectrum", log_scale=scale, peaks=peaks)
        
        # Save results to session
        session_manager.session.spectral["fft_freqs"] = freqs
        session_manager.session.spectral["fft_mag"] = mag
        session_manager.session.spectral["peak_freq"] = dom_f

    # 2. PSD TAB
    def _setup_psd_tab(self):
        layout = QHBoxLayout(self.psd_tab)
        
        ctrl_card = QFrame()
        ctrl_card.setObjectName("CardPanel")
        ctrl_layout = QVBoxLayout(ctrl_card)
        
        form = QFormLayout()
        self.psd_window_cb = QComboBox()
        self.psd_window_cb.addItems(["hann", "hamming", "blackman"])
        form.addRow("Window:", self.psd_window_cb)
        
        self.psd_nperseg_spin = QSpinBox()
        self.psd_nperseg_spin.setRange(32, 8192)
        self.psd_nperseg_spin.setValue(256)
        form.addRow("Segment Length:", self.psd_nperseg_spin)
        
        ctrl_layout.addLayout(form)
        
        btn = QPushButton("Compute PSD")
        btn.setObjectName("PrimaryButton")
        btn.clicked.connect(self._run_psd)
        ctrl_layout.addWidget(btn)
        
        self.psd_peak_card = MetricCard("Peak PSD", "N/A", "V²/Hz")
        self.psd_power_card = MetricCard("Band Power", "N/A", "V²")
        ctrl_layout.addWidget(self.psd_peak_card)
        ctrl_layout.addWidget(self.psd_power_card)
        ctrl_layout.addStretch()
        
        layout.addWidget(ctrl_card, stretch=1)
        
        self.psd_plot = SignalPlotWidget("Welch Power Spectral Density")
        layout.addWidget(self.psd_plot, stretch=3)

    def _run_psd(self):
        sig = session_manager.session.active_signal
        fs = session_manager.session.active_sampling_rate
        if sig is None or len(sig) == 0:
            self.psd_plot.plot_spectrum(None, None)
            return
            
        nperseg = min(self.psd_nperseg_spin.value(), len(sig))
        freqs, psd = compute_psd(sig, fs, window=self.psd_window_cb.currentText(), nperseg=nperseg)
        
        peak_p = np.max(psd)
        power = np.trapz(psd, freqs)
        self.psd_peak_card.update_metric(f"{peak_p:.2e}", "V²/Hz")
        self.psd_power_card.update_metric(f"{power:.2e}", "V²")
        
        self.psd_plot.plot_spectrum(freqs, psd, "Power Spectral Density (Welch)", log_scale="linear")
        session_manager.session.spectral["psd_freqs"] = freqs
        session_manager.session.spectral["psd"] = psd

    # 3. STFT TAB
    def _setup_stft_tab(self):
        layout = QHBoxLayout(self.stft_tab)
        
        ctrl_card = QFrame()
        ctrl_card.setObjectName("CardPanel")
        ctrl_layout = QVBoxLayout(ctrl_card)
        
        form = QFormLayout()
        self.stft_win_cb = QComboBox()
        self.stft_win_cb.addItems(["hann", "hamming", "blackman"])
        form.addRow("Window:", self.stft_win_cb)
        
        self.stft_nperseg = QSpinBox()
        self.stft_nperseg.setRange(32, 2048)
        self.stft_nperseg.setValue(256)
        form.addRow("NFFT Segment:", self.stft_nperseg)
        
        ctrl_layout.addLayout(form)
        
        btn = QPushButton("Compute STFT")
        btn.setObjectName("PrimaryButton")
        btn.clicked.connect(self._run_stft)
        ctrl_layout.addWidget(btn)
        ctrl_layout.addStretch()
        
        layout.addWidget(ctrl_card, stretch=1)
        
        self.stft_plot = SignalPlotWidget("Short-Time Fourier Transform (Spectrogram)")
        layout.addWidget(self.stft_plot, stretch=3)

    def _run_stft(self):
        sig = session_manager.session.active_signal
        fs = session_manager.session.active_sampling_rate
        if sig is None or len(sig) == 0:
            self.stft_plot.plot_spectrogram(None, None, None)
            return
            
        nperseg = min(self.stft_nperseg.value(), len(sig))
        f, t, Zxx = compute_stft(sig, fs, window=self.stft_win_cb.currentText(), nperseg=nperseg)
        self.stft_plot.plot_spectrogram(t, f, np.abs(Zxx))

    # 4. CORRELATION TAB
    def _setup_corr_tab(self):
        layout = QHBoxLayout(self.corr_tab)
        
        ctrl_card = QFrame()
        ctrl_card.setObjectName("CardPanel")
        ctrl_layout = QVBoxLayout(ctrl_card)
        
        btn = QPushButton("Compute Autocorrelation")
        btn.setObjectName("PrimaryButton")
        btn.clicked.connect(self._run_corr)
        ctrl_layout.addWidget(btn)
        
        self.corr_max_card = MetricCard("Max Lag Correlation", "N/A", "")
        ctrl_layout.addWidget(self.corr_max_card)
        ctrl_layout.addStretch()
        
        layout.addWidget(ctrl_card, stretch=1)
        
        self.corr_plot = SignalPlotWidget("Autocorrelation")
        layout.addWidget(self.corr_plot, stretch=3)

    def _run_corr(self):
        sig = session_manager.session.active_signal
        fs = session_manager.session.active_sampling_rate
        if sig is None or len(sig) == 0:
            self.corr_plot.plot_time_domain(None, None)
            return
            
        lags, rxx = compute_cross_correlation(sig, sig)
        t_lags = lags / fs
        self.corr_max_card.update_metric(f"{np.max(rxx):.4f}", "")
        self.corr_plot.plot_time_domain(t_lags, rxx, "Signal Autocorrelation", xlabel="Lag (s)", ylabel="Rxx")

    # 5. HILBERT TAB
    def _setup_hilbert_tab(self):
        layout = QHBoxLayout(self.hilbert_tab)
        
        ctrl_card = QFrame()
        ctrl_card.setObjectName("CardPanel")
        ctrl_layout = QVBoxLayout(ctrl_card)
        
        btn = QPushButton("Compute Hilbert Transform")
        btn.setObjectName("PrimaryButton")
        btn.clicked.connect(self._run_hilbert)
        ctrl_layout.addWidget(btn)
        
        self.hilb_env_card = MetricCard("Peak Envelope", "N/A", "")
        ctrl_layout.addWidget(self.hilb_env_card)
        ctrl_layout.addStretch()
        
        layout.addWidget(ctrl_card, stretch=1)
        
        self.hilbert_plot = SignalPlotWidget("Hilbert Envelope & Analytic Signal")
        layout.addWidget(self.hilbert_plot, stretch=3)

    def _run_hilbert(self):
        sig = session_manager.session.active_signal
        fs = session_manager.session.active_sampling_rate
        if sig is None or len(sig) == 0:
            self.hilbert_plot.plot_time_domain(None, None)
            return
            
        h_res = compute_hilbert(sig, fs)
        env = h_res["envelope"]
        t = np.arange(len(sig)) / fs
        self.hilb_env_card.update_metric(f"{np.max(env):.4f}", "")
        self.hilbert_plot.plot_before_after(t, sig, env, "Original Signal vs Hilbert Envelope")

    # 6. WAVELET TAB
    def _setup_wavelet_tab(self):
        layout = QHBoxLayout(self.wavelet_tab)
        
        ctrl_card = QFrame()
        ctrl_card.setObjectName("CardPanel")
        ctrl_layout = QVBoxLayout(ctrl_card)
        
        form = QFormLayout()
        self.wav_family_cb = QComboBox()
        self.wav_family_cb.addItems(["db4", "db8", "sym4", "coif2", "haar"])
        form.addRow("Wavelet Family:", self.wav_family_cb)
        
        self.wav_level_spin = QSpinBox()
        self.wav_level_spin.setRange(1, 8)
        self.wav_level_spin.setValue(4)
        form.addRow("Level:", self.wav_level_spin)
        
        ctrl_layout.addLayout(form)
        
        btn = QPushButton("Compute Wavelet DWT")
        btn.setObjectName("PrimaryButton")
        btn.clicked.connect(self._run_wavelet)
        ctrl_layout.addWidget(btn)
        ctrl_layout.addStretch()
        
        layout.addWidget(ctrl_card, stretch=1)
        
        self.wavelet_plot = SignalPlotWidget("Wavelet Coefficients")
        layout.addWidget(self.wavelet_plot, stretch=3)

    def _run_wavelet(self):
        sig = session_manager.session.active_signal
        fs = session_manager.session.active_sampling_rate
        if sig is None or len(sig) == 0:
            self.wavelet_plot.plot_time_domain(None, None)
            return
            
        try:
            dwt = discrete_wavelet_transform(sig, wavelet=self.wav_family_cb.currentText())
            cA = dwt["approximation"]
            t = np.linspace(0, len(sig)/fs, len(cA))
            self.wavelet_plot.plot_time_domain(t, cA, "Approximation Coefficients (cA)", xlabel="Time (s)", ylabel="Amplitude")
        except Exception as e:
            print("Wavelet error:", e)

    # 7. CEPSTRUM TAB
    def _setup_cepstrum_tab(self):
        layout = QHBoxLayout(self.cepstrum_tab)
        
        ctrl_card = QFrame()
        ctrl_card.setObjectName("CardPanel")
        ctrl_layout = QVBoxLayout(ctrl_card)
        
        btn = QPushButton("Compute Real Cepstrum")
        btn.setObjectName("PrimaryButton")
        btn.clicked.connect(self._run_cepstrum)
        ctrl_layout.addWidget(btn)
        
        self.ceps_peak_card = MetricCard("Quefrency Peak", "N/A", "s")
        ctrl_layout.addWidget(self.ceps_peak_card)
        ctrl_layout.addStretch()
        
        layout.addWidget(ctrl_card, stretch=1)
        
        self.cepstrum_plot = SignalPlotWidget("Real Cepstrum")
        layout.addWidget(self.cepstrum_plot, stretch=3)

    def _run_cepstrum(self):
        sig = session_manager.session.active_signal
        fs = session_manager.session.active_sampling_rate
        if sig is None or len(sig) == 0:
            self.cepstrum_plot.plot_time_domain(None, None)
            return
            
        quef, ceps = compute_real_cepstrum(sig, fs)
        
        # Avoid quefrency 0
        peak_idx = np.argmax(ceps[1:]) + 1
        self.ceps_peak_card.update_metric(f"{quef[peak_idx]:.4f}", "s")
        self.cepstrum_plot.plot_time_domain(quef[:len(quef)//2], ceps[:len(ceps)//2], "Real Cepstrum Spectrum", xlabel="Quefrency (s)", ylabel="Amplitude")

    def run_active_dsp(self):
        idx = self.tabs.currentIndex()
        if idx == 0:
            self._run_fft()
        elif idx == 1:
            self._run_psd()
        elif idx == 2:
            self._run_stft()
        elif idx == 3:
            self._run_corr()
        elif idx == 4:
            self._run_hilbert()
        elif idx == 5:
            self._run_wavelet()
        elif idx == 6:
            self._run_cepstrum()
