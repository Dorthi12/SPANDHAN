"""
Domain Analysis Page module.
Provides specialized analysis for GENERAL, AUDIO, ECG (QRS R-peak detection), and MCSA (Motor Current Signature Analysis).
"""

import numpy as np
from scipy.signal import find_peaks
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QFrame, QLabel, QPushButton, QComboBox, QDoubleSpinBox, QSpinBox, QFormLayout, QTabWidget, QScrollArea
from PySide6.QtCore import Qt

from app.widgets.section_header import SectionHeader
from app.widgets.signal_plot import SignalPlotWidget
from app.widgets.metric_card import MetricCard
from app.widgets.results_table import ResultsTableWidget
from app.widgets.status_badge import StatusBadge
from core.session import session_manager
from dsp.fft import compute_fft
from dsp.music import estimate_music_frequencies
from app.pages.audio_experiment_page import AudioExperimentPage
from app.pages.image_experiment_page import ImageExperimentPage


class DomainPage(QWidget):
    """
    Domain-Specific Diagnostics View.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(12)
        
        # Header
        self.header = SectionHeader(
            "Domain Analysis Workspace",
            "Domain-specific diagnostic instruments for Audio, ECG R-peak tracking, and Industrial Motor Current Signature Analysis (MCSA)."
        )
        main_layout.addWidget(self.header)

        # Tab Widget
        self.tabs = QTabWidget()
        
        # 1. GENERAL
        self.general_tab = QWidget()
        self._setup_general_tab()
        self.tabs.addTab(self.general_tab, "GENERAL")

        # 2. AUDIO — full experiment pipeline
        self.audio_tab = AudioExperimentPage()
        self.tabs.addTab(self.audio_tab, "AUDIO")

        # 3. IMAGE — full image experiment pipeline
        self.image_tab = ImageExperimentPage()
        self.tabs.addTab(self.image_tab, "IMAGE")

        # 4. ECG
        self.ecg_tab = QWidget()
        self._setup_ecg_tab()
        self.tabs.addTab(self.ecg_tab, "ECG")

        # 4. MCSA (Flagship)
        self.mcsa_tab = QWidget()
        self._setup_mcsa_tab()
        self.tabs.addTab(self.mcsa_tab, "MCSA (MOTOR)")

        main_layout.addWidget(self.tabs)

        # Session connection
        session_manager.session_updated.connect(self.run_active_domain)
        self.tabs.currentChanged.connect(self.run_active_domain)
        self.run_active_domain()

    # --- 1. GENERAL DOMAIN ---
    def _setup_general_tab(self):
        layout = QHBoxLayout(self.general_tab)
        self.gen_plot = SignalPlotWidget("General Signal Characterization")
        layout.addWidget(self.gen_plot)

    def _run_general(self):
        sig = session_manager.session.active_signal
        fs = session_manager.session.active_sampling_rate
        if sig is None or len(sig) == 0:
            self.gen_plot.plot_time_domain(None, None)
            return
        t = np.arange(len(sig)) / fs
        self.gen_plot.plot_time_domain(t, sig, "General Time-Domain Waveform")

    # --- 2. AUDIO DOMAIN --- (full experiment pipeline; logic in AudioExperimentPage)
    def _run_audio(self):
        self.audio_tab.run_active_domain()

    # --- 3. ECG DOMAIN ---
    def _setup_ecg_tab(self):
        layout = QHBoxLayout(self.ecg_tab)
        
        ctrl_card = QFrame()
        ctrl_card.setObjectName("CardPanel")
        ctrl_layout = QVBoxLayout(ctrl_card)
        
        ctrl_title = QLabel("QRS DETECTION INSTRUMENT")
        ctrl_title.setStyleSheet("font-size: 12px; font-weight: 700; color: #38BDF8;")
        ctrl_layout.addWidget(ctrl_title)
        
        self.ecg_hr_card = MetricCard("Heart Rate", "N/A", "BPM", "R-R Interval Derived", "#10B981")
        self.ecg_beats_card = MetricCard("Detected Beats", "N/A", "peaks", "R-Peak Count", "#38BDF8")
        self.ecg_rr_card = MetricCard("Mean RR Interval", "N/A", "ms", "P-QRS-T Metric", "#F59E0B")
        
        ctrl_layout.addWidget(self.ecg_hr_card)
        ctrl_layout.addWidget(self.ecg_beats_card)
        ctrl_layout.addWidget(self.ecg_rr_card)
        
        # Non-medical disclaimer
        disclaimer = QLabel("<b>NOTE:</b> Engineering signal processing analysis only. Not intended for clinical or medical diagnostic use.")
        disclaimer.setStyleSheet("font-size: 10px; color: #F59E0B; background-color: #78350F40; padding: 8px; border-radius: 4px;")
        disclaimer.setWordWrap(True)
        ctrl_layout.addWidget(disclaimer)
        ctrl_layout.addStretch()
        
        layout.addWidget(ctrl_card, stretch=1)
        
        self.ecg_plot = SignalPlotWidget("ECG QRS Peak Markers")
        layout.addWidget(self.ecg_plot, stretch=3)

    def _run_ecg(self):
        sig = session_manager.session.active_signal
        fs = session_manager.session.active_sampling_rate
        if sig is None or len(sig) == 0:
            self.ecg_plot.plot_time_domain(None, None)
            return
            
        t = np.arange(len(sig)) / fs
        # Simple QRS R-peak detection via Pan-Tompkins style peak search
        distance = int(0.4 * fs) # minimum 400ms between beats (150 bpm max)
        peaks, _ = find_peaks(sig, height=np.mean(sig) + 1.2 * np.std(sig), distance=distance)
        
        beat_count = len(peaks)
        if beat_count > 1:
            rr_intervals = np.diff(peaks) / fs # seconds
            mean_rr_ms = np.mean(rr_intervals) * 1000.0
            bpm = 60.0 / np.mean(rr_intervals)
        else:
            mean_rr_ms = 0.0
            bpm = 0.0
            
        self.ecg_hr_card.update_metric(f"{bpm:.1f}", "BPM")
        self.ecg_beats_card.update_metric(f"{beat_count}", "beats")
        self.ecg_rr_card.update_metric(f"{mean_rr_ms:.1f}", "ms")
        
        self.ecg_plot.plot_time_domain(t, sig, f"ECG Waveform with Detected R-Peaks ({beat_count} beats)")

    # --- 4. MCSA DOMAIN (FLAGSHIP) ---
    def _setup_mcsa_tab(self):
        layout = QHBoxLayout(self.mcsa_tab)
        
        # Left Motor Parameters Card
        left_box = QVBoxLayout()
        left_box.setSpacing(14)
        
        param_card = QFrame()
        param_card.setObjectName("CardPanel")
        param_layout = QVBoxLayout(param_card)
        
        param_title = QLabel("MOTOR PARAMETERS & SLIP")
        param_title.setStyleSheet("font-size: 13px; font-weight: 700; color: #38BDF8;")
        param_layout.addWidget(param_title)
        
        form = QFormLayout()
        form.setSpacing(8)
        
        self.fs_spin = QDoubleSpinBox()
        self.fs_spin.setRange(10.0, 400.0)
        self.fs_spin.setValue(50.0)
        self.fs_spin.setSuffix(" Hz")
        form.addRow("Mains Freq (f_s):", self.fs_spin)
        
        self.rpm_spin = QDoubleSpinBox()
        self.rpm_spin.setRange(100.0, 10000.0)
        self.rpm_spin.setValue(1470.0)
        self.rpm_spin.setSuffix(" RPM")
        form.addRow("Rotor Speed (N_r):", self.rpm_spin)
        
        self.poles_spin = QSpinBox()
        self.poles_spin.setRange(2, 12)
        self.poles_spin.setValue(4)
        self.poles_spin.setSingleStep(2)
        form.addRow("Number of Poles (p):", self.poles_spin)
        
        param_layout.addLayout(form)
        
        btn = QPushButton("Analyze Motor Sidebands")
        btn.setObjectName("PrimaryButton")
        btn.clicked.connect(self._run_mcsa)
        param_layout.addWidget(btn)
        
        left_box.addWidget(param_card)

        # MCSA Severity Status Card
        self.severity_card = QFrame()
        self.severity_card.setObjectName("CardPanel")
        sev_layout = QVBoxLayout(self.severity_card)
        
        sev_title = QLabel("MOTOR HEALTH SEVERITY")
        sev_title.setStyleSheet("font-size: 11px; font-weight: 700; color: #94A3B8;")
        self.severity_badge = StatusBadge("Normal")
        
        sev_layout.addWidget(sev_title)
        sev_layout.addWidget(self.severity_badge)
        
        left_box.addWidget(self.severity_card)
        left_box.addStretch()
        
        layout.addLayout(left_box, stretch=1)

        # Right Spectrum Plot & Sidebands Table
        right_box = QVBoxLayout()
        right_box.setSpacing(14)
        
        self.mcsa_plot_card = QFrame()
        self.mcsa_plot_card.setObjectName("CardPanel")
        mcsa_plot_layout = QVBoxLayout(self.mcsa_plot_card)
        
        self.mcsa_plot = SignalPlotWidget("Motor Current Signature Spectrum (MCSA)")
        mcsa_plot_layout.addWidget(self.mcsa_plot)
        
        right_box.addWidget(self.mcsa_plot_card, stretch=2)

        # Sideband Table
        self.sideband_card = QFrame()
        self.sideband_card.setObjectName("CardPanel")
        sb_layout = QVBoxLayout(self.sideband_card)
        
        sb_title = QLabel("SIDEBAND FAULT EIGENVALUES & SEVERITY")
        sb_title.setStyleSheet("font-size: 12px; font-weight: 700; color: #38BDF8; margin-bottom: 4px;")
        sb_layout.addWidget(sb_title)
        
        self.sideband_table = ResultsTableWidget(["Sideband Order", "Expected Freq (Hz)", "Estimated Peak (Hz)", "Error (Hz)", "Amplitude (dB)", "Severity"])
        sb_layout.addWidget(self.sideband_table)
        
        right_box.addWidget(self.sideband_card, stretch=1)
        
        layout.addLayout(right_box, stretch=3)

    def _run_mcsa(self):
        sig = session_manager.session.active_signal
        fs = session_manager.session.active_sampling_rate
        if sig is None or len(sig) == 0:
            self.mcsa_plot.plot_mcsa_spectrum(None, None, 0.0, [])
            self.sideband_table.set_data([])
            return

        f_s = self.fs_spin.value()
        rpm = self.rpm_spin.value()
        poles = self.poles_spin.value()
        
        # Synchronous speed N_s = 120 * f_s / poles
        N_s = (120.0 * f_s) / poles
        slip = max(0.001, (N_s - rpm) / N_s)
        
        # Expected sidebands f_sb = f_s * (1 +- 2*k*s) for k = 1, 2
        sidebands = [f_s * (1.0 - 2.0 * slip), f_s * (1.0 + 2.0 * slip)]
        
        freqs, mag = compute_fft(sig, fs)
        
        self.mcsa_plot.plot_mcsa_spectrum(freqs, mag, f_s, sidebands)
        
        # Sideband table
        db_mag = 20 * np.log10(np.maximum(mag, 1e-12))
        rows = []
        max_ratio = -100.0
        
        for k, sb_f in enumerate(sidebands):
            # Find nearest peak in spectrum
            idx = np.argmin(np.abs(freqs - sb_f))
            est_f = freqs[idx]
            amp_db = db_mag[idx]
            err = abs(est_f - sb_f)
            
            # Severity logic
            if amp_db > -30:
                sev = "Severe"
            elif amp_db > -50:
                sev = "Moderate"
            else:
                sev = "Normal"
                
            order_name = f"Lower Sideband (k=1)" if k == 0 else "Upper Sideband (k=1)"
            rows.append([order_name, f"{sb_f:.2f}", f"{est_f:.2f}", f"{err:.3f}", f"{amp_db:.1f}", sev])
            
        self.sideband_table.set_data(rows)
        
        if any(r[5] == "Severe" for r in rows):
            self.severity_badge.set_status("Error")
        elif any(r[5] == "Moderate" for r in rows):
            self.severity_badge.set_status("Warning")
        else:
            self.severity_badge.set_status("Complete")

    def run_active_domain(self):
        idx = self.tabs.currentIndex()
        if idx == 0:
            self._run_general()
        elif idx == 1:
            self._run_audio()
        elif idx == 2:
            self._run_ecg()
        elif idx == 3:
            self._run_mcsa()
