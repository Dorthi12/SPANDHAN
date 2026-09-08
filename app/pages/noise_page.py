"""
Noise Analysis Page module.
Provides noise characterization, SNR estimation, before/after denoising comparison, and ML noise classification.
"""

import os
import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QFrame, QLabel, QPushButton, QScrollArea, QTableWidget, QTableWidgetItem, QHeaderView, QProgressBar
from PySide6.QtCore import Qt

from app.widgets.section_header import SectionHeader
from app.widgets.metric_card import MetricCard
from app.widgets.status_badge import StatusBadge
from app.widgets.signal_plot import SignalPlotWidget
from core.session import session_manager
from intelligence.noise.analyzer import analyze_noise
from intelligence.noise.features import extract_noise_features
from intelligence.noise.model_io import load_noise_model


class NoisePage(QWidget):
    """
    Noise Assessment & ML Noise Diagnostics View.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        
        container = QWidget()
        scroll.setWidget(container)
        
        main_layout = QVBoxLayout(container)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(16)
        
        # Header
        self.header = SectionHeader(
            "Noise Assessment & Diagnostics",
            "Evaluate signal-to-noise ratio (SNR), noise statistics, and ML-based noise type classification."
        )
        main_layout.addWidget(self.header)

        # 1. Top Section: Assessment Status Bar & Metrics
        status_card = QFrame()
        status_card.setObjectName("CardPanel")
        stat_layout = QHBoxLayout(status_card)
        stat_layout.setContentsMargins(16, 12, 16, 12)
        
        stat_lbl_box = QVBoxLayout()
        stat_lbl_box.setSpacing(2)
        title_lbl = QLabel("NOISE ASSESSMENT STATUS")
        title_lbl.setStyleSheet("font-size: 11px; font-weight: 700; color: #94A3B8; letter-spacing: 0.8px;")
        
        self.assessment_status = StatusBadge("Clean")
        stat_lbl_box.addWidget(title_lbl)
        stat_lbl_box.addWidget(self.assessment_status)
        stat_layout.addLayout(stat_lbl_box)
        stat_layout.addStretch()

        self.snr_card = MetricCard("Estimated SNR", "N/A", "dB", "Signal Quality", "#10B981")
        self.sig_rms_card = MetricCard("Signal RMS", "N/A", "", "Desired Signal", "#38BDF8")
        self.noise_rms_card = MetricCard("Noise RMS", "N/A", "", "Estimated Noise", "#EF4444")
        self.zcr_card = MetricCard("Zero Crossing Rate", "N/A", "", "ZCR Metric", "#F59E0B")
        
        stat_layout.addWidget(self.snr_card)
        stat_layout.addWidget(self.sig_rms_card)
        stat_layout.addWidget(self.noise_rms_card)
        stat_layout.addWidget(self.zcr_card)
        
        main_layout.addWidget(status_card)

        # 2. Middle Section: Plots & Characterization
        mid_layout = QHBoxLayout()
        mid_layout.setSpacing(16)
        
        # Noise PSD Plot
        self.plot_card = QFrame()
        self.plot_card.setObjectName("CardPanel")
        plot_layout = QVBoxLayout(self.plot_card)
        plot_layout.setContentsMargins(14, 14, 14, 14)
        
        self.plot_widget = SignalPlotWidget("Noise Power Spectral Density (PSD)")
        plot_layout.addWidget(self.plot_widget)
        
        mid_layout.addWidget(self.plot_card, stretch=2)

        # DSP Noise Statistics Panel
        self.stats_card = QFrame()
        self.stats_card.setObjectName("CardPanel")
        stats_layout = QVBoxLayout(self.stats_card)
        stats_layout.setContentsMargins(14, 14, 14, 14)
        
        stats_title = QLabel("DSP NOISE STATISTICS")
        stats_title.setStyleSheet("font-size: 12px; font-weight: 700; color: #38BDF8;")
        stats_layout.addWidget(stats_title)
        
        self.stats_table = QTableWidget(6, 2)
        self.stats_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.stats_table.verticalHeader().setVisible(False)
        
        stat_rows = ["Mean", "Variance", "Std Dev", "Peak-to-Peak", "Crest Factor", "Dominant Freq"]
        for idx, rname in enumerate(stat_rows):
            self.stats_table.setItem(idx, 0, QTableWidgetItem(rname))
            self.stats_table.setItem(idx, 1, QTableWidgetItem("--"))
            
        stats_layout.addWidget(self.stats_table)
        mid_layout.addWidget(self.stats_card, stretch=1)
        
        main_layout.addLayout(mid_layout)

        # 3. Bottom Section: ML Noise Classification
        ml_card = QFrame()
        ml_card.setObjectName("CardPanel")
        ml_layout = QVBoxLayout(ml_card)
        ml_layout.setContentsMargins(16, 16, 16, 16)
        
        ml_header_row = QHBoxLayout()
        ml_title_box = QVBoxLayout()
        ml_title = QLabel("ML-BASED NOISE CHARACTERIZATION")
        ml_title.setStyleSheet("font-size: 13px; font-weight: 700; color: #F8FAFC;")
        
        ml_sub = QLabel("Noise characterization combines signal-domain statistics and spectral characteristics using a calibrated SVM classifier.")
        ml_sub.setStyleSheet("font-size: 11px; color: #94A3B8;")
        ml_title_box.addWidget(ml_title)
        ml_title_box.addWidget(ml_sub)
        
        self.ml_model_badge = StatusBadge("Loaded" if session_manager.is_ml_model_loaded else "Not Loaded")
        
        ml_header_row.addLayout(ml_title_box)
        ml_header_row.addStretch()
        ml_header_row.addWidget(self.ml_model_badge)
        
        ml_layout.addLayout(ml_header_row)

        # ML Probability Bars Layout
        self.ml_bars_box = QFrame()
        self.ml_bars_box.setStyleSheet("background-color: #0F172A; border-radius: 6px; padding: 12px;")
        bars_layout = QVBoxLayout(self.ml_bars_box)
        
        self.class_bars = {}
        classes = ["Clean", "Gaussian", "Impulse", "Periodic", "Colored", "Mixed"]
        for cname in classes:
            row = QHBoxLayout()
            lbl = QLabel(f"{cname:10s}")
            lbl.setStyleSheet("font-size: 11px; font-weight: 600; color: #E2E8F0; font-family: monospace;")
            
            pbar = QProgressBar()
            pbar.setRange(0, 100)
            pbar.setValue(0)
            pbar.setFixedHeight(14)
            pbar.setFormat("%v %")
            
            row.addWidget(lbl)
            row.addWidget(pbar)
            bars_layout.addLayout(row)
            self.class_bars[cname] = pbar

        self.no_ml_label = QLabel("ML Model is not loaded. Train or provide a valid model in Settings to unlock classification.")
        self.no_ml_label.setStyleSheet("font-size: 11px; color: #F59E0B; font-weight: 600; text-align: center;")
        self.no_ml_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.no_ml_label.setVisible(not session_manager.is_ml_model_loaded)

        ml_layout.addWidget(self.ml_bars_box)
        ml_layout.addWidget(self.no_ml_label)
        
        main_layout.addWidget(ml_card)

        # Root layout
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.addWidget(scroll)

        # Session connection
        session_manager.session_updated.connect(self.run_noise_analysis)
        self.run_noise_analysis()

    def run_noise_analysis(self):
        sig = session_manager.session.active_signal
        fs = session_manager.session.active_sampling_rate
        if sig is None or len(sig) == 0:
            self.assessment_status.set_status("Ready")
            self.plot_widget.plot_spectrum(None, None)
            return

        # 1. DSP Analysis
        stats = analyze_noise(sig, fs)
        feat = extract_noise_features(sig, fs)
        
        snr = feat.get("snr_db", 0.0)
        rms = stats["rms"]
        
        if snr > 25.0:
            self.assessment_status.set_status("Clean")
        elif snr > 5.0:
            self.assessment_status.set_status("Warning")
        else:
            self.assessment_status.set_status("Error")
            
        self.snr_card.update_metric(f"{snr:.1f}", "dB")
        self.sig_rms_card.update_metric(f"{rms:.4f}", "")
        self.noise_rms_card.update_metric(f"{rms / (10**(snr/20)):.4f}" if snr < 90 else "0.0000", "")
        self.zcr_card.update_metric(f"{stats['zero_crossing_rate']:.4f}", "")

        # Table update
        svals = [stats["mean"], stats["variance"], stats["std"], stats["peak_to_peak"], stats["crest_factor"], stats["dominant_frequency"]]
        for idx, val in enumerate(svals):
            self.stats_table.setItem(idx, 1, QTableWidgetItem(f"{val:.4e}"))

        if len(stats["frequency_axis"]) > 0:
            self.plot_widget.plot_spectrum(stats["frequency_axis"], stats["psd"], "Noise Power Spectral Density")

        # 2. ML Classification if model exists
        if session_manager.is_ml_model_loaded:
            try:
                model = load_noise_model(session_manager.ml_model_path)
                # Feature array order matches ML_FEATURE_NAMES
                X = np.array([[feat[k] for k in feat]], dtype=np.float64)
                probs = model.predict_proba(X)[0]
                classes = model.classes_
                
                for cname, bar in self.class_bars.items():
                    if cname in classes:
                        c_idx = list(classes).index(cname)
                        bar.setValue(int(probs[c_idx] * 100))
                    else:
                        bar.setValue(0)
                        
                session_manager.session.noise["ml_probs"] = dict(zip(classes, probs))
            except Exception as e:
                print("ML Prediction Error:", e)
                self.no_ml_label.setText(f"ML Inference Error: {str(e)}")
                self.no_ml_label.setVisible(True)
        else:
            self.no_ml_label.setVisible(True)
