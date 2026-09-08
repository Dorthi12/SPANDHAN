"""
Dashboard Page module.
Provides an engineering overview with signal summary, visual pipeline progression, quick metrics, preview plot, and summary panel.
"""

import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QFrame, QLabel, QGridLayout, QScrollArea
from PySide6.QtCore import Qt, Signal

from app.widgets.section_header import SectionHeader
from app.widgets.metric_card import MetricCard
from app.widgets.pipeline_step import PipelineStepCard
from app.widgets.signal_plot import SignalPlotWidget
from app.widgets.empty_state import EmptyStateWidget
from core.session import session_manager


class DashboardPage(QWidget):
    """
    Main Dashboard View.
    """
    navigate_requested = Signal(str)

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
        
        # Outer Layout
        top_layout = QVBoxLayout()
        self.header = SectionHeader("Signal Analysis Dashboard", "Multi-domain digital signal analysis and diagnostics overview")
        top_layout.addWidget(self.header)
        main_layout.addLayout(top_layout)

        # 1. Hero / Signal Metadata Card
        self.hero_card = QFrame()
        self.hero_card.setObjectName("CardPanel")
        hero_layout = QGridLayout(self.hero_card)
        hero_layout.setContentsMargins(16, 12, 16, 12)
        
        hero_title = QLabel("CURRENT SIGNAL OVERVIEW")
        hero_title.setStyleSheet("font-size: 11px; font-weight: 700; color: #38BDF8; letter-spacing: 1px;")
        hero_layout.addWidget(hero_title, 0, 0, 1, 6)
        
        self.file_lbl = QLabel("Filename: --")
        self.domain_lbl = QLabel("Domain: General")
        self.rate_lbl = QLabel("Sampling Rate: --")
        self.duration_lbl = QLabel("Duration: --")
        self.samples_lbl = QLabel("Samples: --")
        self.channels_lbl = QLabel("Channels: 1")
        
        for idx, lbl in enumerate([self.file_lbl, self.domain_lbl, self.rate_lbl, self.duration_lbl, self.samples_lbl, self.channels_lbl]):
            lbl.setStyleSheet("font-size: 12px; font-weight: 600; color: #E2E8F0;")
            hero_layout.addWidget(lbl, 1, idx)
            
        main_layout.addWidget(self.hero_card)

        # 2. Pipeline Progression Status Bar
        pipeline_title = QLabel("ANALYSIS PIPELINE PROGRESSION")
        pipeline_title.setStyleSheet("font-size: 11px; font-weight: 700; color: #94A3B8; letter-spacing: 0.8px;")
        main_layout.addWidget(pipeline_title)
        
        self.pipeline_layout = QHBoxLayout()
        self.pipeline_layout.setSpacing(8)
        self.pipeline_cards = {}
        
        for stage in session_manager.get_pipeline_stages():
            card = PipelineStepCard(stage["name"], stage["desc"], stage["status"])
            card.clicked.connect(self.navigate_requested.emit)
            self.pipeline_layout.addWidget(card)
            self.pipeline_cards[stage["name"]] = card
            
        main_layout.addLayout(self.pipeline_layout)

        # 3. Quick Analysis Metric Cards
        metrics_title = QLabel("QUICK SIGNAL METRICS")
        metrics_title.setStyleSheet("font-size: 11px; font-weight: 700; color: #94A3B8; letter-spacing: 0.8px;")
        main_layout.addWidget(metrics_title)
        
        metrics_layout = QHBoxLayout()
        metrics_layout.setSpacing(12)
        
        self.snr_card = MetricCard("Signal Quality (SNR)", "N/A", "dB", "Noise Assessment", "#10B981")
        self.dom_freq_card = MetricCard("Dominant Freq", "N/A", "Hz", "Spectral Peak", "#38BDF8")
        self.rms_card = MetricCard("Signal RMS", "N/A", "", "Time Domain", "#F59E0B")
        self.peak_rate_card = MetricCard("Peak Count Rate", "N/A", "", "Impulse Metric", "#A855F7")
        self.energy_card = MetricCard("Total Energy", "N/A", "J", "Signal Power", "#EC4899")
        
        metrics_layout.addWidget(self.snr_card)
        metrics_layout.addWidget(self.dom_freq_card)
        metrics_layout.addWidget(self.rms_card)
        metrics_layout.addWidget(self.peak_rate_card)
        metrics_layout.addWidget(self.energy_card)
        
        main_layout.addLayout(metrics_layout)

        # 4. Preview Plot & Summary Side Panel
        preview_layout = QHBoxLayout()
        preview_layout.setSpacing(14)
        
        self.plot_widget = SignalPlotWidget("Dashboard Signal Preview")
        preview_layout.addWidget(self.plot_widget, stretch=3)
        
        self.summary_panel = QFrame()
        self.summary_panel.setObjectName("CardPanel")
        sum_layout = QVBoxLayout(self.summary_panel)
        sum_layout.setContentsMargins(14, 14, 14, 14)
        
        sum_title = QLabel("RECENT ANALYSIS SUMMARY")
        sum_title.setStyleSheet("font-size: 12px; font-weight: 700; color: #38BDF8;")
        
        self.summary_text = QLabel("No active session results yet.\n\nUse the left sidebar or pipeline cards to run DSP analysis, Noise characterization, or Domain Diagnostics.")
        self.summary_text.setStyleSheet("font-size: 11px; color: #94A3B8; line-height: 1.4;")
        self.summary_text.setWordWrap(True)
        
        sum_layout.addWidget(sum_title)
        sum_layout.addWidget(self.summary_text)
        sum_layout.addStretch()
        
        preview_layout.addWidget(self.summary_panel, stretch=1)
        main_layout.addLayout(preview_layout)

        # Root layout setup
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.addWidget(scroll)

        # Connect session updates
        session_manager.session_updated.connect(self.update_dashboard)
        session_manager.pipeline_updated.connect(self.update_pipeline)
        
        self.update_dashboard()

    def update_dashboard(self):
        raw = session_manager.session.raw
        if raw is not None and len(raw.signal) > 0:
            self.file_lbl.setText(f"Filename: {raw.filename}")
            self.domain_lbl.setText(f"Domain: {raw.domain.upper()}")
            self.rate_lbl.setText(f"Sampling Rate: {raw.sampling_rate:.0f} Hz")
            self.duration_lbl.setText(f"Duration: {raw.duration:.2f} s")
            self.samples_lbl.setText(f"Samples: {raw.num_samples}")
            self.channels_lbl.setText(f"Channels: {raw.channels}")
            
            sig = session_manager.session.active_signal
            fs = session_manager.session.active_sampling_rate
            t = np.arange(len(sig)) / fs
            self.plot_widget.plot_time_domain(t, sig, "Active Signal Preview (Time Domain)")
            
            # Update quick metrics if available
            features = session_manager.session.features
            spectral = session_manager.session.spectral
            
            if features:
                self.snr_card.update_metric(f"{features.get('snr_db', 0.0):.1f}", "dB", "Estimated SNR")
                self.rms_card.update_metric(f"{features.get('rms', 0.0):.4f}", "", "Root Mean Square")
                self.peak_rate_card.update_metric(f"{features.get('peak_count_rate', 0.0):.4f}", "", "Peak Exceedance Rate")
                self.energy_card.update_metric(f"{np.sum(sig**2):.2e}", "J", "Total Power")
            else:
                self.rms_card.update_metric(f"{np.sqrt(np.mean(sig**2)):.4f}", "", "Root Mean Square")
                self.energy_card.update_metric(f"{np.sum(sig**2):.2e}", "J", "Total Power")
                
            if spectral and "peak_freq" in spectral:
                self.dom_freq_card.update_metric(f"{spectral['peak_freq']:.1f}", "Hz", "Peak Spectrum")
            else:
                self.dom_freq_card.update_metric("N/A", "Hz", "Peak Spectrum")

            # Summary panel update
            logs = session_manager.session.log
            if logs:
                self.summary_text.setText("\n".join(f"• {log}" for log in logs[-6:]))
        else:
            self.file_lbl.setText("Filename: --")
            self.domain_lbl.setText("Domain: General")
            self.rate_lbl.setText("Sampling Rate: --")
            self.duration_lbl.setText("Duration: --")
            self.samples_lbl.setText("Samples: --")
            self.channels_lbl.setText("Channels: 1")
            self.plot_widget.plot_time_domain(None, None)

    def update_pipeline(self):
        for stage in session_manager.get_pipeline_stages():
            if stage["name"] in self.pipeline_cards:
                self.pipeline_cards[stage["name"]].set_status(stage["status"])
