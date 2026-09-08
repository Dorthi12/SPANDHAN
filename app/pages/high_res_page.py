"""
High-Resolution Spectral Analysis Page module.
Flagship feature demonstrating sub-resolution frequency estimation via MUSIC and ESPRIT algorithms.
"""

import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QFrame, QLabel, QPushButton, QSpinBox, QFormLayout, QScrollArea, QTableWidget, QTableWidgetItem, QHeaderView
from PySide6.QtCore import Qt

from app.widgets.section_header import SectionHeader
from app.widgets.signal_plot import SignalPlotWidget
from app.widgets.results_table import ResultsTableWidget
from core.session import session_manager
from dsp.fft import compute_fft
from dsp.music import estimate_music_frequencies
from dsp.esprit import estimate_esprit_frequencies


class HighResPage(QWidget):
    """
    High-Resolution Subspace Spectral Estimation View (MUSIC & ESPRIT vs FFT).
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
            "High-Resolution Spectral Analysis",
            "Sub-resolution frequency estimation using MUSIC and ESPRIT subspace algorithms."
        )
        main_layout.addWidget(self.header)

        # Primary Split Layout
        split_layout = QHBoxLayout()
        split_layout.setSpacing(16)
        
        # Left Parameters & Technical Panel
        left_box = QVBoxLayout()
        left_box.setSpacing(14)
        
        # Parameter Card
        param_card = QFrame()
        param_card.setObjectName("CardPanel")
        param_layout = QVBoxLayout(param_card)
        param_layout.setContentsMargins(14, 14, 14, 14)
        
        param_title = QLabel("SUBSPACE ESTIMATOR CONTROLS")
        param_title.setStyleSheet("font-size: 13px; font-weight: 700; color: #38BDF8;")
        param_layout.addWidget(param_title)
        
        form = QFormLayout()
        form.setSpacing(10)
        
        self.model_order_spin = QSpinBox()
        self.model_order_spin.setRange(2, 200)
        self.model_order_spin.setValue(20)
        self.model_order_spin.setToolTip("Dimension of covariance snapshot matrix.")
        form.addRow("Model Order (M):", self.model_order_spin)
        
        self.num_sources_spin = QSpinBox()
        self.num_sources_spin.setRange(1, 50)
        self.num_sources_spin.setValue(2)
        self.num_sources_spin.setToolTip("Number of physical sinusoidal sources to extract.")
        form.addRow("Number of Sources (p):", self.num_sources_spin)
        
        self.nfft_spin = QSpinBox()
        self.nfft_spin.setRange(512, 16384)
        self.nfft_spin.setValue(4096)
        self.nfft_spin.setSingleStep(512)
        form.addRow("MUSIC Grid NFFT:", self.nfft_spin)
        
        param_layout.addLayout(form)
        
        self.run_btn = QPushButton("Run High-Res Estimation")
        self.run_btn.setObjectName("PrimaryButton")
        self.run_btn.clicked.connect(self.run_high_res_analysis)
        param_layout.addWidget(self.run_btn)
        
        left_box.addWidget(param_card)

        # Technical Explanation Panel
        info_card = QFrame()
        info_card.setObjectName("CardPanel")
        info_layout = QVBoxLayout(info_card)
        info_layout.setContentsMargins(14, 14, 14, 14)
        
        info_title = QLabel("ALGORITHM CHARACTERISTICS")
        info_title.setStyleSheet("font-size: 12px; font-weight: 700; color: #94A3B8;")
        info_layout.addWidget(info_title)
        
        desc = QLabel(
            "<b>FFT:</b> Conventional Fourier estimate limited by Rayleigh resolution limit Δf = 1/T.<br><br>"
            "<b>MUSIC:</b> Multiple Signal Classification. Exploits noise subspace orthogonality to form pseudospectrum.<br><br>"
            "<b>ESPRIT:</b> Estimation of Signal Parameters via Rotational Invariance Techniques. Search-free estimation of precise frequencies."
        )
        desc.setStyleSheet("font-size: 11px; color: #94A3B8; line-height: 1.4;")
        desc.setWordWrap(True)
        info_layout.addWidget(desc)
        
        left_box.addWidget(info_card)
        left_box.addStretch()
        
        split_layout.addLayout(left_box, stretch=1)

        # Right Plot & Table Area
        right_box = QVBoxLayout()
        right_box.setSpacing(14)
        
        # Plot
        self.plot_card = QFrame()
        self.plot_card.setObjectName("CardPanel")
        plot_layout = QVBoxLayout(self.plot_card)
        plot_layout.setContentsMargins(14, 14, 14, 14)
        
        self.plot_widget = SignalPlotWidget("High-Resolution Frequency Comparison")
        plot_layout.addWidget(self.plot_widget)
        
        right_box.addWidget(self.plot_card, stretch=2)

        # Table
        self.table_card = QFrame()
        self.table_card.setObjectName("CardPanel")
        tbl_layout = QVBoxLayout(self.table_card)
        tbl_layout.setContentsMargins(14, 14, 14, 14)
        
        tbl_title = QLabel("ESTIMATED FREQUENCY TABLE")
        tbl_title.setStyleSheet("font-size: 12px; font-weight: 700; color: #38BDF8; margin-bottom: 4px;")
        tbl_layout.addWidget(tbl_title)
        
        self.results_table = ResultsTableWidget(["Method", "Peak / Source #", "Frequency (Hz)", "Status"])
        tbl_layout.addWidget(self.results_table)
        
        right_box.addWidget(self.table_card, stretch=1)
        
        split_layout.addLayout(right_box, stretch=3)
        main_layout.addLayout(split_layout)

        # Root layout
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.addWidget(scroll)

        # Session updates
        session_manager.session_updated.connect(self.run_high_res_analysis)
        self.run_high_res_analysis()

    def run_high_res_analysis(self):
        sig = session_manager.session.active_signal
        fs = session_manager.session.active_sampling_rate
        if sig is None or len(sig) == 0:
            self.plot_widget.plot_high_res_comparison(None, None, None, None, None)
            self.results_table.set_data([])
            return

        order = min(self.model_order_spin.value(), len(sig) // 2 - 1)
        sources = min(self.num_sources_spin.value(), order // 2 - 1)
        nfft = self.nfft_spin.value()
        
        if order < 2 or sources < 1:
            return

        try:
            # 1. FFT
            fft_freqs, fft_mag = compute_fft(sig, fs)
            
            # 2. MUSIC
            music_res = estimate_music_frequencies(sig, fs, model_order=order, num_sources=sources, nfft=nfft)
            music_freqs = music_res["frequency_axis"]
            music_psd = music_res["pseudospectrum"]
            music_peaks = music_res["frequencies"]
            
            # 3. ESPRIT
            esprit_res = estimate_esprit_frequencies(sig, fs, model_order=order, num_sources=sources)
            esprit_peaks = esprit_res["frequencies"]
            
            self.plot_widget.plot_high_res_comparison(fft_freqs, fft_mag, music_freqs, music_psd, esprit_peaks)
            
            # Table data
            rows = []
            for idx, f in enumerate(music_peaks):
                rows.append(["MUSIC", f"Source #{idx+1}", f"{f:.3f}", "Subspace Peak"])
            for idx, f in enumerate(esprit_peaks):
                rows.append(["ESPRIT", f"Source #{idx+1}", f"{f:.3f}", "Search-free Estimate"])
                
            self.results_table.set_data(rows)
            
            # Save to session
            session_manager.session.high_res["music_peaks"] = music_peaks
            session_manager.session.high_res["esprit_peaks"] = esprit_peaks
        except Exception as e:
            print("High-Res Analysis error:", e)
