"""
Preprocessing Page module.
Provides synchronized BEFORE -> AFTER plots, DC removal, detrending, normalization, denoising, and digital filtering.
"""

import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QFrame, QLabel, QPushButton, QComboBox, QDoubleSpinBox, QSpinBox, QFormLayout, QScrollArea, QTableWidget, QTableWidgetItem, QHeaderView
from PySide6.QtCore import Qt

from app.widgets.section_header import SectionHeader
from app.widgets.signal_plot import SignalPlotWidget
from app.widgets.empty_state import EmptyStateWidget
from core.session import session_manager
from preprocessing.dc_removal import remove_dc
from preprocessing.detrending import detrend_signal
from preprocessing.normalization import normalize_signal
from dsp.filters import apply_lowpass, apply_highpass, apply_bandpass, apply_bandstop


class PreprocessingPage(QWidget):
    """
    Signal Preprocessing View.
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
        self.header = SectionHeader("Signal Preprocessing", "Condition signal using zero-phase filtering, detrending, DC removal, and amplitude normalization.")
        main_layout.addWidget(self.header)

        # Content Layout
        content_layout = QHBoxLayout()
        content_layout.setSpacing(16)
        
        # Left Control Panel
        self.control_card = QFrame()
        self.control_card.setObjectName("CardPanel")
        ctrl_layout = QVBoxLayout(self.control_card)
        ctrl_layout.setContentsMargins(14, 14, 14, 14)
        
        ctrl_title = QLabel("PREPROCESSING OPERATIONS")
        ctrl_title.setStyleSheet("font-size: 13px; font-weight: 700; color: #38BDF8;")
        ctrl_layout.addWidget(ctrl_title)
        
        form = QFormLayout()
        form.setSpacing(10)
        
        self.op_cb = QComboBox()
        self.op_cb.addItems(["Low-pass Filter", "High-pass Filter", "Band-pass Filter", "Band-stop Filter", "DC Removal", "Detrending", "Peak Normalization", "Z-Score Normalization"])
        self.op_cb.currentTextChanged.connect(self.on_op_changed)
        form.addRow("Operation:", self.op_cb)
        
        self.cutoff_spin = QDoubleSpinBox()
        self.cutoff_spin.setRange(0.1, 50000.0)
        self.cutoff_spin.setValue(100.0)
        self.cutoff_spin.setSuffix(" Hz")
        form.addRow("Cutoff Freq:", self.cutoff_spin)
        
        self.low_cutoff_spin = QDoubleSpinBox()
        self.low_cutoff_spin.setRange(0.1, 50000.0)
        self.low_cutoff_spin.setValue(20.0)
        self.low_cutoff_spin.setSuffix(" Hz")
        form.addRow("Low Cutoff Freq:", self.low_cutoff_spin)
        
        self.high_cutoff_spin = QDoubleSpinBox()
        self.high_cutoff_spin.setRange(0.1, 50000.0)
        self.high_cutoff_spin.setValue(200.0)
        self.high_cutoff_spin.setSuffix(" Hz")
        form.addRow("High Cutoff Freq:", self.high_cutoff_spin)
        
        self.order_spin = QSpinBox()
        self.order_spin.setRange(1, 10)
        self.order_spin.setValue(4)
        form.addRow("Filter Order:", self.order_spin)
        
        ctrl_layout.addLayout(form)
        
        # Action Buttons
        btn_layout = QHBoxLayout()
        self.apply_btn = QPushButton("Apply Operation")
        self.apply_btn.setObjectName("PrimaryButton")
        self.apply_btn.clicked.connect(self.apply_operation)
        
        self.reset_btn = QPushButton("Reset Raw")
        self.reset_btn.clicked.connect(self.reset_preprocessing)
        
        btn_layout.addWidget(self.apply_btn)
        btn_layout.addWidget(self.reset_btn)
        ctrl_layout.addLayout(btn_layout)
        
        # Metrics Comparison Table
        metrics_title = QLabel("SIGNAL METRICS COMPARISON")
        metrics_title.setStyleSheet("font-size: 12px; font-weight: 700; color: #94A3B8; margin-top: 10px;")
        ctrl_layout.addWidget(metrics_title)
        
        self.metrics_table = QTableWidget(5, 3)
        self.metrics_table.setHorizontalHeaderLabels(["Metric", "Original", "Processed"])
        self.metrics_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.metrics_table.verticalHeader().setVisible(False)
        self.metrics_table.setMaximumHeight(160)
        
        metric_names = ["Mean", "RMS", "Peak", "Variance", "Energy"]
        for idx, name in enumerate(metric_names):
            self.metrics_table.setItem(idx, 0, QTableWidgetItem(name))
            self.metrics_table.setItem(idx, 1, QTableWidgetItem("--"))
            self.metrics_table.setItem(idx, 2, QTableWidgetItem("--"))
            
        ctrl_layout.addWidget(self.metrics_table)
        ctrl_layout.addStretch()
        
        content_layout.addWidget(self.control_card, stretch=1)

        # Right Plots Panel (Synchronized BEFORE -> AFTER)
        self.plot_card = QFrame()
        self.plot_card.setObjectName("CardPanel")
        plot_layout = QVBoxLayout(self.plot_card)
        plot_layout.setContentsMargins(14, 14, 14, 14)
        
        self.plot_widget = SignalPlotWidget("Before vs After Preprocessing")
        plot_layout.addWidget(self.plot_widget)
        
        content_layout.addWidget(self.plot_card, stretch=2)
        main_layout.addLayout(content_layout)

        # Root layout
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.addWidget(scroll)

        # Session connection
        session_manager.session_updated.connect(self.update_view)
        self.on_op_changed(self.op_cb.currentText())
        self.update_view()

    def on_op_changed(self, op_text: str):
        is_lp_hp = op_text in ["Low-pass Filter", "High-pass Filter"]
        is_bp_bs = op_text in ["Band-pass Filter", "Band-stop Filter"]
        
        self.cutoff_spin.setVisible(is_lp_hp)
        self.low_cutoff_spin.setVisible(is_bp_bs)
        self.high_cutoff_spin.setVisible(is_bp_bs)
        self.order_spin.setVisible(is_lp_hp or is_bp_bs)

    def apply_operation(self):
        raw = session_manager.session.raw
        if raw is None or len(raw.signal) == 0:
            return
            
        current_sig = session_manager.session.active_signal
        fs = session_manager.session.active_sampling_rate
        op_text = self.op_cb.currentText()
        
        try:
            if op_text == "Low-pass Filter":
                fc = min(self.cutoff_spin.value(), (fs / 2.0) - 1.0)
                processed = apply_lowpass(current_sig, fs, fc, self.order_spin.value())
            elif op_text == "High-pass Filter":
                fc = min(self.cutoff_spin.value(), (fs / 2.0) - 1.0)
                processed = apply_highpass(current_sig, fs, fc, self.order_spin.value())
            elif op_text == "Band-pass Filter":
                flo = self.low_cutoff_spin.value()
                fhi = min(self.high_cutoff_spin.value(), (fs / 2.0) - 1.0)
                processed = apply_bandpass(current_sig, fs, flo, fhi, self.order_spin.value())
            elif op_text == "Band-stop Filter":
                flo = self.low_cutoff_spin.value()
                fhi = min(self.high_cutoff_spin.value(), (fs / 2.0) - 1.0)
                processed = apply_bandstop(current_sig, fs, flo, fhi, self.order_spin.value())
            elif op_text == "DC Removal":
                processed = remove_dc(current_sig)
            elif op_text == "Detrending":
                processed = detrend_signal(current_sig)
            elif op_text == "Peak Normalization":
                processed = normalize_signal(current_sig, method="peak")
            elif op_text == "Z-Score Normalization":
                processed = normalize_signal(current_sig, method="zscore")
            else:
                processed = current_sig

            session_manager.update_processed_signal(processed, f"{op_text}")
        except Exception as e:
            print("Preprocessing error:", e)

    def reset_preprocessing(self):
        session_manager.reset_preprocessing()

    def update_view(self):
        raw = session_manager.session.raw
        if raw is not None and len(raw.signal) > 0:
            raw_sig = raw.signal
            proc_sig = session_manager.session.processed_signal
            fs = raw.sampling_rate
            t = np.arange(len(raw_sig)) / fs
            
            self.plot_widget.plot_before_after(t, raw_sig, proc_sig)
            
            # Metrics
            orig_metrics = [np.mean(raw_sig), np.sqrt(np.mean(raw_sig**2)), np.max(np.abs(raw_sig)), np.var(raw_sig), np.sum(raw_sig**2)]
            for idx, val in enumerate(orig_metrics):
                self.metrics_table.setItem(idx, 1, QTableWidgetItem(f"{val:.4e}"))
                
            if proc_sig is not None:
                proc_metrics = [np.mean(proc_sig), np.sqrt(np.mean(proc_sig**2)), np.max(np.abs(proc_sig)), np.var(proc_sig), np.sum(proc_sig**2)]
                for idx, val in enumerate(proc_metrics):
                    self.metrics_table.setItem(idx, 2, QTableWidgetItem(f"{val:.4e}"))
            else:
                for idx in range(5):
                    self.metrics_table.setItem(idx, 2, QTableWidgetItem("--"))
        else:
            self.plot_widget.plot_before_after(None, None, None)
