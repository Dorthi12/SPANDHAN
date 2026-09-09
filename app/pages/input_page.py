"""
Input Page module.
Provides file loading (WAV, CSV, MAT, TXT) and engineering synthetic test-signal generation.
"""

import os
import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QFrame, QLabel, QPushButton, QComboBox, QDoubleSpinBox, QSpinBox, QFormLayout, QScrollArea, QFileDialog
from PySide6.QtCore import Qt, Signal

from app.widgets.section_header import SectionHeader
from app.widgets.file_drop_zone import FileDropZone
from app.widgets.signal_plot import SignalPlotWidget
from core.session import session_manager
from data_io.signal_loader import load_signal
from generators.waveform_generator import generate_sine, generate_multi_tone, generate_square, generate_sawtooth
from generators.noise_generator import add_gaussian_noise, add_impulse_noise


class InputPage(QWidget):
    """
    Signal Input View.
    """
    signal_ready = Signal()

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
        self.header = SectionHeader("Signal Input Workspace", "Import digital signals from files or generate precision engineering test signals.")
        main_layout.addWidget(self.header)

        # Primary Cards Layout (Split into Load Signal vs Synthetic Generator)
        cards_layout = QHBoxLayout()
        cards_layout.setSpacing(16)
        
        # Card A: LOAD SIGNAL FILE
        self.load_card = QFrame()
        self.load_card.setObjectName("CardPanel")
        load_layout = QVBoxLayout(self.load_card)
        load_layout.setContentsMargins(14, 14, 14, 14)
        
        load_title = QLabel("A. LOAD SIGNAL FILE")
        load_title.setStyleSheet("font-size: 13px; font-weight: 700; color: #38BDF8;")
        load_layout.addWidget(load_title)
        
        self.drop_zone = FileDropZone()
        self.drop_zone.file_dropped.connect(self.load_file_path)
        load_layout.addWidget(self.drop_zone)
        
        self.file_info_lbl = QLabel("No file selected.\nSupported formats: .wav, .csv, .mat, .txt")
        self.file_info_lbl.setStyleSheet("font-size: 11px; color: #94A3B8; background-color: #0F172A; padding: 10px; border-radius: 6px;")
        load_layout.addWidget(self.file_info_lbl)
        
        cards_layout.addWidget(self.load_card, stretch=1)

        # Card B: SYNTHETIC TEST SIGNAL GENERATOR
        self.gen_card = QFrame()
        self.gen_card.setObjectName("CardPanel")
        gen_layout = QVBoxLayout(self.gen_card)
        gen_layout.setContentsMargins(14, 14, 14, 14)
        
        gen_title = QLabel("B. SYNTHETIC TEST-SIGNAL GENERATOR")
        gen_title.setStyleSheet("font-size: 13px; font-weight: 700; color: #38BDF8;")
        gen_layout.addWidget(gen_title)
        
        form = QFormLayout()
        form.setSpacing(8)
        
        self.waveform_type_cb = QComboBox()
        self.waveform_type_cb.addItems(["Sine", "Multi-tone", "Chirp", "Square", "Triangle", "Composite"])
        form.addRow("Waveform Type:", self.waveform_type_cb)
        
        self.freq_spin = QDoubleSpinBox()
        self.freq_spin.setRange(0.1, 100000.0)
        self.freq_spin.setValue(50.0)
        self.freq_spin.setSuffix(" Hz")
        form.addRow("Fundamental Freq:", self.freq_spin)
        
        self.amp_spin = QDoubleSpinBox()
        self.amp_spin.setRange(0.01, 1000.0)
        self.amp_spin.setValue(1.0)
        form.addRow("Amplitude:", self.amp_spin)
        
        self.fs_spin = QDoubleSpinBox()
        self.fs_spin.setRange(10.0, 1000000.0)
        self.fs_spin.setValue(1000.0)
        self.fs_spin.setSuffix(" Hz")
        form.addRow("Sampling Rate:", self.fs_spin)
        
        self.dur_spin = QDoubleSpinBox()
        self.dur_spin.setRange(0.01, 600.0)
        self.dur_spin.setValue(2.0)
        self.dur_spin.setSuffix(" s")
        form.addRow("Duration:", self.dur_spin)
        
        self.num_tones_spin = QSpinBox()
        self.num_tones_spin.setRange(1, 10)
        self.num_tones_spin.setValue(3)
        form.addRow("Number of Tones:", self.num_tones_spin)
        
        # Noise controls
        self.noise_type_cb = QComboBox()
        self.noise_type_cb.addItems(["None", "Gaussian", "Impulse", "Periodic", "Colored", "Mixed"])
        form.addRow("Add Noise Type:", self.noise_type_cb)
        
        self.snr_spin = QDoubleSpinBox()
        self.snr_spin.setRange(-30.0, 100.0)
        self.snr_spin.setValue(20.0)
        self.snr_spin.setSuffix(" dB")
        form.addRow("Noise SNR:", self.snr_spin)
        
        gen_layout.addLayout(form)
        
        self.generate_btn = QPushButton("Generate Test Signal")
        self.generate_btn.setObjectName("SuccessButton")
        self.generate_btn.clicked.connect(self.generate_signal)
        gen_layout.addWidget(self.generate_btn)
        
        cards_layout.addWidget(self.gen_card, stretch=1)
        main_layout.addLayout(cards_layout)

        # Plot Preview Card
        self.plot_card = QFrame()
        self.plot_card.setObjectName("CardPanel")
        plot_layout = QVBoxLayout(self.plot_card)
        plot_layout.setContentsMargins(14, 14, 14, 14)
        
        self.plot_widget = SignalPlotWidget("Loaded / Generated Signal Preview")
        plot_layout.addWidget(self.plot_widget)
        
        main_layout.addWidget(self.plot_card)

        # Root layout
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.addWidget(scroll)

        # Connect session updates
        session_manager.session_updated.connect(self.on_session_updated)

    def load_file_path(self, filepath: str):
        try:
            current_fs = self.fs_spin.value()
            signal_data = load_signal(filepath, sampling_rate=current_fs)
            session_manager.load_signal(
                signal_data.signal,
                signal_data.sampling_rate,
                filename=os.path.basename(filepath),
                domain=signal_data.domain,
                source=signal_data.source or "file"
            )
            
            # Synchronize sampling rate spinbox with the loaded file
            if signal_data.sampling_rate > 0:
                self.fs_spin.setValue(signal_data.sampling_rate)

            filename = os.path.basename(filepath)
            self.file_info_lbl.setText(
                f"File: {filename}\nType: {os.path.splitext(filename)[1].upper()}\nSize: {os.path.getsize(filepath)/1024:.1f} KB\n"
                f"Sampling Rate: {signal_data.sampling_rate:.0f} Hz | Duration: {signal_data.duration:.2f} s | Samples: {signal_data.num_samples:,}"
            )
            self.file_info_lbl.setStyleSheet("font-size: 11px; color: #4ADE80; background-color: #0F172A; padding: 10px; border-radius: 6px;")
        except Exception as e:
            self.file_info_lbl.setText(f"Error loading file: {str(e)}")
            self.file_info_lbl.setStyleSheet("font-size: 11px; color: #F87171; background-color: #0F172A; padding: 10px; border-radius: 6px;")

    def generate_signal(self):
        try:
            wtype = self.waveform_type_cb.currentText()
            freq = self.freq_spin.value()
            amp = self.amp_spin.value()
            fs = self.fs_spin.value()
            dur = self.dur_spin.value()
            ntones = self.num_tones_spin.value()
            
            if wtype == "Sine":
                _, sig = generate_sine(freq, fs, dur, amplitude=amp)
            elif wtype == "Multi-tone":
                freqs = [freq * (i + 1) for i in range(ntones)]
                amps = [amp / (i + 1) for i in range(ntones)]
                _, sig = generate_multi_tone(freqs, fs, dur, amplitudes=amps)
            elif wtype == "Square":
                _, sig = generate_square(freq, fs, dur, amplitude=amp)
            elif wtype in ["Triangle", "Chirp", "Sawtooth"]:
                _, sig = generate_sawtooth(freq, fs, dur, amplitude=amp)
            else:
                # Composite
                _, sig1 = generate_sine(freq, fs, dur, amplitude=amp)
                _, sig2 = generate_sine(freq * 2.5, fs, dur, amplitude=amp * 0.4)
                sig = sig1 + sig2
                
            noise_type = self.noise_type_cb.currentText()
            if noise_type != "None":
                snr = self.snr_spin.value()
                if noise_type == "Impulse":
                    sig = add_impulse_noise(sig, snr_db=snr)
                else:
                    sig = add_gaussian_noise(sig, snr_db=snr)

            filename = f"Synthetic_{wtype}_{freq:.0f}Hz.wav"
            session_manager.load_signal(sig, fs, filename=filename, domain="general", source="synthetic")
        except Exception as e:
            self.file_info_lbl.setText(f"Generation error: {str(e)}")

    def on_session_updated(self):
        sig = session_manager.session.active_signal
        fs = session_manager.session.active_sampling_rate
        if sig is not None and len(sig) > 0:
            t = np.arange(len(sig)) / fs
            self.plot_widget.plot_time_domain(t, sig, f"Input Signal Preview ({session_manager.session.raw.filename})")
        else:
            self.plot_widget.plot_time_domain(None, None)
