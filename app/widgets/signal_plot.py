"""
SignalPlot Widget using Matplotlib QtAgg Canvas for high-precision PySide6 rendering.
Supports time-domain, frequency-domain, spectrograms, MUSIC/ESPRIT overlays, and MCSA sidebands.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('QtAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QCheckBox
from PySide6.QtCore import Qt


class MplCanvas(FigureCanvasQTAgg):
    """Matplotlib Canvas configured with dark neutral engineering palette."""
    def __init__(self, width=8, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor='#0F172A')
        super().__init__(self.fig)
        self.setStyleSheet("background-color: #0F172A;")


class SignalPlotWidget(QWidget):
    """
    Complete interactive plotting panel with Toolbar & Toggle Controls.
    """
    def __init__(self, title: str = "Plot", parent=None):
        super().__init__(parent)
        self.title = title
        
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(4)
        
        # Matplotlib canvas
        self.canvas = MplCanvas()
        
        # Toolbar and control bar
        controls_layout = QHBoxLayout()
        controls_layout.setContentsMargins(4, 0, 4, 0)
        
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.toolbar.setStyleSheet("""
            QToolBar { background-color: #1E293B; border: none; }
            QToolButton { color: #F8FAFC; background-color: transparent; border-radius: 4px; padding: 2px; }
            QToolButton:hover { background-color: #334155; }
        """)
        
        self.grid_cb = QCheckBox("Grid")
        self.grid_cb.setChecked(True)
        self.grid_cb.setStyleSheet("color: #94A3B8; font-size: 11px;")
        self.grid_cb.stateChanged.connect(self._toggle_grid)
        
        self.export_btn = QPushButton("Export Figure")
        self.export_btn.setStyleSheet("font-size: 11px; padding: 4px 8px;")
        self.export_btn.clicked.connect(self.export_figure)
        
        controls_layout.addWidget(self.toolbar)
        controls_layout.addStretch()
        controls_layout.addWidget(self.grid_cb)
        controls_layout.addWidget(self.export_btn)
        
        main_layout.addLayout(controls_layout)
        main_layout.addWidget(self.canvas)
        
        # State settings
        self.show_grid = True

    def _setup_ax(self, ax):
        ax.set_facecolor('#0F172A')
        ax.tick_params(colors='#94A3B8', labelsize=9)
        for spine in ax.spines.values():
            spine.set_color('#334155')
        ax.xaxis.label.set_color('#CBD5E1')
        ax.yaxis.label.set_color('#CBD5E1')
        ax.title.set_color('#F8FAFC')
        if self.show_grid:
            ax.grid(True, color='#1E293B', linestyle='--', linewidth=0.7)

    def _toggle_grid(self, state):
        self.show_grid = (state == Qt.CheckState.Checked.value)
        for ax in self.canvas.fig.get_axes():
            ax.grid(self.show_grid, color='#1E293B', linestyle='--', linewidth=0.7)
        self.canvas.draw()

    def plot_time_domain(self, time_axis: np.ndarray, signal_data: np.ndarray, title: str = "Time Domain Signal", xlabel: str = "Time (s)", ylabel: str = "Amplitude", color: str = "#38BDF8"):
        self.canvas.fig.clear()
        ax = self.canvas.fig.add_subplot(111)
        self._setup_ax(ax)
        
        if time_axis is not None and signal_data is not None and len(signal_data) > 0:
            ax.plot(time_axis, signal_data, color=color, linewidth=1.2, label="Signal")
            ax.set_xlim(time_axis[0], time_axis[-1])
        else:
            ax.text(0.5, 0.5, "No signal loaded", color="#64748B", ha='center', va='center', transform=ax.transAxes, fontsize=12)
            
        ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        self.canvas.draw()

    def plot_before_after(self, time_axis: np.ndarray, raw_signal: np.ndarray, processed_signal: np.ndarray, title: str = "Before vs After Preprocessing"):
        self.canvas.fig.clear()
        ax1 = self.canvas.fig.add_subplot(211)
        ax2 = self.canvas.fig.add_subplot(212, sharex=ax1)
        
        self._setup_ax(ax1)
        self._setup_ax(ax2)
        
        if time_axis is not None and len(time_axis) > 0:
            ax1.plot(time_axis, raw_signal, color="#94A3B8", linewidth=1.0, label="Raw")
            ax1.set_title("Original Raw Signal", fontsize=10, pad=4)
            ax1.set_ylabel("Amplitude", fontsize=8)
            
            if processed_signal is not None:
                ax2.plot(time_axis, processed_signal, color="#10B981", linewidth=1.2, label="Processed")
                ax2.set_title("Preprocessed Signal", fontsize=10, pad=4)
            else:
                ax2.text(0.5, 0.5, "Click 'Apply' to process", color="#64748B", ha='center', va='center', transform=ax2.transAxes)
            ax2.set_xlabel("Time (s)", fontsize=8)
            ax2.set_ylabel("Amplitude", fontsize=8)
        
        self.canvas.fig.tight_layout()
        self.canvas.draw()

    def plot_spectrum(self, freqs: np.ndarray, mag: np.ndarray, title: str = "Magnitude Spectrum", log_scale: str = "linear", peaks=None):
        self.canvas.fig.clear()
        ax = self.canvas.fig.add_subplot(111)
        self._setup_ax(ax)
        
        if freqs is not None and mag is not None and len(freqs) > 0:
            y_data = 20 * np.log10(np.maximum(mag, 1e-12)) if log_scale == "dB" else mag
            ax.plot(freqs, y_data, color="#38BDF8", linewidth=1.2)
            ax.set_xlim(freqs[0], freqs[-1])
            if peaks is not None and len(peaks) > 0:
                ax.plot(peaks["freqs"], peaks["mags"], "v", color="#EF4444", markersize=6, label="Peaks")
        else:
            ax.text(0.5, 0.5, "No spectral data available", color="#64748B", ha='center', va='center', transform=ax.transAxes)

        ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
        ax.set_xlabel("Frequency (Hz)", fontsize=9)
        ax.set_ylabel("Magnitude (dB)" if log_scale == "dB" else "Magnitude", fontsize=9)
        self.canvas.draw()

    def plot_spectrogram(self, times: np.ndarray, freqs: np.ndarray, Sxx: np.ndarray, title: str = "Spectrogram (STFT)"):
        self.canvas.fig.clear()
        ax = self.canvas.fig.add_subplot(111)
        self._setup_ax(ax)
        
        if Sxx is not None and Sxx.size > 0:
            im = ax.pcolormesh(times, freqs, 10 * np.log10(np.maximum(Sxx, 1e-12)), cmap="viridis", shading='gouraud')
            cbar = self.canvas.fig.colorbar(im, ax=ax)
            cbar.ax.yaxis.set_tick_params(color='#94A3B8', labelcolor='#94A3B8')
            cbar.set_label('Power/Frequency (dB/Hz)', color='#CBD5E1', fontsize=8)
        else:
            ax.text(0.5, 0.5, "No spectrogram data", color="#64748B", ha='center', va='center', transform=ax.transAxes)

        ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
        ax.set_xlabel("Time (s)", fontsize=9)
        ax.set_ylabel("Frequency (Hz)", fontsize=9)
        self.canvas.draw()

    def plot_high_res_comparison(self, fft_freqs, fft_mag, music_freqs, music_psd, esprit_freqs, title="High-Resolution Spectral Comparison (FFT vs MUSIC vs ESPRIT)"):
        self.canvas.fig.clear()
        ax = self.canvas.fig.add_subplot(111)
        self._setup_ax(ax)
        
        if fft_freqs is not None and fft_mag is not None:
            # Normalize FFT
            norm_fft = fft_mag / np.max(fft_mag) if np.max(fft_mag) > 0 else fft_mag
            ax.plot(fft_freqs, norm_fft, color="#64748B", linestyle="--", linewidth=1.0, label="FFT (Conventional)")
            
        if music_freqs is not None and music_psd is not None:
            ax.plot(music_freqs, music_psd, color="#38BDF8", linewidth=1.8, label="MUSIC Pseudospectrum")
            
        if esprit_freqs is not None and len(esprit_freqs) > 0:
            ax.vlines(esprit_freqs, 0, 1.0, color="#F59E0B", linewidth=2.0, label="ESPRIT Estimates")
            ax.scatter(esprit_freqs, np.ones_like(esprit_freqs), color="#F59E0B", s=30, zorder=5)

        ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
        ax.set_xlabel("Frequency (Hz)", fontsize=9)
        ax.set_ylabel("Normalized Amplitude", fontsize=9)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, facecolor='#1E293B', edgecolor='#334155', labelcolor='#F8FAFC', fontsize=8)
        self.canvas.draw()

    def plot_mcsa_spectrum(self, freqs, mag, f_s: float, sidebands: list[float], title="Motor Current Signature Spectrum (MCSA)"):
        self.canvas.fig.clear()
        ax = self.canvas.fig.add_subplot(111)
        self._setup_ax(ax)
        
        if freqs is not None and mag is not None:
            db_mag = 20 * np.log10(np.maximum(mag, 1e-12))
            ax.plot(freqs, db_mag, color="#38BDF8", linewidth=1.2, label="Current Spectrum")
            
            if f_s > 0:
                ax.axvline(f_s, color="#10B981", linestyle="--", linewidth=1.5, label=f"Fundamental ({f_s:.1f} Hz)")
                
            for i, sb in enumerate(sidebands):
                label = "Fault Sidebands" if i == 0 else ""
                ax.axvline(sb, color="#EF4444", linestyle=":", linewidth=1.5, label=label)

        ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
        ax.set_xlabel("Frequency (Hz)", fontsize=9)
        ax.set_ylabel("Amplitude (dB)", fontsize=9)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, facecolor='#1E293B', edgecolor='#334155', labelcolor='#F8FAFC', fontsize=8)
        self.canvas.draw()

    def export_figure(self):
        filename, _ = QFileDialog.getSaveFileName(self, "Export Figure", f"{self.title.replace(' ', '_')}.png", "PNG Image (*.png);;SVG Image (*.svg);;PDF File (*.pdf)")
        if filename:
            self.canvas.fig.savefig(filename, dpi=300, facecolor=self.canvas.fig.get_facecolor(), bbox_inches='tight')
