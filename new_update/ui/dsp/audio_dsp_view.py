"""
SPANDHAN — Audio DSP Scientific Visualization View
===================================================
Renders rich, class-aware scientific audio DSP visualizations:
  1. Signal Overview (Full waveform + Zoomed transient/cycle inset)
  2. FFT Spectrum (Linear + Decibel magnitude, dominant peak marker, -3dB bandwidth)
  3. STFT Spectrogram (Time-frequency map, colormap, overlaid instantaneous frequency)
  4. Wavelet Multi-Resolution Analysis (Approximation A5 + Detail levels D1..D5 + subband energy)
  5. FIR Filter Response (Magnitude in dB, impulse response h[n])
  6. IIR Filter Response (Magnitude in dB, impulse response, stability status)
  7. Linear Convolution (Input, system response h[n], convolution output y[n])
  8. Signal Deconvolution (Observed signal, recovered signal x̂[n])

All visualizations strictly consume MATLAB result fields without Python DSP calculation.
"""

from __future__ import annotations

from typing import Optional, List, Tuple
import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QSizePolicy,
)

from ui.dsp.dsp_theme import DSP_THEME, PLOT_PALETTE, apply_scientific_plot_style
from ui.dsp.dsp_plot_widget import DSPScientificPlotWidget
from ui.dsp.dsp_cards import ScientificVisualizationCard, FullscreenPlotDialog
from ui.dsp.dsp_result_adapter import AudioDSPResult


class AudioDSPView(QWidget):
    """
    Dynamic container rendering all executed audio DSP analyses.
    Inspects AudioDSPResult and only instantiates cards for modules that ran.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.MinimumExpanding)

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(16)

        self._current_result: Optional[AudioDSPResult] = None
        self._cards: List[ScientificVisualizationCard] = []

    def set_result(self, result: AudioDSPResult):
        """Rebuilds the entire visual workspace for the given AudioDSPResult."""
        self._current_result = result
        self._clear_layout()

        # 1. Signal Overview (Always rendered when audio data is available)
        self._build_signal_overview_card(result)

        # 2. FFT Spectrum (Rendered if dsp.fft exists)
        if result.fft is not None and result.fft.magnitude.size > 0:
            self._build_fft_card(result)

        # 3. STFT Spectrogram (Rendered if dsp.stft exists - e.g. Chirp, Step, etc.)
        if result.stft is not None and result.stft.magnitude_db.size > 0:
            self._build_stft_card(result)

        # 4. Wavelet Multi-Resolution (Rendered if dsp.wavelet exists - e.g. Impulse, Step, Chirp)
        if result.wavelet is not None and len(result.wavelet.details) > 0:
            self._build_wavelet_card(result)

        # 5. FIR Filter Response (Rendered if dsp.fir exists - e.g. Sinusoidal, White Noise, Impulse)
        if result.fir is not None and result.fir.magnitude_db.size > 0:
            self._build_fir_card(result)

        # 6. IIR Filter Response (Rendered if dsp.iir exists - e.g. Sinusoidal, White Noise, Step)
        if result.iir is not None and result.iir.magnitude_db.size > 0:
            self._build_iir_card(result)

        # 7. Convolution (Rendered only if actually executed)
        if result.convolution is not None and result.convolution.output_signal.size > 0:
            self._build_convolution_card(result)

        # 8. Deconvolution (Rendered only if actually executed)
        if result.deconvolution is not None and result.deconvolution.reconstructed_signal.size > 0:
            self._build_deconvolution_card(result)

    def _clear_layout(self):
        while self.main_layout.count():
            item = self.main_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._cards.clear()

    # -----------------------------------------------------------------------
    # 1. Signal Overview Card
    # -----------------------------------------------------------------------
    def _build_signal_overview_card(self, result: AudioDSPResult):
        sig = result.analysis_signal if result.analysis_signal.size > 0 else result.raw_signal
        if sig.size == 0:
            return

        fs = result.sampling_frequency if result.sampling_frequency > 0 else 16000.0
        time_axis = np.arange(len(sig)) / fs

        card = ScientificVisualizationCard(
            title="1. Signal Overview & Time-Domain Waveform",
            subtitle=f"Discrete-time representation of {result.signal_class} audio signal @ {fs:,.0f} Hz",
            module_tag="Time-Domain",
            span_full_width=True,
            parent=self,
        )

        plot_widget = DSPScientificPlotWidget(width=8.0, height=2.8, parent=card)

        def render_waveform(p_widget: DSPScientificPlotWidget):
            p_widget.clear()
            fig = p_widget.figure

            # If signal has characteristic transient or cycle, show 2 subplots (Full + Zoomed)
            is_tonal_or_transient = result.signal_class.lower() in ("sinusoidal", "impulse", "step", "chirp")

            if is_tonal_or_transient and len(sig) > 200:
                ax_full = fig.add_subplot(1, 2, 1)
                ax_zoom = fig.add_subplot(1, 2, 2)

                # Full signal
                ax_full.plot(time_axis, sig, color=DSP_THEME.CYAN, linewidth=0.9, alpha=0.9)
                ax_full.set_title("Full Waveform", fontsize=10, color=DSP_THEME.TEXT_PRIMARY)
                ax_full.set_xlabel("Time (s)")
                ax_full.set_ylabel("Amplitude")
                ax_full.set_xlim(0, time_axis[-1])

                # Zoomed segment
                if result.signal_class.lower() == "impulse":
                    # Center on max absolute amplitude
                    peak_idx = int(np.argmax(np.abs(sig)))
                    start_idx = max(0, peak_idx - 100)
                    end_idx = min(len(sig), peak_idx + 100)
                    zoom_title = "Zoomed Impulse Response Peak"
                elif result.signal_class.lower() == "step":
                    # Transition region
                    diffs = np.abs(np.diff(sig))
                    trans_idx = int(np.argmax(diffs)) if diffs.size > 0 else len(sig) // 2
                    start_idx = max(0, trans_idx - 120)
                    end_idx = min(len(sig), trans_idx + 120)
                    zoom_title = "Zoomed Step Transition"
                else:
                    # Representative mid-segment
                    mid = len(sig) // 2
                    start_idx = mid
                    end_idx = min(len(sig), mid + 180)
                    zoom_title = "Zoomed Cycles (Detailed Structure)"

                ax_zoom.plot(
                    time_axis[start_idx:end_idx],
                    sig[start_idx:end_idx],
                    color=DSP_THEME.PERSIAN_GREEN,
                    linewidth=1.2,
                    marker="o",
                    markersize=2.5,
                )
                ax_zoom.set_title(zoom_title, fontsize=10, color=DSP_THEME.TEXT_PRIMARY)
                ax_zoom.set_xlabel("Time (s)")
                ax_zoom.set_ylabel("Amplitude")

                p_widget.apply_style()
            else:
                ax = fig.add_subplot(1, 1, 1)
                ax.plot(time_axis, sig, color=DSP_THEME.CYAN, linewidth=0.9)
                ax.set_title("Full Waveform", fontsize=10, color=DSP_THEME.TEXT_PRIMARY)
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Amplitude")
                ax.set_xlim(0, time_axis[-1])
                p_widget.apply_style()

        render_waveform(plot_widget)
        plot_widget.set_coord_formatter(lambda t, y: f"Time: {t:.4f} s  |  Amp: {y:.4f}")
        plot_widget.fullscreen_requested.connect(
            lambda _: self._open_fullscreen("Signal Waveform Overview", render_waveform)
        )

        card.set_plot_widget(plot_widget)

        # Numerical Metadata
        energy = float(np.sum(sig ** 2))
        rms = float(np.sqrt(np.mean(sig ** 2)))
        peak_amp = float(np.max(np.abs(sig)))
        metrics = [
            ("Sampling Rate", f"{fs:,.0f}", "Hz"),
            ("Duration", f"{result.duration:.2f}", "s"),
            ("Samples", f"{len(sig):,}", "pts"),
            ("Peak Amplitude", f"{peak_amp:.4f}", ""),
            ("RMS Level", f"{rms:.4f}", ""),
            ("Signal Energy", f"{energy:.2f}", ""),
        ]
        card.set_metrics(metrics)

        self.main_layout.addWidget(card)
        self._cards.append(card)

    # -----------------------------------------------------------------------
    # 2. FFT Spectrum Card
    # -----------------------------------------------------------------------
    def _build_fft_card(self, result: AudioDSPResult):
        fft_data = result.fft
        if fft_data is None or fft_data.magnitude.size == 0:
            return

        card = ScientificVisualizationCard(
            title="2. Fast Fourier Transform (FFT) Magnitude Spectrum",
            subtitle="One-sided calibrated amplitude and decibel power distribution computed by MATLAB",
            module_tag="FFT",
            span_full_width=True,
            parent=self,
        )

        plot_widget = DSPScientificPlotWidget(width=8.0, height=3.2, parent=card)

        def render_fft(p_widget: DSPScientificPlotWidget):
            p_widget.clear()
            fig = p_widget.figure
            ax1 = fig.add_subplot(1, 2, 1)
            ax2 = fig.add_subplot(1, 2, 2)

            freq = fft_data.frequency
            mag = fft_data.magnitude
            mag_db = fft_data.magnitude_db

            # Linear Magnitude
            ax1.plot(freq, mag, color=DSP_THEME.CYAN, linewidth=1.0)
            ax1.set_title("Linear Magnitude Spectrum", fontsize=10, color=DSP_THEME.TEXT_PRIMARY)
            ax1.set_xlabel("Frequency (Hz)")
            ax1.set_ylabel("Amplitude")
            ax1.set_xlim(0, max(freq[-1], 100.0))

            # Mark dominant peak if detected
            if fft_data.peak_frequency > 0:
                ax1.axvline(fft_data.peak_frequency, color=DSP_THEME.AMBER, linestyle=":", alpha=0.8)
                ax1.plot([fft_data.peak_frequency], [fft_data.peak_magnitude], "o", color=DSP_THEME.AMBER, markersize=5)
                ax1.text(
                    fft_data.peak_frequency,
                    fft_data.peak_magnitude * 1.05,
                    f" {fft_data.peak_frequency:.0f} Hz",
                    color=DSP_THEME.AMBER,
                    fontsize=8,
                    fontweight="bold",
                )

            # Decibel Magnitude
            ax2.plot(freq, mag_db, color=DSP_THEME.BLUE_LIGHT, linewidth=1.0)
            ax2.set_title("Log Magnitude (dB)", fontsize=10, color=DSP_THEME.TEXT_PRIMARY)
            ax2.set_xlabel("Frequency (Hz)")
            ax2.set_ylabel("Power (dB)")
            ax2.set_xlim(0, max(freq[-1], 100.0))

            p_widget.apply_style()

        render_fft(plot_widget)
        plot_widget.set_coord_formatter(lambda f, y: f"Freq: {f:.1f} Hz  |  Val: {y:.2f}")
        plot_widget.fullscreen_requested.connect(
            lambda _: self._open_fullscreen("FFT Spectrum Analysis", render_fft)
        )

        card.set_plot_widget(plot_widget)

        metrics = [
            ("Peak Frequency", f"{fft_data.peak_frequency:,.1f}", "Hz"),
            ("Peak Magnitude", f"{fft_data.peak_magnitude:.4f}", ""),
            ("Spectral Centroid", f"{fft_data.spectral_centroid:,.1f}", "Hz"),
            ("Spectral Bandwidth", f"{fft_data.spectral_bandwidth:,.1f}", "Hz"),
            ("Flatness", f"{fft_data.spectral_flatness:.4f}", ""),
            ("FFT Resolution (NFFT)", f"{fft_data.nfft}", "pts"),
        ]
        if fft_data.bandwidth_3db > 0:
            metrics.append(("-3dB Bandwidth", f"{fft_data.bandwidth_3db:.1f}", "Hz"))

        card.set_metrics(metrics)
        self.main_layout.addWidget(card)
        self._cards.append(card)

    # -----------------------------------------------------------------------
    # 3. STFT Spectrogram Card
    # -----------------------------------------------------------------------
    def _build_stft_card(self, result: AudioDSPResult):
        stft_data = result.stft
        if stft_data is None or stft_data.magnitude_db.size == 0:
            return

        card = ScientificVisualizationCard(
            title="3. Short-Time Fourier Transform (STFT) Spectrogram",
            subtitle="Time-frequency energy evolution with tracked dominant instantaneous frequency",
            module_tag="STFT",
            span_full_width=True,
            parent=self,
        )

        plot_widget = DSPScientificPlotWidget(width=8.0, height=3.6, parent=card)

        def render_stft(p_widget: DSPScientificPlotWidget):
            p_widget.clear()
            fig = p_widget.figure
            ax = fig.add_subplot(1, 1, 1)

            t = stft_data.time
            f = stft_data.frequency
            s_db = stft_data.magnitude_db

            # Robust extent
            t_min, t_max = (t[0], t[-1]) if t.size > 1 else (0.0, 1.0)
            f_min, f_max = (f[0], f[-1]) if f.size > 1 else (0.0, 8000.0)

            v_min = float(np.percentile(s_db, 5)) if s_db.size > 0 else -100.0
            v_max = float(np.max(s_db)) if s_db.size > 0 else 0.0

            im = ax.imshow(
                s_db,
                origin="lower",
                aspect="auto",
                extent=[t_min, t_max, f_min, f_max],
                cmap="viridis",
                vmin=v_min,
                vmax=v_max,
                interpolation="bilinear",
            )

            # Overlay Instantaneous Frequency if present from MATLAB
            if stft_data.dominant_frequency.size == t.size and t.size > 0:
                ax.plot(
                    t,
                    stft_data.dominant_frequency,
                    color=DSP_THEME.AMBER,
                    linewidth=1.5,
                    linestyle="--",
                    label="Tracked Instantaneous Frequency",
                )
                ax.legend(loc="upper right", facecolor=DSP_THEME.BG_PLOT, edgecolor=DSP_THEME.BORDER_SOLID, labelcolor=DSP_THEME.TEXT_PRIMARY)

            ax.set_title("Time-Frequency Spectral Energy Distribution", fontsize=10, color=DSP_THEME.TEXT_PRIMARY)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Frequency (Hz)")

            # Colorbar
            cb = fig.colorbar(im, ax=ax, orientation="vertical", pad=0.02, aspect=20)
            cb.set_label("Magnitude (dB)", color=DSP_THEME.TEXT_SECONDARY, fontsize=9)
            cb.ax.tick_params(colors=DSP_THEME.TEXT_SECONDARY, labelsize=8)

            p_widget.apply_style()

        render_stft(plot_widget)
        plot_widget.set_coord_formatter(lambda t, f: f"Time: {t:.3f} s  |  Freq: {f:.1f} Hz")
        plot_widget.fullscreen_requested.connect(
            lambda _: self._open_fullscreen("STFT Spectrogram Analysis", render_stft)
        )

        card.set_plot_widget(plot_widget)

        metrics = [
            ("Window", f"{stft_data.window_length}", "samples"),
            ("Overlap", f"{stft_data.overlap}", "samples"),
            ("Hop Size", f"{stft_data.hop_size}", "samples"),
            ("Time Resolution", f"{stft_data.time_resolution * 1000.0:.1f}", "ms"),
            ("Freq Resolution", f"{stft_data.frequency_resolution:.2f}", "Hz/bin"),
            ("Total Energy", f"{stft_data.total_energy:,.1f}", ""),
        ]
        card.set_metrics(metrics)
        self.main_layout.addWidget(card)
        self._cards.append(card)

    # -----------------------------------------------------------------------
    # 4. Wavelet Analysis Card
    # -----------------------------------------------------------------------
    def _build_wavelet_card(self, result: AudioDSPResult):
        w_data = result.wavelet
        if w_data is None or len(w_data.details) == 0:
            return

        card = ScientificVisualizationCard(
            title=f"4. Discrete Wavelet Multi-Resolution Analysis ({w_data.wavelet_name.upper()}, Level {w_data.decomposition_level})",
            subtitle="Dyadic subband decomposition into coarse approximation and multi-scale detail coefficients",
            module_tag="Wavelet",
            span_full_width=True,
            parent=self,
        )

        plot_widget = DSPScientificPlotWidget(width=8.0, height=4.2, parent=card)

        def render_wavelet(p_widget: DSPScientificPlotWidget):
            p_widget.clear()
            fig = p_widget.figure

            level = w_data.decomposition_level
            n_plots = min(level + 1, 6)

            # Left column: Stacked Wavelet signals (Approximation + Details)
            # Right column: Relative subband energy distribution bar chart
            gs = fig.add_gridspec(n_plots, 2, width_ratios=[2.5, 1.0], hspace=0.35, wspace=0.25)

            # Plot Approximation on top left
            ax_a = fig.add_subplot(gs[0, 0])
            approx_sig = (
                w_data.reconstructed_approximation
                if w_data.reconstructed_approximation is not None and w_data.reconstructed_approximation.size > 0
                else w_data.approximation
            )
            if approx_sig is not None and approx_sig.size > 0:
                ax_a.plot(approx_sig, color=DSP_THEME.CYAN, linewidth=0.8)
            ax_a.set_ylabel(f"A{level}", fontsize=8, color=DSP_THEME.TEXT_SECONDARY)
            ax_a.tick_params(labelbottom=False)

            # Plot Details D1..D_level
            for idx in range(n_plots - 1):
                ax_d = fig.add_subplot(gs[idx + 1, 0])
                if idx < len(w_data.reconstructed_details) and w_data.reconstructed_details[idx].size > 0:
                    d_sig = w_data.reconstructed_details[idx]
                elif idx < len(w_data.details) and w_data.details[idx].size > 0:
                    d_sig = w_data.details[idx]
                else:
                    d_sig = np.array([])

                if d_sig.size > 0:
                    ax_d.plot(d_sig, color=PLOT_PALETTE[(idx + 1) % len(PLOT_PALETTE)], linewidth=0.8)
                ax_d.set_ylabel(f"D{idx + 1}", fontsize=8, color=DSP_THEME.TEXT_SECONDARY)
                if idx < n_plots - 2:
                    ax_d.tick_params(labelbottom=False)
                else:
                    ax_d.set_xlabel("Sample Index", fontsize=8)

            # Right Column: Subband Energy Distribution
            ax_energy = fig.add_subplot(gs[:, 1])
            if w_data.detail_energy.size > 0:
                levels = [f"D{k+1}" for k in range(len(w_data.detail_energy))]
                energies = w_data.detail_energy
                bars = ax_energy.barh(levels, energies, color=DSP_THEME.INDIGO_LINE, edgecolor=DSP_THEME.BORDER_SOLID, height=0.6)

                # Highlight dominant level
                if 1 <= w_data.dominant_level <= len(bars):
                    bars[w_data.dominant_level - 1].set_color(DSP_THEME.PERSIAN_GREEN)

                ax_energy.set_title("Subband Detail Energy", fontsize=9, color=DSP_THEME.TEXT_PRIMARY)
                ax_energy.set_xlabel("Energy (Σ d_k²)", fontsize=8)
                ax_energy.invert_yaxis()

            p_widget.apply_style()

        render_wavelet(plot_widget)
        plot_widget.fullscreen_requested.connect(
            lambda _: self._open_fullscreen("Wavelet Multi-Resolution Analysis", render_wavelet)
        )

        card.set_plot_widget(plot_widget)

        metrics = [
            ("Wavelet Family", w_data.wavelet_name.upper(), ""),
            ("Decomp. Level", f"{w_data.decomposition_level}", ""),
            ("Dominant Level", f"D{w_data.dominant_level}", ""),
            ("Wavelet Entropy", f"{w_data.wavelet_entropy:.3f}", "bits"),
        ]
        if w_data.reconstruction_rmse > 0:
            metrics.append(("Reconstruction RMSE", f"{w_data.reconstruction_rmse:.2e}", ""))

        card.set_metrics(metrics)
        self.main_layout.addWidget(card)
        self._cards.append(card)

    # -----------------------------------------------------------------------
    # 5. FIR Filter Response Card
    # -----------------------------------------------------------------------
    def _build_fir_card(self, result: AudioDSPResult):
        fir = result.fir
        if fir is None or fir.magnitude_db.size == 0:
            return

        card = ScientificVisualizationCard(
            title=f"5. FIR Digital Filter Response (Order {fir.order}, {fir.filter_type.title()}-pass)",
            subtitle=f"Designed via {fir.window.title()} window with {fir.cutoff_frequency:,.0f} Hz cutoff",
            module_tag="FIR Filter",
            span_full_width=True,
            parent=self,
        )

        plot_widget = DSPScientificPlotWidget(width=8.0, height=3.0, parent=card)

        def render_fir(p_widget: DSPScientificPlotWidget):
            p_widget.clear()
            fig = p_widget.figure
            ax1 = fig.add_subplot(1, 2, 1)
            ax2 = fig.add_subplot(1, 2, 2)

            # Frequency response
            ax1.plot(fir.frequency, fir.magnitude_db, color=DSP_THEME.CYAN, linewidth=1.1)
            ax1.axvline(fir.cutoff_frequency, color=DSP_THEME.AMBER, linestyle="--", alpha=0.8, label="Cutoff")
            ax1.set_title("FIR Frequency Response |H(f)|", fontsize=10, color=DSP_THEME.TEXT_PRIMARY)
            ax1.set_xlabel("Frequency (Hz)")
            ax1.set_ylabel("Magnitude (dB)")
            ax1.set_ylim(bottom=max(float(np.min(fir.magnitude_db)), -120.0), top=5.0)
            ax1.legend(loc="upper right", facecolor=DSP_THEME.BG_PLOT, labelcolor=DSP_THEME.TEXT_PRIMARY)

            # Impulse response
            t_imp = fir.impulse_time if fir.impulse_time.size == fir.impulse_response.size else np.arange(len(fir.impulse_response))
            ax2.plot(t_imp, fir.impulse_response, color=DSP_THEME.PERSIAN_GREEN, linewidth=1.0)
            ax2.set_title("FIR Impulse Response h[n]", fontsize=10, color=DSP_THEME.TEXT_PRIMARY)
            ax2.set_xlabel("Time (s)" if fir.impulse_time.size > 0 else "Sample")
            ax2.set_ylabel("Amplitude")

            p_widget.apply_style()

        render_fir(plot_widget)
        plot_widget.set_coord_formatter(lambda f, y: f"Freq: {f:.1f} Hz  |  Gain: {y:.2f} dB")
        plot_widget.fullscreen_requested.connect(
            lambda _: self._open_fullscreen("FIR Filter Response", render_fir)
        )

        card.set_plot_widget(plot_widget)

        metrics = [
            ("Filter Order", f"{fir.order}", "taps"),
            ("Filter Type", fir.filter_type.upper(), ""),
            ("Window", fir.window.upper(), ""),
            ("Design Cutoff", f"{fir.cutoff_frequency:,.0f}", "Hz"),
            ("-3dB Cutoff", f"{fir.estimated_3db_cutoff:,.0f}", "Hz"),
        ]
        card.set_metrics(metrics)
        self.main_layout.addWidget(card)
        self._cards.append(card)

    # -----------------------------------------------------------------------
    # 6. IIR Filter Response Card
    # -----------------------------------------------------------------------
    def _build_iir_card(self, result: AudioDSPResult):
        iir = result.iir
        if iir is None or iir.magnitude_db.size == 0:
            return

        card = ScientificVisualizationCard(
            title=f"6. IIR Digital Filter Response ({iir.filter_family.title()}, Order {iir.order})",
            subtitle=f"Infinite impulse response with {iir.passband_frequency:,.0f} Hz passband edge",
            module_tag="IIR Filter",
            span_full_width=True,
            parent=self,
        )

        plot_widget = DSPScientificPlotWidget(width=8.0, height=3.0, parent=card)

        def render_iir(p_widget: DSPScientificPlotWidget):
            p_widget.clear()
            fig = p_widget.figure
            ax1 = fig.add_subplot(1, 2, 1)
            ax2 = fig.add_subplot(1, 2, 2)

            # Frequency response
            ax1.plot(iir.frequency, iir.magnitude_db, color=DSP_THEME.BLUE_LIGHT, linewidth=1.1)
            ax1.axvline(iir.passband_frequency, color=DSP_THEME.AMBER, linestyle="--", alpha=0.8, label="Passband")
            ax1.axvline(iir.stopband_frequency, color=DSP_THEME.ROSE, linestyle=":", alpha=0.8, label="Stopband")
            ax1.set_title("IIR Frequency Response |H(f)|", fontsize=10, color=DSP_THEME.TEXT_PRIMARY)
            ax1.set_xlabel("Frequency (Hz)")
            ax1.set_ylabel("Magnitude (dB)")
            ax1.set_ylim(bottom=max(float(np.min(iir.magnitude_db)), -120.0), top=5.0)
            ax1.legend(loc="upper right", facecolor=DSP_THEME.BG_PLOT, labelcolor=DSP_THEME.TEXT_PRIMARY)

            # Impulse response
            t_imp = iir.impulse_time if iir.impulse_time.size == iir.impulse_response.size else np.arange(len(iir.impulse_response))
            ax2.plot(t_imp, iir.impulse_response, color=DSP_THEME.INDIGO_LINE, linewidth=1.0)
            ax2.set_title("IIR Impulse Response h[n]", fontsize=10, color=DSP_THEME.TEXT_PRIMARY)
            ax2.set_xlabel("Time (s)" if iir.impulse_time.size > 0 else "Sample")
            ax2.set_ylabel("Amplitude")

            p_widget.apply_style()

        render_iir(plot_widget)
        plot_widget.set_coord_formatter(lambda f, y: f"Freq: {f:.1f} Hz  |  Gain: {y:.2f} dB")
        plot_widget.fullscreen_requested.connect(
            lambda _: self._open_fullscreen("IIR Filter Response", render_iir)
        )

        card.set_plot_widget(plot_widget)

        stability_str = "STABLE (Inside unit circle)" if iir.stable else "UNSTABLE"
        metrics = [
            ("Family", iir.filter_family.upper(), ""),
            ("Order", f"{iir.order}", ""),
            ("Passband Edge", f"{iir.passband_frequency:,.0f}", "Hz"),
            ("Stopband Edge", f"{iir.stopband_frequency:,.0f}", "Hz"),
            ("Stability", stability_str, ""),
        ]
        card.set_metrics(metrics)
        self.main_layout.addWidget(card)
        self._cards.append(card)

    # -----------------------------------------------------------------------
    # 7. Convolution Card
    # -----------------------------------------------------------------------
    def _build_convolution_card(self, result: AudioDSPResult):
        conv = result.convolution
        if conv is None or conv.output_signal.size == 0:
            return

        card = ScientificVisualizationCard(
            title="7. Discrete LTI System Convolution",
            subtitle="Linear convolution y[n] = x[n] ∗ h[n] with supplied impulse response",
            module_tag="Convolution",
            span_full_width=True,
            parent=self,
        )

        plot_widget = DSPScientificPlotWidget(width=8.0, height=3.0, parent=card)

        def render_conv(p_widget: DSPScientificPlotWidget):
            p_widget.clear()
            fig = p_widget.figure
            ax1 = fig.add_subplot(1, 3, 1)
            ax2 = fig.add_subplot(1, 3, 2)
            ax3 = fig.add_subplot(1, 3, 3)

            ax1.plot(conv.input_signal, color=DSP_THEME.CYAN, linewidth=0.9)
            ax1.set_title("Input Signal x[n]", fontsize=9, color=DSP_THEME.TEXT_PRIMARY)
            ax1.set_xlabel("Sample")

            ax2.plot(conv.impulse_response, color=DSP_THEME.AMBER, linewidth=0.9)
            ax2.set_title("System Impulse Response h[n]", fontsize=9, color=DSP_THEME.TEXT_PRIMARY)
            ax2.set_xlabel("Sample")

            ax3.plot(conv.output_signal, color=DSP_THEME.PERSIAN_GREEN, linewidth=0.9)
            ax3.set_title("Convolution Output y[n]", fontsize=9, color=DSP_THEME.TEXT_PRIMARY)
            ax3.set_xlabel("Sample")

            p_widget.apply_style()

        render_conv(plot_widget)
        card.set_plot_widget(plot_widget)

        metrics = [
            ("Method", conv.method.upper(), ""),
            ("Mode", conv.output_mode.upper(), ""),
            ("Input Energy", f"{conv.input_energy:.2f}", ""),
            ("Output Energy", f"{conv.output_energy:.2f}", ""),
        ]
        card.set_metrics(metrics)
        self.main_layout.addWidget(card)
        self._cards.append(card)

    # -----------------------------------------------------------------------
    # 8. Deconvolution Card
    # -----------------------------------------------------------------------
    def _build_deconvolution_card(self, result: AudioDSPResult):
        deconv = result.deconvolution
        if deconv is None or deconv.reconstructed_signal.size == 0:
            return

        card = ScientificVisualizationCard(
            title="8. Signal Deconvolution & Inverse Filtering",
            subtitle=f"Signal reconstruction via {deconv.method} deconvolution",
            module_tag="Deconvolution",
            span_full_width=True,
            parent=self,
        )

        plot_widget = DSPScientificPlotWidget(width=8.0, height=3.0, parent=card)

        def render_deconv(p_widget: DSPScientificPlotWidget):
            p_widget.clear()
            fig = p_widget.figure
            ax1 = fig.add_subplot(1, 2, 1)
            ax2 = fig.add_subplot(1, 2, 2)

            ax1.plot(deconv.input_signal, color=DSP_THEME.ROSE, linewidth=0.9)
            ax1.set_title("Observed Degradation y[n]", fontsize=9, color=DSP_THEME.TEXT_PRIMARY)
            ax1.set_xlabel("Sample")

            ax2.plot(deconv.reconstructed_signal, color=DSP_THEME.PERSIAN_GREEN, linewidth=0.9)
            ax2.set_title("Recovered Source x̂[n]", fontsize=9, color=DSP_THEME.TEXT_PRIMARY)
            ax2.set_xlabel("Sample")

            p_widget.apply_style()

        render_deconv(plot_widget)
        card.set_plot_widget(plot_widget)

        metrics = [
            ("Method", deconv.method.upper(), ""),
            ("Lambda Regularizer", f"{deconv.lambda_param:.4f}", ""),
        ]
        card.set_metrics(metrics)
        self.main_layout.addWidget(card)
        self._cards.append(card)

    # -----------------------------------------------------------------------
    # Fullscreen Inspection Dialog
    # -----------------------------------------------------------------------
    def _open_fullscreen(self, title: str, generator_func):
        dlg = FullscreenPlotDialog(title, generator_func, parent=self)
        dlg.exec()
