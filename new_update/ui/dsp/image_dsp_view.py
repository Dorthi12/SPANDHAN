"""
SPANDHAN — Image DSP Scientific Visualization View
===================================================
Renders rich, class-aware scientific 2D image DSP visualizations:
  1. Spatial Image Overview (2D Intensity Map + 1D Center Row Spatial Slice)
  2. 2D Fast Fourier Transform Spectrum (Centered Log Magnitude |F(u,v)| in dB)
  3. 2D Discrete Wavelet Decomposition (2×2 Quadrant: Approx, Horiz, Vert, Diag)
  4. Spatial Image Filtering (Original vs. Filtered vs. Difference Image)
  5. 2D Spatial Convolution (Input, Spatial Kernel, Output)
  6. 2D Image Deconvolution (Degraded, PSF, Restored x̂)

Strictly consumes MATLAB 2D DSP results without local algorithmic reimplementation.
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
from ui.dsp.dsp_result_adapter import ImageDSPResult


class ImageDSPView(QWidget):
    """
    Dynamic container rendering all executed 2D image DSP analyses.
    Inspects ImageDSPResult and only instantiates cards for modules that ran.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.MinimumExpanding)

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(16)

        self._current_result: Optional[ImageDSPResult] = None
        self._cards: List[ScientificVisualizationCard] = []

    def set_result(self, result: ImageDSPResult):
        """Rebuilds the entire visual workspace for the given ImageDSPResult."""
        self._current_result = result
        self._clear_layout()

        # 1. Spatial Image Overview (Always rendered when image data is available)
        self._build_spatial_image_card(result)

        # 2. 2D FFT Spectrum (Rendered if dsp.fft2 exists)
        if result.fft2 is not None and result.fft2.magnitude_db.size > 0:
            self._build_fft2_card(result)

        # 3. 2D Wavelet Decomposition (Rendered if dsp.wavelet2D exists)
        if result.wavelet2d is not None and result.wavelet2d.approximation.size > 0:
            self._build_wavelet2d_card(result)

        # 4. Image Filter (Rendered if dsp.filter exists)
        if result.filter is not None and result.filter.filtered_image.size > 0:
            self._build_filter_card(result)

        # 5. 2D Convolution (Rendered only if executed)
        if result.convolution is not None and result.convolution.output_image.size > 0:
            self._build_convolution_card(result)

        # 6. 2D Deconvolution (Rendered only if executed)
        if result.deconvolution is not None and result.deconvolution.restored_image.size > 0:
            self._build_deconvolution_card(result)

    def _clear_layout(self):
        while self.main_layout.count():
            item = self.main_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._cards.clear()

    # -----------------------------------------------------------------------
    # 1. Spatial Image Overview Card
    # -----------------------------------------------------------------------
    def _build_spatial_image_card(self, result: ImageDSPResult):
        img = result.analysis_image if result.analysis_image.size > 0 else result.raw_image
        if img.size == 0:
            return

        rows, cols = img.shape[:2]
        center_row = rows // 2
        slice_1d = img[center_row, :]

        card = ScientificVisualizationCard(
            title="1. Spatial Domain Image & Center 1D Profile Slice",
            subtitle=f"Preprocessed {result.signal_class} 2D matrix ({cols} × {rows} pixels) with center-row spatial intensity",
            module_tag="Spatial Domain",
            span_full_width=True,
            parent=self,
        )

        plot_widget = DSPScientificPlotWidget(width=8.0, height=3.4, parent=card)

        def render_spatial(p_widget: DSPScientificPlotWidget):
            p_widget.clear()
            fig = p_widget.figure
            ax_img = fig.add_subplot(1, 2, 1)
            ax_slice = fig.add_subplot(1, 2, 2)

            # 2D Image View
            im = ax_img.imshow(img, cmap="viridis", origin="upper", aspect="equal")
            ax_img.axhline(center_row, color=DSP_THEME.ROSE, linestyle="--", linewidth=1.0, alpha=0.8, label=f"Row {center_row}")
            ax_img.set_title(f"Spatial Intensity Map ({cols}×{rows})", fontsize=10, color=DSP_THEME.TEXT_PRIMARY)
            ax_img.set_xlabel("X (Pixels)")
            ax_img.set_ylabel("Y (Pixels)")
            ax_img.legend(loc="lower right", facecolor=DSP_THEME.BG_PLOT, labelcolor=DSP_THEME.TEXT_PRIMARY, fontsize=8)

            cb = fig.colorbar(im, ax=ax_img, orientation="vertical", pad=0.03, aspect=20)
            cb.set_label("Intensity [0.0, 1.0]", color=DSP_THEME.TEXT_SECONDARY, fontsize=8)
            cb.ax.tick_params(colors=DSP_THEME.TEXT_SECONDARY, labelsize=7)

            # 1D Spatial Slice
            ax_slice.plot(np.arange(cols), slice_1d, color=DSP_THEME.CYAN, linewidth=1.2)
            ax_slice.set_title(f"1D Spatial Profile (Slice at Row {center_row})", fontsize=10, color=DSP_THEME.TEXT_PRIMARY)
            ax_slice.set_xlabel("Pixel Column X")
            ax_slice.set_ylabel("Intensity")
            ax_slice.set_xlim(0, cols - 1)
            ax_slice.set_ylim(bottom=min(0.0, float(np.min(slice_1d)) - 0.05), top=max(1.0, float(np.max(slice_1d)) + 0.05))

            p_widget.apply_style()

        render_spatial(plot_widget)
        plot_widget.set_coord_formatter(lambda x, y: f"X: {x:.1f} px  |  Val: {y:.3f}")
        plot_widget.fullscreen_requested.connect(
            lambda _: self._open_fullscreen("Spatial Domain Image & Profile", render_spatial)
        )

        card.set_plot_widget(plot_widget)

        mean_val = float(np.mean(img))
        std_val = float(np.std(img))
        min_val = float(np.min(img))
        max_val = float(np.max(img))
        metrics = [
            ("Dimensions", f"{cols} × {rows}", "px"),
            ("Total Pixels", f"{rows * cols:,}", ""),
            ("Mean Intensity", f"{mean_val:.4f}", ""),
            ("Std Dev", f"{std_val:.4f}", ""),
            ("Dynamic Range", f"[{min_val:.3f}, {max_val:.3f}]", ""),
        ]
        card.set_metrics(metrics)

        self.main_layout.addWidget(card)
        self._cards.append(card)

    # -----------------------------------------------------------------------
    # 2. 2D FFT Spectrum Card
    # -----------------------------------------------------------------------
    def _build_fft2_card(self, result: ImageDSPResult):
        fft2 = result.fft2
        if fft2 is None or fft2.magnitude_db.size == 0:
            return

        card = ScientificVisualizationCard(
            title="2. Two-Dimensional Fourier Transform (2-D FFT) Spatial Spectrum",
            subtitle="Centered log magnitude spectrum |F(u, v)| (dB) showing directional spatial frequencies",
            module_tag="2-D FFT",
            span_full_width=True,
            parent=self,
        )

        plot_widget = DSPScientificPlotWidget(width=8.0, height=3.8, parent=card)

        def render_fft2(p_widget: DSPScientificPlotWidget):
            p_widget.clear()
            fig = p_widget.figure
            ax = fig.add_subplot(1, 1, 1)

            mag_db = fft2.magnitude_db
            r, c = mag_db.shape
            v_min = float(np.percentile(mag_db, 5)) if mag_db.size > 0 else -100.0
            v_max = float(np.max(mag_db)) if mag_db.size > 0 else 0.0

            # Frequency bounds in cycles/pixel (from -0.5 to +0.5)
            extent = [-0.5, 0.5, -0.5, 0.5]

            im = ax.imshow(
                mag_db,
                origin="lower",
                extent=extent,
                cmap="jet",
                aspect="equal",
                vmin=v_min,
                vmax=v_max,
                interpolation="bilinear",
            )

            # Center DC crosshair
            ax.axhline(0, color="white", linestyle=":", linewidth=0.6, alpha=0.5)
            ax.axvline(0, color="white", linestyle=":", linewidth=0.6, alpha=0.5)

            # Mark dominant spatial frequency if available
            if fft2.dominant_spatial_frequency > 0:
                ax.plot(
                    [fft2.dominant_frequency_x],
                    [fft2.dominant_frequency_y],
                    "ro",
                    markersize=6,
                    label=f"Dominant: {fft2.dominant_spatial_frequency:.3f} cyc/px",
                )
                ax.legend(loc="upper right", facecolor=DSP_THEME.BG_PLOT, labelcolor=DSP_THEME.TEXT_PRIMARY, fontsize=8)

            ax.set_title("2-D Spatial Frequency Log Magnitude Spectrum (dB)", fontsize=10, color=DSP_THEME.TEXT_PRIMARY)
            ax.set_xlabel("Spatial Frequency U (cycles/pixel)")
            ax.set_ylabel("Spatial Frequency V (cycles/pixel)")

            cb = fig.colorbar(im, ax=ax, orientation="vertical", pad=0.03, aspect=20)
            cb.set_label("Magnitude (dB)", color=DSP_THEME.TEXT_SECONDARY, fontsize=8)
            cb.ax.tick_params(colors=DSP_THEME.TEXT_SECONDARY, labelsize=7)

            p_widget.apply_style()

        render_fft2(plot_widget)
        plot_widget.set_coord_formatter(lambda u, v: f"U: {u:.3f}  |  V: {v:.3f} cyc/px")
        plot_widget.fullscreen_requested.connect(
            lambda _: self._open_fullscreen("2-D FFT Spatial Frequency Spectrum", render_fft2)
        )

        card.set_plot_widget(plot_widget)

        metrics = [
            ("Dominant Spatial Freq", f"{fft2.dominant_spatial_frequency:.4f}", "cyc/px"),
            ("Centroid (Fx, Fy)", f"({fft2.spectral_centroid_x:.3f}, {fft2.spectral_centroid_y:.3f})", "cyc/px"),
            ("Spectral Bandwidth", f"{fft2.spectral_bandwidth:.4f}", "cyc/px"),
            ("Total Energy", f"{fft2.total_energy:,.1f}", ""),
        ]
        card.set_metrics(metrics)

        self.main_layout.addWidget(card)
        self._cards.append(card)

    # -----------------------------------------------------------------------
    # 3. 2D Wavelet Decomposition Card
    # -----------------------------------------------------------------------
    def _build_wavelet2d_card(self, result: ImageDSPResult):
        w2d = result.wavelet2d
        if w2d is None or w2d.approximation.size == 0:
            return

        level = w2d.decomposition_level
        lvl_idx = level - 1

        card = ScientificVisualizationCard(
            title=f"3. 2-D Discrete Wavelet Decomposition ({w2d.wavelet_name.upper()}, Level {level})",
            subtitle="Multi-scale subband representation: LL (Approximation), LH (Horizontal), HL (Vertical), HH (Diagonal)",
            module_tag="2-D Wavelet",
            span_full_width=True,
            parent=self,
        )

        plot_widget = DSPScientificPlotWidget(width=8.0, height=4.5, parent=card)

        def render_wavelet2d(p_widget: DSPScientificPlotWidget):
            p_widget.clear()
            fig = p_widget.figure

            ax_ll = fig.add_subplot(2, 2, 1)
            ax_lh = fig.add_subplot(2, 2, 2)
            ax_hl = fig.add_subplot(2, 2, 3)
            ax_hh = fig.add_subplot(2, 2, 4)

            # LL: Approximation
            ax_ll.imshow(w2d.approximation, cmap="gray", aspect="equal")
            ax_ll.set_title(f"LL: Approximation (A{level})", fontsize=9, color=DSP_THEME.TEXT_PRIMARY)
            ax_ll.tick_params(labelsize=7)

            # LH: Horizontal details
            h_det = w2d.horizontal_details[lvl_idx] if lvl_idx < len(w2d.horizontal_details) else np.zeros((10, 10))
            ax_lh.imshow(np.abs(h_det), cmap="magma", aspect="equal")
            ax_lh.set_title(f"LH: Horizontal Details (H{level})", fontsize=9, color=DSP_THEME.TEXT_PRIMARY)
            ax_lh.tick_params(labelsize=7)

            # HL: Vertical details
            v_det = w2d.vertical_details[lvl_idx] if lvl_idx < len(w2d.vertical_details) else np.zeros((10, 10))
            ax_hl.imshow(np.abs(v_det), cmap="magma", aspect="equal")
            ax_hl.set_title(f"HL: Vertical Details (V{level})", fontsize=9, color=DSP_THEME.TEXT_PRIMARY)
            ax_hl.tick_params(labelsize=7)

            # HH: Diagonal details
            d_det = w2d.diagonal_details[lvl_idx] if lvl_idx < len(w2d.diagonal_details) else np.zeros((10, 10))
            ax_hh.imshow(np.abs(d_det), cmap="magma", aspect="equal")
            ax_hh.set_title(f"HH: Diagonal Details (D{level})", fontsize=9, color=DSP_THEME.TEXT_PRIMARY)
            ax_hh.tick_params(labelsize=7)

            p_widget.apply_style(grid=False)

        render_wavelet2d(plot_widget)
        plot_widget.fullscreen_requested.connect(
            lambda _: self._open_fullscreen("2-D Wavelet Subband Decomposition", render_wavelet2d)
        )

        card.set_plot_widget(plot_widget)

        metrics = [
            ("Wavelet Name", w2d.wavelet_name.upper(), ""),
            ("Decomp. Level", f"{level}", ""),
            ("Wavelet Entropy", f"{w2d.wavelet_entropy:.3f}", "bits"),
        ]
        if w2d.reconstruction_rmse > 0:
            metrics.append(("Reconstruction RMSE", f"{w2d.reconstruction_rmse:.2e}", ""))

        card.set_metrics(metrics)
        self.main_layout.addWidget(card)
        self._cards.append(card)

    # -----------------------------------------------------------------------
    # 4. Image Filter Card
    # -----------------------------------------------------------------------
    def _build_filter_card(self, result: ImageDSPResult):
        flt = result.filter
        if flt is None or flt.filtered_image.size == 0:
            return

        card = ScientificVisualizationCard(
            title=f"4. Spatial Image Filtering ({flt.filter_type.title()} Filter, {flt.kernel_size[0]}×{flt.kernel_size[1]})",
            subtitle="Original input vs. 2D filtered result and absolute residual difference image",
            module_tag="Image Filter",
            span_full_width=True,
            parent=self,
        )

        plot_widget = DSPScientificPlotWidget(width=8.0, height=3.0, parent=card)

        def render_filter(p_widget: DSPScientificPlotWidget):
            p_widget.clear()
            fig = p_widget.figure

            ax1 = fig.add_subplot(1, 3, 1)
            ax2 = fig.add_subplot(1, 3, 2)
            ax3 = fig.add_subplot(1, 3, 3)

            # Original
            ax1.imshow(flt.input_image, cmap="gray", aspect="equal")
            ax1.set_title("Input Image", fontsize=9, color=DSP_THEME.TEXT_PRIMARY)
            ax1.tick_params(labelsize=7)

            # Filtered
            ax2.imshow(flt.filtered_image, cmap="gray", aspect="equal")
            ax2.set_title(f"Filtered ({flt.filter_type.title()})", fontsize=9, color=DSP_THEME.TEXT_PRIMARY)
            ax2.tick_params(labelsize=7)

            # Difference
            diff = np.abs(flt.difference_image)
            im3 = ax3.imshow(diff, cmap="inferno", aspect="equal")
            ax3.set_title("Absolute Difference", fontsize=9, color=DSP_THEME.TEXT_PRIMARY)
            ax3.tick_params(labelsize=7)

            cb = fig.colorbar(im3, ax=ax3, orientation="vertical", pad=0.03, aspect=20)
            cb.ax.tick_params(colors=DSP_THEME.TEXT_SECONDARY, labelsize=7)

            p_widget.apply_style(grid=False)

        render_filter(plot_widget)
        plot_widget.fullscreen_requested.connect(
            lambda _: self._open_fullscreen("Image Filtering Comparative Inspection", render_filter)
        )

        card.set_plot_widget(plot_widget)

        metrics = [
            ("Filter Type", flt.filter_type.upper(), ""),
            ("Kernel Size", f"{flt.kernel_size[0]}×{flt.kernel_size[1]}", ""),
            ("Output Mean", f"{flt.output_mean:.4f}", ""),
            ("Output Std", f"{flt.output_std:.4f}", ""),
            ("Diff Energy", f"{flt.difference_energy:.3f}", ""),
            ("Compute Time", f"{flt.computation_time * 1000.0:.1f}", "ms"),
        ]
        card.set_metrics(metrics)

        self.main_layout.addWidget(card)
        self._cards.append(card)

    # -----------------------------------------------------------------------
    # 5. 2D Convolution Card
    # -----------------------------------------------------------------------
    def _build_convolution_card(self, result: ImageDSPResult):
        conv = result.convolution
        if conv is None or conv.output_image.size == 0:
            return

        card = ScientificVisualizationCard(
            title="5. 2-D Spatial Convolution",
            subtitle=f"Linear discrete 2D convolution with spatial kernel (Mode: {conv.mode})",
            module_tag="2-D Convolution",
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

            ax1.imshow(conv.input_image, cmap="gray", aspect="equal")
            ax1.set_title("Input Image", fontsize=9, color=DSP_THEME.TEXT_PRIMARY)

            ax2.imshow(conv.kernel, cmap="gray", aspect="equal")
            ax2.set_title("Kernel", fontsize=9, color=DSP_THEME.TEXT_PRIMARY)

            ax3.imshow(conv.output_image, cmap="gray", aspect="equal")
            ax3.set_title("Convolved Output", fontsize=9, color=DSP_THEME.TEXT_PRIMARY)

            p_widget.apply_style(grid=False)

        render_conv(plot_widget)
        card.set_plot_widget(plot_widget)

        card.set_metrics([("Mode", conv.mode.upper(), "")])
        self.main_layout.addWidget(card)
        self._cards.append(card)

    # -----------------------------------------------------------------------
    # 6. 2D Deconvolution Card
    # -----------------------------------------------------------------------
    def _build_deconvolution_card(self, result: ImageDSPResult):
        deconv = result.deconvolution
        if deconv is None or deconv.restored_image.size == 0:
            return

        card = ScientificVisualizationCard(
            title="6. 2-D Image Deconvolution & Restoration",
            subtitle=f"Restoration via {deconv.method} deconvolution with Point Spread Function",
            module_tag="2-D Deconvolution",
            span_full_width=True,
            parent=self,
        )

        plot_widget = DSPScientificPlotWidget(width=8.0, height=3.0, parent=card)

        def render_deconv(p_widget: DSPScientificPlotWidget):
            p_widget.clear()
            fig = p_widget.figure
            ax1 = fig.add_subplot(1, 3, 1)
            ax2 = fig.add_subplot(1, 3, 2)
            ax3 = fig.add_subplot(1, 3, 3)

            ax1.imshow(deconv.input_image, cmap="gray", aspect="equal")
            ax1.set_title("Degraded Input", fontsize=9, color=DSP_THEME.TEXT_PRIMARY)

            ax2.imshow(deconv.psf, cmap="gray", aspect="equal")
            ax2.set_title("PSF Kernel", fontsize=9, color=DSP_THEME.TEXT_PRIMARY)

            ax3.imshow(deconv.restored_image, cmap="gray", aspect="equal")
            ax3.set_title("Restored Image x̂", fontsize=9, color=DSP_THEME.TEXT_PRIMARY)

            p_widget.apply_style(grid=False)

        render_deconv(plot_widget)
        card.set_plot_widget(plot_widget)

        card.set_metrics([
            ("Method", deconv.method.upper(), ""),
            ("NSR", f"{deconv.nsr:.4f}", ""),
        ])
        self.main_layout.addWidget(card)
        self._cards.append(card)

    # -----------------------------------------------------------------------
    # Fullscreen Inspection Dialog
    # -----------------------------------------------------------------------
    def _open_fullscreen(self, title: str, generator_func):
        dlg = FullscreenPlotDialog(title, generator_func, parent=self)
        dlg.exec()
