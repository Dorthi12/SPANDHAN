"""
app/pages/audio_experiment_page.py
====================================
Audio Synthetic Experiment Page — Milestone D frontend.

Renders the full AudioExperimentResult from domains.audio.experiment.
ALL computation runs in a background Worker thread.
The UI thread only updates widgets from the result.

Layout (scrollable)
-------------------
┌─ Header ─────────────────────────────────────────────────────────┐
│  AUDIO SYNTHETIC EXPERIMENT PIPELINE                             │
├─ Config Panel ───────────────────────────────────────────────────┤
│  Signal Type │ Noise Type │ SNR target │ Denoise Method │ [Run]  │
├─ Status Bar ─────────────────────────────────────────────────────┤
│  Pipeline status badge │ Measured SNR │ SNR ↑ │ RMSE ↓ │ Time   │
├─ Waveform Comparison (3-panel) ──────────────────────────────────┤
│  Clean │ Noisy │ Cleaned                                         │
├─ Spectral Analysis ──────────────────────────────────────────────┤
│  PSD Overlay (clean vs noisy vs cleaned)                         │
├─ Noise Classification ───────────────────────────────────────────┤
│  Ground Truth │ ML Prediction │ Probability Bars                 │
├─ Quality Metrics Table ──────────────────────────────────────────┤
│  SNR before/after │ RMSE before/after │ Corr │ Spectral Dist    │
└──────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFrame, QLabel,
    QPushButton, QComboBox, QDoubleSpinBox, QSpinBox,
    QScrollArea, QProgressBar, QTableWidget, QTableWidgetItem,
    QHeaderView, QSizePolicy,
)
from PySide6.QtCore import Qt, QThreadPool, Signal

from app.widgets.section_header import SectionHeader
from app.widgets.metric_card import MetricCard
from app.widgets.status_badge import StatusBadge
from app.widgets.signal_plot import SignalPlotWidget
from core.worker import Worker

from domains.audio.experiment import (
    AudioExperimentConfig,
    AudioExperimentResult,
    run_audio_experiment,
)
from domains.audio.noise_pipeline import AUDIO_NOISE_TYPES
from preprocessing.audio_denoising import SUPPORTED_METHODS


# Colour palette — consistent with the app theme
_CLR_CLEAN   = "#38BDF8"   # sky blue
_CLR_NOISY   = "#EF4444"   # red
_CLR_CLEANED = "#10B981"   # emerald
_CLR_ACCENT  = "#F59E0B"   # amber


class AudioExperimentPage(QWidget):
    """
    Full end-to-end audio experiment frontend.

    Receives AudioExperimentResult from the background worker and
    renders every panel without touching any computation logic.
    """

    experiment_completed = Signal(object)   # emits AudioExperimentResult

    # ----------------------------------------------------------------
    # Construction
    # ----------------------------------------------------------------

    def __init__(self, parent=None):
        super().__init__(parent)
        self._last_result: AudioExperimentResult | None = None
        self._threadpool = QThreadPool()

        # Outer scroll wrapper
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        container = QWidget()
        scroll.setWidget(container)

        self._main = QVBoxLayout(container)
        self._main.setContentsMargins(16, 16, 16, 16)
        self._main.setSpacing(14)

        self._build_header()
        self._build_config_panel()
        self._build_status_bar()
        self._build_waveform_panel()
        self._build_spectral_panel()
        self._build_classification_panel()
        self._build_metrics_table()

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(scroll)

    # ----------------------------------------------------------------
    # Panel builders
    # ----------------------------------------------------------------

    def _build_header(self):
        self._main.addWidget(SectionHeader(
            "Audio Synthetic Experiment Pipeline",
            "End-to-end: generate → inject noise → analyse → denoise → compare. "
            "All results come from actual DSP computation."
        ))

    def _build_config_panel(self):
        card = QFrame()
        card.setObjectName("CardPanel")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(10)

        title = QLabel("EXPERIMENT CONFIGURATION")
        title.setStyleSheet("font-size: 12px; font-weight: 700; color: #38BDF8;")
        layout.addWidget(title)

        row1 = QHBoxLayout()
        row2 = QHBoxLayout()
        row1.setSpacing(12)
        row2.setSpacing(12)

        # Signal type
        self._sig_type = self._make_combo(
            ["clean_sine", "multi_tone", "harmonic", "chirp",
             "amplitude_modulated", "frequency_modulated"],
            label="Signal Type"
        )
        row1.addLayout(self._sig_type["box"])

        # Noise type
        self._noise_type = self._make_combo(
            list(AUDIO_NOISE_TYPES), label="Noise Type"
        )
        row1.addLayout(self._noise_type["box"])

        # Target SNR
        self._target_snr = self._make_dspin(
            label="Target SNR (dB)", lo=-10.0, hi=40.0, val=10.0, step=1.0
        )
        row1.addLayout(self._target_snr["box"])

        # Seed
        self._seed = self._make_spin(
            label="Seed", lo=0, hi=99999, val=42
        )
        row1.addLayout(self._seed["box"])

        row1.addStretch()

        # Denoising method
        denoise_options = ["auto"] + [m for m in SUPPORTED_METHODS if m != "wavelet"]
        self._denoise_method = self._make_combo(
            denoise_options, label="Denoising Method"
        )
        row2.addLayout(self._denoise_method["box"])

        # Duration
        self._duration = self._make_dspin(
            label="Duration (s)", lo=0.1, hi=5.0, val=1.0, step=0.1
        )
        row2.addLayout(self._duration["box"])

        # Sampling rate
        self._sr = self._make_dspin(
            label="Sample Rate (Hz)", lo=4000.0, hi=48000.0, val=8000.0, step=1000.0
        )
        row2.addLayout(self._sr["box"])

        # Fundamental frequency
        self._freq = self._make_dspin(
            label="Frequency (Hz)", lo=20.0, hi=4000.0, val=440.0, step=10.0
        )
        row2.addLayout(self._freq["box"])

        row2.addStretch()

        # Run button
        self._run_btn = QPushButton("▶  Run Experiment")
        self._run_btn.setObjectName("PrimaryButton")
        self._run_btn.setFixedHeight(36)
        self._run_btn.clicked.connect(self._run_experiment)

        layout.addLayout(row1)
        layout.addLayout(row2)
        layout.addWidget(self._run_btn, alignment=Qt.AlignmentFlag.AlignRight)

        self._main.addWidget(card)

    def _build_status_bar(self):
        card = QFrame()
        card.setObjectName("CardPanel")
        layout = QHBoxLayout(card)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(16)

        status_box = QVBoxLayout()
        status_box.setSpacing(2)
        lbl = QLabel("PIPELINE STATUS")
        lbl.setStyleSheet("font-size: 10px; font-weight: 700; color: #94A3B8; letter-spacing: 0.6px;")
        self._status_badge = StatusBadge("Ready")
        status_box.addWidget(lbl)
        status_box.addWidget(self._status_badge)
        layout.addLayout(status_box)

        layout.addWidget(self._vline())

        self._msnr_card  = MetricCard("Measured SNR", "—", "dB", "Actual noise level", _CLR_NOISY)
        self._snri_card  = MetricCard("SNR Improvement", "—", "dB", "After denoising", _CLR_CLEANED)
        self._rmse_card  = MetricCard("RMSE Reduction", "—", "", "vs clean reference", _CLR_ACCENT)
        self._corr_card  = MetricCard("Correlation (cleaned)", "—", "", "vs clean", _CLR_CLEAN)
        self._time_card  = MetricCard("Pipeline Time", "—", "s", "Total wall-clock", "#A78BFA")

        for c in [self._msnr_card, self._snri_card, self._rmse_card,
                  self._corr_card, self._time_card]:
            layout.addWidget(c)

        layout.addStretch()
        self._main.addWidget(card)

    def _build_waveform_panel(self):
        card = QFrame()
        card.setObjectName("CardPanel")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)

        title = QLabel("WAVEFORM COMPARISON  —  Clean · Noisy · Cleaned")
        title.setStyleSheet("font-size: 12px; font-weight: 700; color: #F8FAFC;")
        layout.addWidget(title)

        plots_row = QHBoxLayout()
        plots_row.setSpacing(8)

        self._clean_plot   = SignalPlotWidget("Clean Signal")
        self._noisy_plot   = SignalPlotWidget("Noisy Signal")
        self._cleaned_plot = SignalPlotWidget("Cleaned Signal")

        for plot in [self._clean_plot, self._noisy_plot, self._cleaned_plot]:
            plot.setMinimumHeight(200)
            plots_row.addWidget(plot)

        layout.addLayout(plots_row)
        self._main.addWidget(card)

    def _build_spectral_panel(self):
        card = QFrame()
        card.setObjectName("CardPanel")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)

        title = QLabel("PSD OVERLAY  —  Clean (blue) · Noisy (red) · Cleaned (green)")
        title.setStyleSheet("font-size: 12px; font-weight: 700; color: #F8FAFC;")
        layout.addWidget(title)

        self._psd_plot = SignalPlotWidget("Power Spectral Density Comparison")
        self._psd_plot.setMinimumHeight(220)
        layout.addWidget(self._psd_plot)

        self._main.addWidget(card)

    def _build_classification_panel(self):
        card = QFrame()
        card.setObjectName("CardPanel")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)

        header_row = QHBoxLayout()
        title = QLabel("NOISE CLASSIFICATION")
        title.setStyleSheet("font-size: 12px; font-weight: 700; color: #F8FAFC;")

        self._gt_badge   = StatusBadge("Ground Truth: —")
        self._pred_badge = StatusBadge("Predicted: —")
        self._conf_label = QLabel("Confidence: —")
        self._conf_label.setStyleSheet("font-size: 11px; color: #94A3B8;")

        header_row.addWidget(title)
        header_row.addStretch()
        header_row.addWidget(self._gt_badge)
        header_row.addWidget(self._pred_badge)
        header_row.addWidget(self._conf_label)
        layout.addLayout(header_row)

        # Probability bars
        self._prob_bars: dict[str, QProgressBar] = {}
        bars_frame = QFrame()
        bars_frame.setStyleSheet(
            "background-color: #0F172A; border-radius: 6px; padding: 8px;"
        )
        bars_layout = QVBoxLayout(bars_frame)
        bars_layout.setSpacing(4)

        classes = ["Clean", "Gaussian", "Impulse", "Periodic", "Colored", "Mixed"]
        for cls in classes:
            row = QHBoxLayout()
            lbl = QLabel(f"{cls:<10}")
            lbl.setStyleSheet(
                "font-size: 11px; font-weight: 600; color: #E2E8F0; font-family: monospace;"
            )
            lbl.setFixedWidth(80)
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setFixedHeight(14)
            bar.setFormat("%v%")
            row.addWidget(lbl)
            row.addWidget(bar)
            bars_layout.addLayout(row)
            self._prob_bars[cls] = bar

        layout.addWidget(bars_frame)
        self._main.addWidget(card)

    def _build_metrics_table(self):
        card = QFrame()
        card.setObjectName("CardPanel")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)

        title = QLabel("BEFORE / AFTER QUALITY METRICS")
        title.setStyleSheet("font-size: 12px; font-weight: 700; color: #F8FAFC;")
        layout.addWidget(title)

        self._metrics_table = QTableWidget(5, 4)
        self._metrics_table.setHorizontalHeaderLabels(
            ["Metric", "Before (noisy)", "After (cleaned)", "Improvement"]
        )
        hdr = self._metrics_table.horizontalHeader()
        hdr.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._metrics_table.verticalHeader().setVisible(False)
        self._metrics_table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers
        )

        rows = ["SNR (dB)", "RMSE", "Correlation", "Energy Error", "Spectral Distortion"]
        for i, r in enumerate(rows):
            self._metrics_table.setItem(i, 0, QTableWidgetItem(r))
            for j in range(1, 4):
                self._metrics_table.setItem(i, j, QTableWidgetItem("—"))

        layout.addWidget(self._metrics_table)
        self._main.addWidget(card)

    # ----------------------------------------------------------------
    # Helper widget factories
    # ----------------------------------------------------------------

    @staticmethod
    def _make_combo(options: list[str], label: str) -> dict:
        box = QVBoxLayout()
        box.setSpacing(3)
        lbl = QLabel(label)
        lbl.setStyleSheet("font-size: 10px; color: #94A3B8; font-weight: 600;")
        combo = QComboBox()
        for o in options:
            combo.addItem(o)
        box.addWidget(lbl)
        box.addWidget(combo)
        return {"box": box, "widget": combo}

    @staticmethod
    def _make_dspin(label: str, lo: float, hi: float, val: float, step: float) -> dict:
        box = QVBoxLayout()
        box.setSpacing(3)
        lbl = QLabel(label)
        lbl.setStyleSheet("font-size: 10px; color: #94A3B8; font-weight: 600;")
        spin = QDoubleSpinBox()
        spin.setRange(lo, hi)
        spin.setValue(val)
        spin.setSingleStep(step)
        spin.setDecimals(1)
        box.addWidget(lbl)
        box.addWidget(spin)
        return {"box": box, "widget": spin}

    @staticmethod
    def _make_spin(label: str, lo: int, hi: int, val: int) -> dict:
        box = QVBoxLayout()
        box.setSpacing(3)
        lbl = QLabel(label)
        lbl.setStyleSheet("font-size: 10px; color: #94A3B8; font-weight: 600;")
        spin = QSpinBox()
        spin.setRange(lo, hi)
        spin.setValue(val)
        box.addWidget(lbl)
        box.addWidget(spin)
        return {"box": box, "widget": spin}

    @staticmethod
    def _vline() -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.Shape.VLine)
        line.setStyleSheet("color: #334155;")
        return line

    # ----------------------------------------------------------------
    # Experiment execution
    # ----------------------------------------------------------------

    def _read_config(self) -> AudioExperimentConfig:
        return AudioExperimentConfig(
            seed=self._seed["widget"].value(),
            signal_type=self._sig_type["widget"].currentText(),
            sampling_rate=self._sr["widget"].value(),
            duration=self._duration["widget"].value(),
            frequency=self._freq["widget"].value(),
            noise_type=self._noise_type["widget"].currentText(),
            target_snr_db=self._target_snr["widget"].value(),
            denoise_method=self._denoise_method["widget"].currentText(),
        )

    def _run_experiment(self):
        cfg = self._read_config()
        self._run_btn.setEnabled(False)
        self._status_badge.set_status("Warning")  # "Running…"

        def _task():
            return run_audio_experiment(cfg)

        worker = Worker(_task)
        worker.signals.result.connect(self._on_result)
        worker.signals.error.connect(self._on_error)
        worker.signals.finished.connect(lambda: self._run_btn.setEnabled(True))
        self._threadpool.start(worker)

    def _on_error(self, error_tuple):
        _exctype, value, _tb = error_tuple
        self._status_badge.set_status("Error")
        self._msnr_card.update_metric(f"Error: {value}", "")

    def _on_result(self, result: AudioExperimentResult):
        self._last_result = result
        self._render(result)
        self.experiment_completed.emit(result)

    # ----------------------------------------------------------------
    # Rendering — no computation, pure widget updates
    # ----------------------------------------------------------------

    def _render(self, r: AudioExperimentResult):
        # Status
        if r.errors:
            self._status_badge.set_status("Error")
        elif r.warnings:
            self._status_badge.set_status("Warning")
        else:
            self._status_badge.set_status("Complete")

        # Status bar cards
        if r.injection is not None:
            snr = r.injection.measured_snr_db
            self._msnr_card.update_metric(
                f"{snr:.2f}" if np.isfinite(snr) else "∞", "dB"
            )

        if r.denoising is not None:
            snri = r.denoising.snr_improvement_db
            rmse = r.denoising.rmse_reduction
            self._snri_card.update_metric(
                f"{snri:+.2f}" if snri is not None else "—", "dB"
            )
            self._rmse_card.update_metric(
                f"{rmse:.4f}" if rmse is not None else "—", ""
            )

        if r.cleaned_analysis is not None and r.quality_comparison:
            corr = r.quality_comparison.get("after", {}).get("correlation", None)
            if corr is not None:
                self._corr_card.update_metric(f"{corr:.4f}", "")

        self._time_card.update_metric(f"{r.total_time_s:.3f}", "s")

        # Waveforms
        t = r.time_axis
        self._clean_plot.plot_time_domain(
            t, r.clean_signal, "Clean Signal", color=_CLR_CLEAN
        )
        self._noisy_plot.plot_time_domain(
            t, r.noisy_signal, "Noisy Signal", color=_CLR_NOISY
        )
        self._cleaned_plot.plot_time_domain(
            t, r.cleaned_signal, "Cleaned Signal", color=_CLR_CLEANED
        )

        # PSD overlay (custom — not in SignalPlotWidget, draw directly)
        self._render_psd_overlay(r)

        # Classification
        self._render_classification(r)

        # Metrics table
        self._render_metrics_table(r)

    def _render_psd_overlay(self, r: AudioExperimentResult):
        """Draw three PSDs on a single axes using the canvas directly."""
        canvas = self._psd_plot.canvas
        canvas.fig.clear()
        ax = canvas.fig.add_subplot(111)
        ax.set_facecolor("#0F172A")
        ax.tick_params(colors="#94A3B8", labelsize=9)
        for spine in ax.spines.values():
            spine.set_color("#334155")
        ax.xaxis.label.set_color("#CBD5E1")
        ax.yaxis.label.set_color("#CBD5E1")
        ax.title.set_color("#F8FAFC")
        ax.grid(True, color="#1E293B", linestyle="--", linewidth=0.7)

        def _plot_psd(analysis, color, label):
            if analysis and analysis.metrics.psd_frequencies is not None:
                f = analysis.metrics.psd_frequencies
                p = analysis.metrics.psd_values
                if len(f) > 0 and len(p) > 0:
                    db = 10 * np.log10(np.maximum(p, 1e-12))
                    ax.plot(f, db, color=color, linewidth=1.3, label=label, alpha=0.9)

        _plot_psd(r.clean_analysis,   _CLR_CLEAN,   "Clean")
        _plot_psd(r.noisy_analysis,   _CLR_NOISY,   "Noisy")
        _plot_psd(r.cleaned_analysis, _CLR_CLEANED, "Cleaned")

        ax.set_title("Power Spectral Density Comparison", fontsize=11,
                     fontweight="bold", pad=8)
        ax.set_xlabel("Frequency (Hz)", fontsize=9)
        ax.set_ylabel("Power (dB)", fontsize=9)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, facecolor="#1E293B",
                      edgecolor="#334155", labelcolor="#F8FAFC", fontsize=8)
        canvas.fig.tight_layout()
        canvas.draw()

    def _render_classification(self, r: AudioExperimentResult):
        # Ground truth always from generator
        gt = r.config.noise_type
        self._gt_badge.set_status(
            "Complete" if gt else "Ready"
        )
        # Rebuild the badge text
        self._gt_badge.findChild(QLabel).setText(f"Ground Truth: {gt or '—'}")

        # Prediction from noisy_analysis classifier
        if r.noisy_analysis and r.noisy_analysis.classification:
            clf = r.noisy_analysis.classification
            pred = clf.prediction or "—"
            conf = clf.confidence

            self._pred_badge.set_status(
                "Complete" if clf.model_available and not clf.is_unknown
                else "Warning"
            )
            self._pred_badge.findChild(QLabel).setText(f"Predicted: {pred}")
            self._conf_label.setText(
                f"Confidence: {conf:.1%}" if clf.model_available else "Model unavailable"
            )

            # Probability bars
            for cls_name, bar in self._prob_bars.items():
                bar.setValue(0)

            if clf.model_available and clf.probabilities is not None and clf.classes:
                for cls_name, bar in self._prob_bars.items():
                    if cls_name in clf.classes:
                        idx = list(clf.classes).index(cls_name)
                        bar.setValue(int(clf.probabilities[idx] * 100))

    def _render_metrics_table(self, r: AudioExperimentResult):
        cmp = r.quality_comparison
        if not cmp:
            return

        before = cmp.get("before", {})
        after  = cmp.get("after",  {})
        imp    = cmp.get("improvement", {})

        def _fmt(v):
            if v is None:
                return "—"
            if abs(v) == float("inf"):
                return "∞"
            return f"{v:.4f}"

        rows_data = [
            # (metric, before_key, after_key, imp_key, fmt_fn)
            ("SNR (dB)",
             _fmt(before.get("snr_db")),
             _fmt(after.get("snr_db")),
             f'{imp.get("snr_db", 0):+.4f}'),
            ("RMSE",
             _fmt(before.get("rmse")),
             _fmt(after.get("rmse")),
             f'{imp.get("rmse_reduction", 0):+.4f}'),
            ("Correlation",
             _fmt(before.get("correlation")),
             _fmt(after.get("correlation")),
             f'{imp.get("correlation_change", 0):+.4f}'),
            ("Energy Error",
             _fmt(before.get("energy_error")),
             _fmt(after.get("energy_error")),
             f'{imp.get("energy_error_reduction", 0):+.4f}'),
            ("Spectral Distortion",
             _fmt(before.get("spectral_distortion")),
             _fmt(after.get("spectral_distortion")),
             f'{imp.get("spectral_distortion_reduction", 0):+.4f}'),
        ]

        for i, (metric, bval, aval, ival) in enumerate(rows_data):
            self._metrics_table.setItem(i, 0, QTableWidgetItem(metric))
            self._metrics_table.setItem(i, 1, QTableWidgetItem(bval))
            self._metrics_table.setItem(i, 2, QTableWidgetItem(aval))
            imp_item = QTableWidgetItem(ival)
            # Colour improvement column
            try:
                f_val = float(ival.replace("+", "").replace("∞", "0"))
                imp_item.setForeground(
                    __import__("PySide6.QtGui", fromlist=["QColor"]).QColor(
                        _CLR_CLEANED if f_val >= 0 else _CLR_NOISY
                    )
                )
            except (ValueError, AttributeError):
                pass
            self._metrics_table.setItem(i, 3, imp_item)

    # ----------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------

    def get_last_result(self) -> AudioExperimentResult | None:
        return self._last_result

    def run_active_domain(self):
        """Called by DomainPage when the AUDIO tab becomes active.
        Only refreshes if a result already exists."""
        if self._last_result is not None:
            self._render(self._last_result)
