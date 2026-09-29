"""
SPANDHAN — Interactive Audio Waveform Preview Widget
=====================================================
Renders high-fidelity 1D acoustic signal waveforms with:
  - Min-max decimation rendering (guaranteeing zero missed impulse spikes)
  - Dark plotting surface (#081426)
  - Electric blue / indigo primary waveform with subtle Persian-green highlights
  - Clean low-contrast grid and precise Amplitude / Time (s) axes
  - Interactive Zoom In, Zoom Out, Reset controls, and wheel zoom
  - Real-time hover cursor with Time/Amplitude floating readout tooltip
  - Clean empty placeholder state when no audio signal is loaded
"""

from __future__ import annotations

from typing import Optional
import numpy as np
from PySide6.QtCore import Qt, QRectF, QPointF, Signal
from PySide6.QtGui import (
    QPainter,
    QColor,
    QPen,
    QFont,
    QPainterPath,
    QLinearGradient,
    QWheelEvent,
    QMouseEvent,
)
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFrame,
    QSizePolicy,
)

from ui.styles.theme import COLORS


class WaveformCanvas(QWidget):
    """Core interactive canvas rendering the 1D signal waveform."""

    browse_requested = Signal()
    view_changed = Signal(float, float, float)  # t_min, t_max, zoom

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setMinimumSize(360, 220)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)

        self._signal: Optional[np.ndarray] = None
        self._sr: int = 16000
        self._duration: float = 0.0

        # Viewport time boundaries
        self._view_t_min: float = 0.0
        self._view_t_max: float = 0.0
        self._zoom: float = 1.0

        # Pan state
        self._is_panning: bool = False
        self._pan_start_x: float = 0.0
        self._pan_start_t_min: float = 0.0
        self._pan_start_t_max: float = 0.0

        # Hover state
        self._hover_pt: Optional[QPointF] = None
        self._hover_time: Optional[float] = None
        self._hover_amp: Optional[float] = None

        # Empty state button
        self._empty_btn = QPushButton("Browse Audio", self)
        self._empty_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._empty_btn.setStyleSheet(
            f"QPushButton {{ background-color: {COLORS.BG_SURFACE_ALT}; color: {COLORS.TEXT_PRIMARY}; "
            f"border: 1px solid {COLORS.BORDER_SUBTLE}; border-radius: 6px; padding: 7px 16px; font-size: 11px; font-weight: 600; }} "
            f"QPushButton:hover {{ background-color: {COLORS.INDIGO}; border-color: {COLORS.BLUE}; }}"
        )
        self._empty_btn.clicked.connect(self.browse_requested.emit)
        self._empty_btn.hide()

    def set_signal(self, signal: Optional[np.ndarray], sr: int = 16000):
        if signal is None or len(signal) == 0:
            self._signal = None
            self._sr = 16000
            self._duration = 0.0
            self._view_t_min = 0.0
            self._view_t_max = 0.0
            self._zoom = 1.0
            self._empty_btn.show()
        else:
            self._signal = np.asarray(signal, dtype=np.float64).ravel()
            self._sr = max(1, int(sr))
            self._duration = float(len(self._signal) / self._sr)
            self._view_t_min = 0.0
            self._view_t_max = self._duration
            self._zoom = 1.0
            self._empty_btn.hide()

        self._hover_pt = None
        self._hover_time = None
        self._hover_amp = None
        self.update()
        self.view_changed.emit(self._view_t_min, self._view_t_max, self._zoom)

    def zoom_in(self):
        if self._signal is None or self._duration <= 0:
            return
        span = self._view_t_max - self._view_t_min
        center = (self._view_t_min + self._view_t_max) / 2.0
        new_span = span / 1.5
        if new_span < 0.0005:  # Limit maximum zoom (0.5 ms window)
            return
        self._view_t_min = max(0.0, center - new_span / 2.0)
        self._view_t_max = min(self._duration, center + new_span / 2.0)
        self._zoom = self._duration / max(1e-9, (self._view_t_max - self._view_t_min))
        self.update()
        self.view_changed.emit(self._view_t_min, self._view_t_max, self._zoom)

    def zoom_out(self):
        if self._signal is None or self._duration <= 0:
            return
        span = self._view_t_max - self._view_t_min
        center = (self._view_t_min + self._view_t_max) / 2.0
        new_span = min(self._duration, span * 1.5)
        self._view_t_min = max(0.0, center - new_span / 2.0)
        self._view_t_max = min(self._duration, self._view_t_min + new_span)
        if self._view_t_max >= self._duration:
            self._view_t_min = max(0.0, self._duration - new_span)
        self._zoom = self._duration / max(1e-9, (self._view_t_max - self._view_t_min))
        self.update()
        self.view_changed.emit(self._view_t_min, self._view_t_max, self._zoom)

    def reset_view(self):
        if self._signal is None or self._duration <= 0:
            return
        self._view_t_min = 0.0
        self._view_t_max = self._duration
        self._zoom = 1.0
        self.update()
        self.view_changed.emit(self._view_t_min, self._view_t_max, self._zoom)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Position the empty state button in the center
        btn_w, btn_h = 130, 32
        self._empty_btn.setGeometry(
            int((self.width() - btn_w) / 2),
            int(self.height() / 2 + 28),
            btn_w,
            btn_h,
        )

    def wheelEvent(self, event: QWheelEvent):
        if self._signal is None or self._duration <= 0:
            return
        delta = event.angleDelta().y()
        if delta == 0:
            return

        m_left = 55.0
        m_right = 20.0
        plot_w = self.width() - m_left - m_right
        if plot_w <= 0:
            return

        cursor_x = event.position().x()
        rel_x = np.clip((cursor_x - m_left) / plot_w, 0.0, 1.0)
        curr_t = self._view_t_min + rel_x * (self._view_t_max - self._view_t_min)

        factor = 0.8 if delta > 0 else 1.25
        span = (self._view_t_max - self._view_t_min) * factor
        span = max(0.0005, min(self._duration, span))

        self._view_t_min = max(0.0, curr_t - rel_x * span)
        self._view_t_max = min(self._duration, self._view_t_min + span)
        if self._view_t_max >= self._duration:
            self._view_t_min = max(0.0, self._duration - span)

        self._zoom = self._duration / max(1e-9, (self._view_t_max - self._view_t_min))
        self.update()
        self.view_changed.emit(self._view_t_min, self._view_t_max, self._zoom)
        event.accept()

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton and self._signal is not None:
            self._is_panning = True
            self._pan_start_x = event.position().x()
            self._pan_start_t_min = self._view_t_min
            self._pan_start_t_max = self._view_t_max
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        m_left = 55.0
        m_right = 20.0
        m_top = 16.0
        m_bot = 32.0
        plot_w = self.width() - m_left - m_right
        plot_h = self.height() - m_top - m_bot

        if self._signal is not None and plot_w > 0:
            cur_x = event.position().x()
            cur_y = event.position().y()

            if self._is_panning:
                dx = cur_x - self._pan_start_x
                span = self._pan_start_t_max - self._pan_start_t_min
                dt = -(dx / plot_w) * span

                new_min = self._pan_start_t_min + dt
                new_max = self._pan_start_t_max + dt

                if new_min < 0.0:
                    new_max -= new_min
                    new_min = 0.0
                if new_max > self._duration:
                    new_min -= (new_max - self._duration)
                    new_max = self._duration
                    new_min = max(0.0, new_min)

                self._view_t_min = new_min
                self._view_t_max = new_max
                self.update()
                self.view_changed.emit(self._view_t_min, self._view_t_max, self._zoom)
            else:
                # Update hover readout
                if m_left <= cur_x <= m_left + plot_w and m_top <= cur_y <= m_top + plot_h:
                    rel_t = (cur_x - m_left) / plot_w
                    t_val = self._view_t_min + rel_t * (self._view_t_max - self._view_t_min)
                    sample_idx = int(np.clip(t_val * self._sr, 0, len(self._signal) - 1))
                    amp_val = float(self._signal[sample_idx])

                    self._hover_pt = QPointF(cur_x, cur_y)
                    self._hover_time = t_val
                    self._hover_amp = amp_val
                else:
                    self._hover_pt = None
                    self._hover_time = None
                    self._hover_amp = None
                self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton and self._is_panning:
            self._is_panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def leaveEvent(self, event):
        self._hover_pt = None
        self._hover_time = None
        self._hover_amp = None
        self.update()
        super().leaveEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = self.rect()
        w = rect.width()
        h = rect.height()

        # 1. Background dark navy surface
        painter.fillRect(rect, QColor(COLORS.BG_INPUT))

        # Margins for axes
        m_left = 55.0
        m_right = 20.0
        m_top = 16.0
        m_bot = 32.0
        plot_w = max(10.0, w - m_left - m_right)
        plot_h = max(10.0, h - m_top - m_bot)
        plot_rect = QRectF(m_left, m_top, plot_w, plot_h)

        # Plot frame border
        painter.setPen(QColor("#1E293B"))
        painter.setBrush(QColor("#060F1D"))
        painter.drawRoundedRect(plot_rect, 4, 4)

        if self._signal is None:
            # -------------------------------------------------------------
            # EMPTY STATE PLACEHOLDER
            # -------------------------------------------------------------
            self._empty_btn.show()

            # Waveform glyph
            font = painter.font()
            font.setPointSize(28)
            painter.setFont(font)
            painter.setPen(QColor(COLORS.BLUE_LIGHT))
            painter.drawText(
                QRectF(m_left, m_top + plot_h * 0.15, plot_w, 40),
                Qt.AlignmentFlag.AlignCenter,
                "〰",
            )

            # Title
            font.setPointSize(11)
            font.setBold(True)
            font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 1.0)
            painter.setFont(font)
            painter.setPen(QColor(COLORS.TEXT_PRIMARY))
            painter.drawText(
                QRectF(m_left, m_top + plot_h * 0.35, plot_w, 24),
                Qt.AlignmentFlag.AlignCenter,
                "NO AUDIO SIGNAL SELECTED",
            )

            # Subtitle
            font.setPointSize(10)
            font.setBold(False)
            font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 0.0)
            painter.setFont(font)
            painter.setPen(QColor(COLORS.TEXT_MUTED))
            painter.drawText(
                QRectF(m_left, m_top + plot_h * 0.47, plot_w, 20),
                Qt.AlignmentFlag.AlignCenter,
                "Import an audio file to begin ML classification.",
            )
            return

        self._empty_btn.hide()

        # -----------------------------------------------------------------
        # 2. GRIDLINES & AXES TICKS
        # -----------------------------------------------------------------
        y_zero = m_top + plot_h / 2.0

        # Horizontal gridlines & Amplitude ticks (-1.0, -0.5, 0.0, +0.5, +1.0)
        amp_ticks = [-1.0, -0.5, 0.0, 0.5, 1.0]
        font = painter.font()
        font.setPointSize(8)
        font.setFamily("Consolas")
        painter.setFont(font)

        for a_val in amp_ticks:
            y_pos = y_zero - (a_val / 1.15) * (plot_h / 2.0)
            if m_top <= y_pos <= m_top + plot_h:
                # Gridline
                if a_val == 0.0:
                    painter.setPen(QPen(QColor(59, 130, 246, 55), 1, Qt.PenStyle.DashLine))
                else:
                    painter.setPen(QPen(QColor(59, 130, 246, 20), 1, Qt.PenStyle.DotLine))
                painter.drawLine(int(m_left), int(y_pos), int(m_left + plot_w), int(y_pos))

                # Tick label (Amplitude)
                painter.setPen(QColor(COLORS.TEXT_MUTED))
                lbl_str = f"{a_val:+.1f}" if a_val != 0.0 else " 0.0"
                painter.drawText(
                    QRectF(4, y_pos - 7, m_left - 10, 14),
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                    lbl_str,
                )

        # Vertical gridlines & Time ticks
        t_span = max(1e-6, self._view_t_max - self._view_t_min)
        n_time_ticks = 5
        dt = t_span / (n_time_ticks - 1)
        for i in range(n_time_ticks):
            t_curr = self._view_t_min + i * dt
            x_pos = m_left + (i / (n_time_ticks - 1)) * plot_w

            # Vertical gridline
            painter.setPen(QPen(QColor(59, 130, 246, 20), 1, Qt.PenStyle.DotLine))
            painter.drawLine(int(x_pos), int(m_top), int(x_pos), int(m_top + plot_h))

            # Tick label (Time in seconds)
            painter.setPen(QColor(COLORS.TEXT_MUTED))
            t_str = f"{t_curr:.2f}s" if t_span >= 0.1 else f"{t_curr * 1000:.1f}ms"
            painter.drawText(
                QRectF(x_pos - 25, m_top + plot_h + 4, 50, 16),
                Qt.AlignmentFlag.AlignCenter,
                t_str,
            )

        # Axis Titles
        font.setPointSize(8)
        font.setFamily("Segoe UI")
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QColor(COLORS.TEXT_SECONDARY))

        # Y Axis title: Amplitude (rotated or top-left)
        painter.drawText(QRectF(8, 2, 80, 14), Qt.AlignmentFlag.AlignLeft, "Amplitude")

        # X Axis title: Time (s)
        painter.drawText(
            QRectF(m_left, h - 14, plot_w, 14),
            Qt.AlignmentFlag.AlignCenter,
            "Time (s)",
        )

        # -----------------------------------------------------------------
        # 3. WAVEFORM PLOT (Min-Max Decimation for 100% Transient Fidelity)
        # -----------------------------------------------------------------
        n_cols = int(plot_w)
        if n_cols > 1 and len(self._signal) > 0:
            painter.save()
            painter.setClipRect(plot_rect)

            # Gradient for area under waveform
            fill_path = QPainterPath()
            fill_path.moveTo(m_left, y_zero)

            stroke_path = QPainterPath()
            first_pt = True

            # Calculate indices corresponding to viewport
            idx_start = int(np.clip(self._view_t_min * self._sr, 0, len(self._signal) - 1))
            idx_end = int(np.clip(self._view_t_max * self._sr, idx_start + 1, len(self._signal)))
            total_visible_samples = max(1, idx_end - idx_start)

            samples_per_pixel = total_visible_samples / float(n_cols)

            pts_top = []
            pts_bot = []

            for col in range(n_cols):
                px = m_left + col
                s_i = int(idx_start + col * samples_per_pixel)
                e_i = int(min(len(self._signal), s_i + max(1, int(np.ceil(samples_per_pixel)))))

                chunk = self._signal[s_i:e_i]
                if len(chunk) > 0:
                    c_min = float(np.min(chunk))
                    c_max = float(np.max(chunk))
                else:
                    c_min = 0.0
                    c_max = 0.0

                y_max_px = y_zero - (c_max / 1.15) * (plot_h / 2.0)
                y_min_px = y_zero - (c_min / 1.15) * (plot_h / 2.0)

                pts_top.append(QPointF(px, y_max_px))
                pts_bot.append(QPointF(px, y_min_px))

            # Create filled polygon: top envelope then reversed bottom envelope
            if pts_top:
                fill_path.moveTo(pts_top[0])
                for pt in pts_top:
                    fill_path.lineTo(pt)
                for pt in reversed(pts_bot):
                    fill_path.lineTo(pt)
                fill_path.closeSubpath()

                # Soft electric blue / Persian green fill
                grad = QLinearGradient(0, m_top, 0, m_top + plot_h)
                grad.setColorAt(0.0, QColor(59, 130, 246, 30))
                grad.setColorAt(0.5, QColor(16, 185, 129, 20))
                grad.setColorAt(1.0, QColor(67, 56, 202, 25))
                painter.setBrush(grad)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawPath(fill_path)

                # Primary Stroke
                pen_wave = QPen(QColor(COLORS.BLUE_LIGHT), 1.3)
                painter.setPen(pen_wave)
                for i in range(len(pts_top)):
                    painter.drawLine(pts_top[i], pts_bot[i])

                # Subtle Persian-Green highlight line on top ridge
                pen_ridge = QPen(QColor(COLORS.PERSIAN_GREEN), 1.0)
                painter.setPen(pen_ridge)
                for i in range(len(pts_top) - 1):
                    painter.drawLine(pts_top[i], pts_top[i + 1])

            painter.restore()

        # -----------------------------------------------------------------
        # 4. HOVER CURSOR & TOOLTIP
        # -----------------------------------------------------------------
        if self._hover_pt is not None and self._hover_time is not None and self._hover_amp is not None:
            hx = self._hover_pt.x()
            hy = y_zero - (self._hover_amp / 1.15) * (plot_h / 2.0)

            # Vertical crosshair line
            painter.setPen(QPen(QColor(COLORS.CYAN), 1, Qt.PenStyle.DashLine))
            painter.drawLine(int(hx), int(m_top), int(hx), int(m_top + plot_h))

            # Point highlight on signal
            painter.setBrush(QColor(COLORS.PERSIAN_GREEN))
            painter.setPen(QColor("#FFFFFF"))
            painter.drawEllipse(QPointF(hx, hy), 3.5, 3.5)

            # Tooltip badge
            tip_str = f"Time: {self._hover_time:.3f} s  |  Amp: {self._hover_amp:+.3f}"
            font = painter.font()
            font.setPointSize(9)
            font.setFamily("Consolas")
            painter.setFont(font)

            badge_w = 210.0
            badge_h = 22.0
            badge_x = min(m_left + plot_w - badge_w - 4, max(m_left + 4, hx - badge_w / 2.0))
            badge_y = m_top + 8 if hy > m_top + 40 else m_top + plot_h - badge_h - 8

            tip_rect = QRectF(badge_x, badge_y, badge_w, badge_h)
            painter.setBrush(QColor(7, 17, 31, 230))
            painter.setPen(QColor(COLORS.BORDER_HOVER))
            painter.drawRoundedRect(tip_rect, 4, 4)

            painter.setPen(QColor(COLORS.TEXT_PRIMARY))
            painter.drawText(tip_rect, Qt.AlignmentFlag.AlignCenter, tip_str)


class AudioSignalPreviewWidget(QFrame):
    """
    Complete Signal Preview Card with embedded WaveformCanvas and lightweight controls:
    [ Zoom In ] [ Zoom Out ] [ Reset ]
    """

    browse_requested = Signal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("AudioSignalPreviewCard")
        self.setStyleSheet(
            f"QFrame#AudioSignalPreviewCard {{ background-color: {COLORS.BG_CARD}; "
            f"border: 1px solid {COLORS.BORDER_SUBTLE}; border-radius: 9px; }} "
            f"QFrame#AudioSignalPreviewCard:hover {{ border: 1px solid {COLORS.BORDER_HOVER}; }}"
        )

        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 14)
        layout.setSpacing(10)

        # Header bar
        header = QHBoxLayout()
        header.setSpacing(8)

        t_col = QVBoxLayout()
        t_col.setSpacing(1)
        self.title_lbl = QLabel("SIGNAL PREVIEW")
        self.title_lbl.setStyleSheet(
            f"color: {COLORS.TEXT_PRIMARY}; font-size: 11px; font-weight: 700; letter-spacing: 1.0px;"
        )
        self.sub_lbl = QLabel("1D Acoustic waveform amplitude vs time (s)")
        self.sub_lbl.setStyleSheet(f"color: {COLORS.TEXT_MUTED}; font-size: 10px;")
        t_col.addWidget(self.title_lbl)
        t_col.addWidget(self.sub_lbl)
        header.addLayout(t_col)

        header.addStretch()

        # Viewport zoom indicator
        self.zoom_lbl = QLabel("Zoom: 1.0×")
        self.zoom_lbl.setStyleSheet(
            f"color: {COLORS.TEXT_MUTED}; font-size: 10px; font-family: 'Consolas', monospace; margin-right: 6px;"
        )
        header.addWidget(self.zoom_lbl)

        # Lightweight Controls: Zoom In, Zoom Out, Reset
        self.zoom_in_btn = self._make_tool_btn("+", "Zoom In")
        self.zoom_in_btn.clicked.connect(self._on_zoom_in)
        header.addWidget(self.zoom_in_btn)

        self.zoom_out_btn = self._make_tool_btn("−", "Zoom Out")
        self.zoom_out_btn.clicked.connect(self._on_zoom_out)
        header.addWidget(self.zoom_out_btn)

        self.reset_btn = self._make_tool_btn("↺", "Reset View")
        self.reset_btn.clicked.connect(self._on_reset)
        header.addWidget(self.reset_btn)

        layout.addLayout(header)

        # Waveform Canvas
        self.canvas = WaveformCanvas(self)
        self.canvas.browse_requested.connect(self.browse_requested.emit)
        self.canvas.view_changed.connect(self._on_view_changed)
        layout.addWidget(self.canvas, stretch=1)

    def _make_tool_btn(self, text: str, tooltip: str) -> QPushButton:
        btn = QPushButton(text)
        btn.setToolTip(tooltip)
        btn.setFixedSize(26, 24)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setStyleSheet(
            f"QPushButton {{ background-color: {COLORS.BG_SURFACE_ALT}; color: {COLORS.TEXT_SECONDARY}; "
            f"border: 1px solid {COLORS.BORDER_SUBTLE}; border-radius: 4px; font-size: 12px; font-weight: 700; }} "
            f"QPushButton:hover {{ background-color: {COLORS.INDIGO}; color: #FFFFFF; border-color: {COLORS.BLUE}; }}"
        )
        return btn

    def set_signal(self, signal: Optional[np.ndarray], sr: int = 16000):
        self.canvas.set_signal(signal, sr)

    def _on_zoom_in(self):
        self.canvas.zoom_in()

    def _on_zoom_out(self):
        self.canvas.zoom_out()

    def _on_reset(self):
        self.canvas.reset_view()

    def _on_view_changed(self, t_min: float, t_max: float, zoom: float):
        if zoom <= 1.01:
            self.zoom_lbl.setText("Zoom: 1.0× (Full)")
        else:
            self.zoom_lbl.setText(f"Zoom: {zoom:.1f}× [{t_min:.2f}s–{t_max:.2f}s]")
