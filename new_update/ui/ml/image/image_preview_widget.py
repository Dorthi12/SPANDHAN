"""
SPANDHAN — Interactive 2D Signal Preview Widget
================================================
Renders high-resolution 2D matrix representations of signals with:
  - Interactive Pan & Smooth Zoom
  - Real-time hover coordinate (X, Y) and pixel intensity readouts
  - Scientific colormap modes (SPANDHAN Nebula, Grayscale, Viridis, Thermal)
  - Gridlines & dimension markers
  - Minimalistic technical controls (Zoom In/Out, Reset View, Colormap Cycle)
"""

from typing import Optional
import numpy as np
from PySide6.QtCore import Qt, QRectF, QPointF
from PySide6.QtGui import (
    QPainter,
    QColor,
    QImage,
    QPixmap,
    QPen,
    QFont,
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


def create_colormap_lut(mode: str) -> np.ndarray:
    """Generate 256x3 RGB lookup table for visualization."""
    lut = np.zeros((256, 3), dtype=np.uint8)
    indices = np.linspace(0, 1, 256)

    if mode == "nebula":
        # SPANDHAN Nebula palette: Deep Navy (0, 10, 30) -> Indigo (67, 56, 202)
        # -> Electric Blue (59, 130, 246) -> Persian Green (16, 185, 129) -> White
        for i, t in enumerate(indices):
            if t < 0.25:
                s = t / 0.25
                r = int(7 + s * (67 - 7))
                g = int(17 + s * (56 - 17))
                b = int(31 + s * (202 - 31))
            elif t < 0.55:
                s = (t - 0.25) / 0.30
                r = int(67 + s * (59 - 67))
                g = int(56 + s * (130 - 56))
                b = int(202 + s * (246 - 202))
            elif t < 0.85:
                s = (t - 0.55) / 0.30
                r = int(59 + s * (16 - 59))
                g = int(130 + s * (185 - 130))
                b = int(246 + s * (129 - 246))
            else:
                s = (t - 0.85) / 0.15
                r = int(16 + s * (255 - 16))
                g = int(185 + s * (255 - 185))
                b = int(129 + s * (255 - 129))
            lut[i] = [r, g, b]

    elif mode == "viridis":
        # Approximate viridis: Purple -> Teal -> Yellow
        for i, t in enumerate(indices):
            r = int(np.clip(255 * (0.28 + 0.72 * (t ** 2)), 0, 255))
            g = int(np.clip(255 * (0.01 + 0.98 * (t ** 1.2)), 0, 255))
            b = int(np.clip(255 * (0.33 + 0.67 * (1.0 - (t - 0.5) ** 2)), 0, 255))
            lut[i] = [r, g, b]

    elif mode == "thermal":
        # Black -> Red -> Orange -> Yellow -> White
        for i, t in enumerate(indices):
            r = int(np.clip(255 * (t * 2.0), 0, 255))
            g = int(np.clip(255 * ((t - 0.35) * 2.0), 0, 255))
            b = int(np.clip(255 * ((t - 0.75) * 4.0), 0, 255))
            lut[i] = [r, g, b]

    else:
        # Standard Grayscale
        for i, t in enumerate(indices):
            val = int(t * 255)
            lut[i] = [val, val, val]

    return lut


class MatrixCanvas(QWidget):
    """Core interactive canvas rendering the 2D signal matrix with zoom and pan."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setMinimumSize(280, 240)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.CrossCursor)

        self._signal_matrix: Optional[np.ndarray] = None
        self._pixmap: Optional[QPixmap] = None
        self._colormap_mode = "nebula"

        # Transform states
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._last_mouse_pos = QPointF(0, 0)
        self._is_panning = False

        # Hover readout
        self._hover_pixel: Optional[tuple[int, int, float]] = None
        self.hover_changed = None  # callback

    def set_signal_matrix(self, matrix: np.ndarray):
        """Set normalized [0, 1] 2D matrix (128x128)."""
        self._signal_matrix = np.clip(matrix.squeeze().astype(np.float32), 0.0, 1.0)
        self._update_pixmap()
        self.reset_view()

    def set_colormap(self, mode: str):
        self._colormap_mode = mode
        if self._signal_matrix is not None:
            self._update_pixmap()
            self.update()

    def _update_pixmap(self):
        if self._signal_matrix is None:
            self._pixmap = None
            return

        h, w = self._signal_matrix.shape
        lut = create_colormap_lut(self._colormap_mode)

        # Quantize to 0-255 indices
        idx = np.clip(np.round(self._signal_matrix * 255.0), 0, 255).astype(np.uint8)
        rgb_data = lut[idx]  # shape (h, w, 3)

        # Create QImage from RGB buffer
        # Ensure contiguous array
        rgb_contiguous = np.ascontiguousarray(rgb_data)
        qimg = QImage(
            rgb_contiguous.data,
            w,
            h,
            w * 3,
            QImage.Format.Format_RGB888,
        )
        self._pixmap = QPixmap.fromImage(qimg)

    def reset_view(self):
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self.update()

    def zoom_in(self):
        self._zoom = min(8.0, self._zoom * 1.25)
        self.update()

    def zoom_out(self):
        self._zoom = max(0.5, self._zoom / 1.25)
        self.update()

    def wheelEvent(self, event: QWheelEvent):
        degrees = event.angleDelta().y() / 8.0
        steps = degrees / 15.0
        factor = 1.15 ** steps
        new_zoom = max(0.4, min(10.0, self._zoom * factor))
        self._zoom = new_zoom
        self.update()
        event.accept()

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self._is_panning = True
            self._last_mouse_pos = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        elif event.button() == Qt.MouseButton.RightButton:
            self.reset_view()

    def mouseMoveEvent(self, event: QMouseEvent):
        pos = event.position()
        if self._is_panning:
            delta = pos - self._last_mouse_pos
            self._pan_x += delta.x()
            self._pan_y += delta.y()
            self._last_mouse_pos = pos
            self.update()
        else:
            # Calculate pixel coordinate under cursor
            if self._signal_matrix is not None:
                h, w = self._signal_matrix.shape
                # Center coords
                cx = self.width() / 2.0 + self._pan_x
                cy = self.height() / 2.0 + self._pan_y

                scale = min(self.width(), self.height()) * 0.78 * self._zoom
                img_left = cx - scale / 2.0
                img_top = cy - scale / 2.0

                rel_x = (pos.x() - img_left) / scale
                rel_y = (pos.y() - img_top) / scale

                if 0.0 <= rel_x <= 1.0 and 0.0 <= rel_y <= 1.0:
                    px = int(np.clip(rel_x * w, 0, w - 1))
                    py = int(np.clip(rel_y * h, 0, h - 1))
                    val = float(self._signal_matrix[py, px])
                    self._hover_pixel = (px, py, val)
                else:
                    self._hover_pixel = None

                if self.hover_changed:
                    self.hover_changed(self._hover_pixel)
                self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self._is_panning = False
            self.setCursor(Qt.CursorShape.CrossCursor)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        w = self.width()
        h = self.height()

        # Canvas background (Dark midnight navy)
        painter.fillRect(0, 0, w, h, QColor(COLORS.BG_INPUT))

        # Fine background gridlines
        grid_pen = QPen(QColor(30, 41, 59, 80), 1.0, Qt.PenStyle.DotLine)
        painter.setPen(grid_pen)
        grid_step = 32
        for gx in range(0, w, grid_step):
            painter.drawLine(gx, 0, gx, h)
        for gy in range(0, h, grid_step):
            painter.drawLine(0, gy, w, gy)

        if self._pixmap is None or self._pixmap.isNull():
            # Empty placeholder state
            painter.setPen(QColor(COLORS.TEXT_MUTED))
            font = painter.font()
            font.setPointSize(11)
            painter.setFont(font)
            painter.drawText(
                QRectF(0, 0, w, h),
                Qt.AlignmentFlag.AlignCenter,
                "Awaiting signal image for interactive 2D visualization...",
            )
            return

        # Render matrix image centered with transform
        cx = w / 2.0 + self._pan_x
        cy = h / 2.0 + self._pan_y

        scale = min(w, h) * 0.78 * self._zoom
        img_rect = QRectF(cx - scale / 2.0, cy - scale / 2.0, scale, scale)

        # Draw matrix frame with soft border
        painter.setPen(QPen(QColor(COLORS.BORDER_HOVER), 1.2))
        painter.drawRect(img_rect.adjusted(-1, -1, 1, 1))

        # Draw image
        painter.drawPixmap(img_rect.toRect(), self._pixmap)

        # Draw dimension markers along axes
        axis_pen = QPen(QColor(COLORS.TEXT_MUTED), 1.0)
        painter.setPen(axis_pen)
        font = painter.font()
        font.setPointSize(8)
        font.setFamily("Consolas")
        painter.setFont(font)

        # Ticks: 0, 32, 64, 96, 128
        ticks = [0, 32, 64, 96, 128]
        for t in ticks:
            # X axis tick (bottom)
            tx = img_rect.left() + (t / 128.0) * img_rect.width()
            painter.drawLine(int(tx), int(img_rect.bottom()), int(tx), int(img_rect.bottom() + 4))
            painter.drawText(int(tx - 10), int(img_rect.bottom() + 14), str(t))

            # Y axis tick (left)
            ty = img_rect.top() + (t / 128.0) * img_rect.height()
            painter.drawLine(int(img_rect.left() - 4), int(ty), int(img_rect.left()), int(ty))
            painter.drawText(int(img_rect.left() - 24), int(ty + 4), str(t))

        # If hovering inside matrix, draw crosshair lines
        if self._hover_pixel is not None:
            px, py, val = self._hover_pixel
            hx = img_rect.left() + (px / 128.0) * img_rect.width()
            hy = img_rect.top() + (py / 128.0) * img_rect.height()

            ch_pen = QPen(QColor(16, 185, 129, 180), 1.0, Qt.PenStyle.DashLine)
            painter.setPen(ch_pen)
            painter.drawLine(int(hx), int(img_rect.top()), int(hx), int(img_rect.bottom()))
            painter.drawLine(int(img_rect.left()), int(hy), int(img_rect.right()), int(hy))


class ImageSignalPreviewWidget(QFrame):
    """
    Complete interactive Signal Preview card container featuring:
      - MatrixCanvas
      - Zoom / Pan / Reset Controls
      - Colormap Selector
      - Dynamic Hover Coordinate/Intensity Readout
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("SignalPreviewFrame")
        self.setStyleSheet(
            f"QFrame#SignalPreviewFrame {{ background-color: {COLORS.BG_CARD}; "
            f"border: 1px solid {COLORS.BORDER_SUBTLE}; border-radius: 9px; }}"
        )

        self._colormaps = ["nebula", "grayscale", "viridis", "thermal"]
        self._cm_idx = 0

        self._init_ui()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(14, 12, 14, 12)
        main_layout.setSpacing(10)

        # Top Header & Controls Strip
        header_layout = QHBoxLayout()
        header_layout.setSpacing(8)

        # Title
        title_col = QVBoxLayout()
        title_col.setSpacing(1)
        title_lbl = QLabel("SIGNAL PREVIEW")
        title_lbl.setStyleSheet(
            f"color: {COLORS.TEXT_PRIMARY}; font-size: 11px; font-weight: 700; "
            f"letter-spacing: 1.0px;"
        )
        sub_lbl = QLabel("Interactive 2D spatial waveform & spectral matrix")
        sub_lbl.setStyleSheet(f"color: {COLORS.TEXT_MUTED}; font-size: 10px;")
        title_col.addWidget(title_lbl)
        title_col.addWidget(sub_lbl)
        header_layout.addLayout(title_col)

        header_layout.addStretch()

        # Hover readout badge
        self.hover_badge = QLabel("X: —  Y: —  I: —")
        self.hover_badge.setStyleSheet(
            f"color: {COLORS.PERSIAN_GREEN}; background-color: #07111F; "
            f"border: 1px solid {COLORS.BORDER_SUBTLE}; border-radius: 5px; "
            f"padding: 3px 8px; font-size: 10px; font-family: 'Consolas', monospace;"
        )
        header_layout.addWidget(self.hover_badge)

        # Colormap button
        self.cm_btn = QPushButton("Palette: Nebula")
        self.cm_btn.setStyleSheet(
            f"QPushButton {{ background-color: {COLORS.BG_SURFACE_ALT}; color: {COLORS.TEXT_PRIMARY}; "
            f"border: 1px solid {COLORS.BORDER_SUBTLE}; border-radius: 5px; padding: 4px 8px; font-size: 10px; }}"
            f"QPushButton:hover {{ background-color: {COLORS.INDIGO}; border-color: {COLORS.BLUE}; }}"
        )
        self.cm_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.cm_btn.clicked.connect(self._cycle_colormap)
        header_layout.addWidget(self.cm_btn)

        # Zoom Out
        zoom_out_btn = QPushButton("－")
        zoom_out_btn.setFixedSize(26, 26)
        zoom_out_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        zoom_out_btn.setStyleSheet(
            f"QPushButton {{ background-color: {COLORS.BG_SURFACE_ALT}; color: {COLORS.TEXT_PRIMARY}; "
            f"border: 1px solid {COLORS.BORDER_SUBTLE}; border-radius: 5px; font-size: 12px; }}"
            f"QPushButton:hover {{ background-color: {COLORS.INDIGO}; }}"
        )
        zoom_out_btn.clicked.connect(lambda: self.canvas.zoom_out())
        header_layout.addWidget(zoom_out_btn)

        # Zoom In
        zoom_in_btn = QPushButton("＋")
        zoom_in_btn.setFixedSize(26, 26)
        zoom_in_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        zoom_in_btn.setStyleSheet(
            f"QPushButton {{ background-color: {COLORS.BG_SURFACE_ALT}; color: {COLORS.TEXT_PRIMARY}; "
            f"border: 1px solid {COLORS.BORDER_SUBTLE}; border-radius: 5px; font-size: 12px; }}"
            f"QPushButton:hover {{ background-color: {COLORS.INDIGO}; }}"
        )
        zoom_in_btn.clicked.connect(lambda: self.canvas.zoom_in())
        header_layout.addWidget(zoom_in_btn)

        # Reset view
        reset_btn = QPushButton("⟲ Reset")
        reset_btn.setStyleSheet(
            f"QPushButton {{ background-color: {COLORS.BG_SURFACE_ALT}; color: {COLORS.TEXT_SECONDARY}; "
            f"border: 1px solid {COLORS.BORDER_SUBTLE}; border-radius: 5px; padding: 4px 8px; font-size: 10px; }}"
            f"QPushButton:hover {{ background-color: {COLORS.BG_CARD_HOVER}; color: #FFF; }}"
        )
        reset_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        reset_btn.clicked.connect(lambda: self.canvas.reset_view())
        header_layout.addWidget(reset_btn)

        main_layout.addLayout(header_layout)

        # Canvas Area
        self.canvas = MatrixCanvas(self)
        self.canvas.hover_changed = self._on_hover_changed
        main_layout.addWidget(self.canvas)

        # Bottom hint
        hint_lbl = QLabel("Drag to pan  •  Scroll to zoom  •  Hover to inspect pixel intensity  •  Right-click to reset")
        hint_lbl.setStyleSheet(f"color: {COLORS.TEXT_MUTED}; font-size: 9px;")
        hint_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(hint_lbl)

    def set_signal_matrix(self, matrix: np.ndarray):
        self.canvas.set_signal_matrix(matrix)

    def _cycle_colormap(self):
        self._cm_idx = (self._cm_idx + 1) % len(self._colormaps)
        mode = self._colormaps[self._cm_idx]
        self.cm_btn.setText(f"Palette: {mode.capitalize()}")
        self.canvas.set_colormap(mode)

    def _on_hover_changed(self, hover_data):
        if hover_data:
            x, y, val = hover_data
            self.hover_badge.setText(f"X: {x:3d}  Y: {y:3d}  I: {val:.4f}")
        else:
            self.hover_badge.setText("X: —  Y: —  I: —")
