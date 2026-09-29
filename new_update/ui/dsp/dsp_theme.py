"""
SPANDHAN — DSP Visualization Theme & Styling
=============================================
Defines color palettes, typography, Qt stylesheets, and Matplotlib
scientific styling configurations for dark-mode DSP signal visualization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes


@dataclass(frozen=True)
class DSPColors:
    # Foundations
    BG_PRIMARY: str = "#07111F"       # Midnight dark navy
    BG_SECONDARY: str = "#0B1930"     # Deep navy sidebar/container
    BG_CARD: str = "#0D1B2E"          # Glassmorphic card surface
    BG_CARD_HOVER: str = "#11223B"    # Card hover
    BG_PLOT: str = "#081426"          # Dark canvas plot area
    BG_INPUT: str = "#081426"         # Input / well container
    BG_SURFACE_ALT: str = "#13233E"   # Raised chip surface

    # SPANDHAN Accents
    INDIGO: str = "#4338CA"           # Core indigo
    INDIGO_LIGHT: str = "#6366F1"     # Bright indigo
    INDIGO_LINE: str = "#818CF8"      # Plot line indigo
    BLUE: str = "#3B82F6"             # Electric blue
    BLUE_LIGHT: str = "#60A5FA"       # Plot line blue
    PERSIAN_GREEN: str = "#10B981"    # Persian green / emerald
    GREEN_LIGHT: str = "#34D399"      # Plot line green
    CYAN: str = "#38BDF8"             # High-contrast signal cyan
    AMBER: str = "#F59E0B"            # Marker / warning amber
    ROSE: str = "#F43F5E"             # Diagnostic / error red
    VIOLET: str = "#A855F7"           # Wavelet accent violet

    # Typography
    TEXT_PRIMARY: str = "#F8FAFC"     # Slate 50 - High contrast text
    TEXT_SECONDARY: str = "#94A3B8"   # Slate 400 - Supporting labels
    TEXT_MUTED: str = "#64748B"       # Slate 500 - Grid / units / info
    TEXT_HIGHLIGHT: str = "#38BDF8"   # Sky blue highlight

    # Grid & Borders
    BORDER_SUBTLE: str = "rgba(59, 130, 246, 0.15)"
    BORDER_HOVER: str = "rgba(59, 130, 246, 0.35)"
    BORDER_SOLID: str = "#1E293B"
    GRID_COLOR: str = "#1E293B"
    SPINE_COLOR: str = "#334155"


DSP_THEME = DSPColors()

# Sequential scientific line palette
PLOT_PALETTE = [
    DSP_THEME.CYAN,
    DSP_THEME.PERSIAN_GREEN,
    DSP_THEME.BLUE_LIGHT,
    DSP_THEME.INDIGO_LINE,
    DSP_THEME.AMBER,
    DSP_THEME.VIOLET,
    DSP_THEME.ROSE,
]


def apply_scientific_plot_style(
    fig: Figure,
    axes: Optional[Axes | list[Axes]] = None,
    grid: bool = True,
    title_size: int = 10,
    label_size: int = 9,
    tick_size: int = 8,
):
    """
    Applies the SPANDHAN Dark Scientific styling to a Matplotlib Figure and Axes.
    Ensures crisp legibility, subtle borders, high contrast labels, and clean dark canvas.
    """
    fig.patch.set_facecolor(DSP_THEME.BG_PLOT)
    fig.patch.set_edgecolor("none")

    if axes is None:
        ax_list = fig.get_axes()
    elif isinstance(axes, list):
        ax_list = axes
    else:
        ax_list = [axes]

    for ax in ax_list:
        ax.set_facecolor(DSP_THEME.BG_PLOT)

        # Spines
        for spine_name, spine in ax.spines.items():
            spine.set_color(DSP_THEME.SPINE_COLOR)
            spine.set_linewidth(0.8)

        # Ticks
        ax.tick_params(
            axis="both",
            colors=DSP_THEME.TEXT_SECONDARY,
            labelsize=tick_size,
            direction="out",
            length=3.5,
            width=0.8,
            grid_color=DSP_THEME.GRID_COLOR,
            grid_alpha=0.6,
            grid_linewidth=0.6,
        )

        # Labels & Titles
        ax.xaxis.label.set_color(DSP_THEME.TEXT_SECONDARY)
        ax.xaxis.label.set_fontsize(label_size)
        ax.xaxis.label.set_fontweight("normal")

        ax.yaxis.label.set_color(DSP_THEME.TEXT_SECONDARY)
        ax.yaxis.label.set_fontsize(label_size)
        ax.yaxis.label.set_fontweight("normal")

        title = ax.get_title()
        if title:
            ax.title.set_color(DSP_THEME.TEXT_PRIMARY)
            ax.title.set_fontsize(title_size)
            ax.title.set_fontweight("bold")

        if grid:
            ax.grid(True, linestyle="--", linewidth=0.5, color=DSP_THEME.GRID_COLOR, alpha=0.7)

    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            fig.tight_layout()
    except Exception:
        pass


def get_chip_stylesheet(bg: str = "#13233E", text: str = "#F8FAFC", border: str = "#1E293B") -> str:
    """Returns a CSS snippet for small technical metadata chips."""
    return f"""
        QLabel {{
            background-color: {bg};
            color: {text};
            border: 1px solid {border};
            border-radius: 4px;
            padding: 2px 7px;
            font-size: 11px;
            font-family: 'Consolas', monospace;
            font-weight: 600;
        }}
    """
