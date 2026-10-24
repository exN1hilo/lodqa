"""Figures rebuilt from saved runs, without launching QAOA again."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {
    "dp": "#3a3a3a",
    "greedy": "#e07b00",
    "random": "#b0b0b0",
    "qaoa_x": "#1f6fd1",
    "qaoa_xy": "#7b3fc4",
}


def apply_style() -> None:
    """12 pt type, open top and right spines, light y-grid, 12.8 by 7.2 inch figures."""
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": "#e0e0e0",
            "grid.linewidth": 0.8,
            "figure.figsize": (12.8, 7.2),
            "axes.axisbelow": True,
            "legend.frameon": False,
        }
    )


def save_figure(fig: plt.Figure, name: str, out_dir: str | Path = "figures") -> list[Path]:
    """Write PNG at 300 dpi and SVG."""
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    png = directory / f"{name}.png"
    svg = directory / f"{name}.svg"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    return [png, svg]


def label_line_end(ax: plt.Axes, x: float, y: float, text: str, color: str) -> None:
    """Place a label at the end of a line instead of using a legend."""
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(8, 0),
        textcoords="offset points",
        color=color,
        va="center",
        ha="left",
        clip_on=False,
    )
