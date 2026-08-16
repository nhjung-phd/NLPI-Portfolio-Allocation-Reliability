from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable
import matplotlib as mpl
import matplotlib.pyplot as plt

# Okabe-Ito-inspired, color-blind-aware palette.
PROFILE_COLORS: Dict[str, str] = {
    "P1": "#D55E00",  # vermillion
    "P2": "#0072B2",  # blue
    "P3": "#009E73",  # bluish green
    "P4": "#CC79A7",  # reddish purple
    "P5": "#E69F00",  # orange
    "P6": "#56B4E9",  # sky blue
}
MODEL_LINESTYLES: Dict[str, str] = {
    "gemma3:270m": "-",
    "gemma3:1b": "--",
    "llama3.1:8b": "-.",
    "qwen3.5:4b": ":",
}
MODEL_MARKERS: Dict[str, str] = {
    "gemma3:270m": "o",
    "gemma3:1b": "s",
    "llama3.1:8b": "^",
    "qwen3.5:4b": "D",
}
MODEL_DISPLAY: Dict[str, str] = {
    "gemma3:270m": "Gemma 3 270M",
    "gemma3:1b": "Gemma 3 1B",
    "llama3.1:8b": "Llama 3.1 8B",
    "qwen3.5:4b": "Qwen 3.5 4B",
}
BENCHMARK_STYLES = {
    "EQUAL": {"color": "#222222", "linestyle": "-", "linewidth": 1.45},
    "RiskParity": {"color": "#666666", "linestyle": "--", "linewidth": 1.35},
    "MVP": {"color": "#8A8A8A", "linestyle": ":", "linewidth": 1.35},
    "MOM6": {"color": "#444444", "linestyle": "-.", "linewidth": 1.35},
    "SHARPE": {"color": "#AAAAAA", "linestyle": (0, (5, 2)), "linewidth": 1.35},
}


def apply_publication_style() -> None:
    """Apply a deterministic journal-style Matplotlib configuration."""
    serif = ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"]
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": serif,
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.0,
        "axes.linewidth": 0.75,
        "lines.linewidth": 1.35,
        "grid.color": "#D9D9D9",
        "grid.linewidth": 0.45,
        "grid.alpha": 0.55,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.transparent": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def polish_axis(ax, *, grid: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#444444")
    ax.spines["bottom"].set_color("#444444")
    ax.tick_params(direction="out", length=3, width=0.6, colors="#333333")
    if grid:
        ax.grid(True, axis="y", zorder=0)
        ax.grid(False, axis="x")


def save_figure(fig, output_root: Path, stem: str, *, dpi: int = 600) -> dict[str, str]:
    """Save the same deterministic figure as PNG, PDF, and SVG."""
    output_root = Path(output_root)
    paths: dict[str, str] = {}
    for ext in ("png", "pdf", "svg"):
        folder = output_root / ext
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{stem}.{ext}"
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.04}
        if ext == "png":
            kwargs["dpi"] = int(dpi)
        fig.savefig(path, **kwargs)
        paths[ext] = str(path)
    plt.close(fig)
    return paths


def model_display(model: str) -> str:
    return MODEL_DISPLAY.get(str(model), str(model))
