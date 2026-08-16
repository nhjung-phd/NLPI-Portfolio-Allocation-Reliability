from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from .style import (
    PROFILE_COLORS, MODEL_LINESTYLES, MODEL_DISPLAY, BENCHMARK_STYLES,
    apply_publication_style, polish_axis, save_figure, model_display,
)
from .loaders import MAIN_MODELS, MAIN_POLICIES, MAIN_BENCHMARKS


def _plot_series(ax, g: pd.DataFrame, **kwargs) -> None:
    g = g.sort_values("date")
    ax.plot(g["date"], g["equity"], **kwargs)


def plot_main_oos_equity(oos: pd.DataFrame, output_root: Path, dpi: int = 600) -> dict[str, str]:
    apply_publication_style()
    stitched = oos[oos["segment"] == "stitched_oos"].copy()
    fig, ax = plt.subplots(figsize=(7.2, 3.8), constrained_layout=False)

    # Benchmarks remain visible but visually secondary.
    for bench in MAIN_BENCHMARKS:
        g = stitched[stitched["series_key"] == bench]
        if g.empty:
            continue
        st = BENCHMARK_STYLES.get(bench, {"color": "#777777", "linestyle": "--", "linewidth": 1.2})
        _plot_series(ax, g, label=bench, alpha=0.82, zorder=2, **st)

    # NLPI: profile determines color, model determines line style. P4/P5 controls are thinner.
    for model in MAIN_MODELS:
        for profile in MAIN_POLICIES:
            key = f"NLPI[{model}|{profile}]"
            g = stitched[stitched["series_key"] == key]
            if g.empty:
                continue
            _plot_series(
                ax, g,
                color=PROFILE_COLORS[profile],
                linestyle=MODEL_LINESTYLES.get(model, "-"),
                linewidth=1.55 if profile in {"P1", "P2", "P3"} else 1.0,
                alpha=0.88 if profile in {"P1", "P2", "P3"} else 0.45,
                zorder=3,
            )

    ax.set_title("Stitched out-of-sample portfolio equity")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity (initial = 1)")
    polish_axis(ax)
    ax.axhline(1.0, color="#BBBBBB", linewidth=0.65, zorder=1)

    benchmark_handles = [
        Line2D([0], [0], label=b, **BENCHMARK_STYLES[b])
        for b in MAIN_BENCHMARKS if b in set(stitched["series_key"])
    ]
    profile_handles = [Line2D([0], [0], color=PROFILE_COLORS[p], lw=1.7, label=p) for p in MAIN_POLICIES]
    model_handles = [
        Line2D([0], [0], color="#222222", linestyle=MODEL_LINESTYLES.get(m, "-"), lw=1.5, label=model_display(m))
        for m in MAIN_MODELS
    ]
    leg1 = ax.legend(handles=benchmark_handles, title="Benchmarks", loc="upper left", frameon=False, ncol=min(5, len(benchmark_handles)))
    ax.add_artist(leg1)
    leg2 = ax.legend(handles=profile_handles + model_handles, title="NLPI encoding", loc="lower center",
                     bbox_to_anchor=(0.5, -0.34), frameon=False, ncol=4, columnspacing=1.2, handlelength=2.3)
    fig.subplots_adjust(left=0.09, right=0.985, top=0.91, bottom=0.27)
    return save_figure(fig, output_root, "fig_01_main_oos_equity", dpi=dpi)


def plot_per_model_wfcv(oos: pd.DataFrame, output_root: Path, dpi: int = 600) -> dict[str, str]:
    apply_publication_style()
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 8.0), sharex=True, constrained_layout=False)
    stitched = oos[oos["segment"] == "stitched_oos"].copy()
    folds = oos[oos["segment"] == "fold_oos"].copy()

    for idx, (ax, model) in enumerate(zip(axes, MAIN_MODELS)):
        # Common benchmarks in subtle gray.
        for bench in MAIN_BENCHMARKS:
            g = stitched[stitched["series_key"] == bench]
            if not g.empty:
                st = BENCHMARK_STYLES.get(bench, {})
                _plot_series(ax, g, color=st.get("color", "#999999"), linestyle=st.get("linestyle", "--"),
                             linewidth=0.9, alpha=0.42, zorder=1)

        # Correct model-specific filtering: P1-P5 are selected from the model column.
        for profile in MAIN_POLICIES:
            key = f"NLPI[{model}|{profile}]"
            gf = folds[folds["series_key"] == key]
            for _, fold_g in gf.groupby("fold"):
                _plot_series(ax, fold_g, color=PROFILE_COLORS[profile], linewidth=0.55, alpha=0.13, zorder=2)
            gs = stitched[stitched["series_key"] == key]
            if not gs.empty:
                _plot_series(ax, gs, color=PROFILE_COLORS[profile], linewidth=1.7, alpha=0.98, zorder=3, label=profile)

        ax.set_title(f"({chr(97 + idx)}) {model_display(model)}")
        ax.set_ylabel("Equity")
        polish_axis(ax)
        ax.axhline(1.0, color="#BBBBBB", linewidth=0.6)

    axes[-1].set_xlabel("Date")
    profile_handles = [Line2D([0], [0], color=PROFILE_COLORS[p], lw=1.8, label=p) for p in MAIN_POLICIES]
    benchmark_handle = Line2D([0], [0], color="#777777", lw=1.0, linestyle="--", alpha=0.55, label="Benchmarks")
    fig.legend(handles=profile_handles + [benchmark_handle], loc="lower center", ncol=6, frameon=False,
               bbox_to_anchor=(0.5, 0.015), columnspacing=1.7, handlelength=2.6)
    fig.suptitle("Model-specific WFCV out-of-sample equity: fold paths (thin) and stitched path (bold)", y=0.985, fontsize=10)
    fig.subplots_adjust(left=0.09, right=0.985, top=0.94, bottom=0.085, hspace=0.30)
    return save_figure(fig, output_root, "fig_02_per_model_wfcv", dpi=dpi)
