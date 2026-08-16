from __future__ import annotations

from pathlib import Path
import math
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from .style import (
    PROFILE_COLORS, MODEL_MARKERS, MODEL_DISPLAY, apply_publication_style,
    polish_axis, save_figure, model_display,
)
from .loaders import extract_cap_target


def _ordered(values, preferred):
    vals = [str(v) for v in values if str(v) not in {"", "nan", "NA"}]
    return [x for x in preferred if x in vals] + sorted(x for x in set(vals) if x not in preferred)


def _heatmap(ax, matrix: pd.DataFrame, title: str, fmt: str = ".2f", cmap="Blues", vmin=None, vmax=None):
    arr = matrix.to_numpy(dtype=float)
    im = ax.imshow(arr, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(matrix.columns)), matrix.columns)
    ax.set_yticks(range(len(matrix.index)), [model_display(x) for x in matrix.index])
    ax.set_title(title)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            if np.isfinite(v):
                ax.text(j, i, format(v, fmt), ha="center", va="center", fontsize=7,
                        color="white" if (vmax is not None and v > (vmax + (vmin or 0))/2) else "#222222")
    ax.tick_params(length=0)
    return im


def plot_prompt_robustness(log: pd.DataFrame, output_root: Path, dpi=600):
    apply_publication_style()
    d = log[log["experiment_id"] == "prompt_robustness"].copy()
    if d.empty:
        raise ValueError("No prompt_robustness records found.")
    models = _ordered(d["model_id"].unique(), list(MODEL_DISPLAY))
    policies = _ordered(d["policy_id"].unique(), ["P1", "P2", "P3", "P4", "P5", "P6"])
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))

    # Grouped boxplots of allocation deviation; smaller is better.
    width = 0.72 / max(1, len(models))
    handles = []
    for mi, model in enumerate(models):
        pos, data = [], []
        for pi, policy in enumerate(policies):
            vals = pd.to_numeric(d[(d.model_id == model) & (d.policy_id == policy)]["allocation_l1_to_reference"], errors="coerce").dropna().values
            if len(vals):
                pos.append(pi + (mi - (len(models)-1)/2) * width)
                data.append(vals)
        if data:
            bp = axes[0].boxplot(data, positions=pos, widths=width*0.82, patch_artist=True,
                                 showfliers=False, medianprops={"color": "#111111", "linewidth": 0.8},
                                 whiskerprops={"linewidth": 0.6}, capprops={"linewidth": 0.6}, boxprops={"linewidth": 0.6})
            color = plt.get_cmap("tab10")(mi % 10)
            for patch in bp["boxes"]:
                patch.set_facecolor(color); patch.set_alpha(0.58)
            handles.append(Line2D([0], [0], color=color, lw=5, alpha=0.58, label=model_display(model)))
    axes[0].set_xticks(range(len(policies)), policies)
    axes[0].set_ylabel(r"$L_1$ distance to policy reference")
    axes[0].set_title("(a) Allocation consistency across paraphrases")
    polish_axis(axes[0])
    axes[0].legend(handles=handles, frameon=False, ncol=2, loc="upper left")

    pivot = d.groupby(["model_id", "policy_id"])["projected_fidelity"].mean().unstack("policy_id")
    pivot = pivot.reindex(index=models, columns=policies)
    im = _heatmap(axes[1], pivot, "(b) Mean projected policy fidelity", fmt=".2f", cmap="Blues", vmin=0, vmax=1)
    cbar = fig.colorbar(im, ax=axes[1], fraction=0.045, pad=0.03)
    cbar.set_label("Fidelity rate")
    fig.subplots_adjust(left=0.09, right=0.965, top=0.90, bottom=0.16, wspace=0.32)
    return save_figure(fig, output_root, "fig_03_prompt_robustness", dpi=dpi)


def plot_ticker_masking(log: pd.DataFrame, output_root: Path, dpi=600):
    apply_publication_style()
    d = log[log["experiment_id"] == "ticker_masking"].copy()
    if d.empty:
        raise ValueError("No ticker_masking records found.")
    conditions = _ordered(d["mask_condition"].unique(), ["real_ticker", "masked_ticker", "shuffled_masked"])
    models = _ordered(d["model_id"].unique(), list(MODEL_DISPLAY))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))

    agg = d.groupby(["model_id", "mask_condition"])["projected_fidelity"].mean().reset_index()
    x = np.arange(len(models)); width = 0.72 / max(1, len(conditions))
    for ci, cond in enumerate(conditions):
        vals = [float(agg[(agg.model_id == m) & (agg.mask_condition == cond)]["projected_fidelity"].mean()) for m in models]
        axes[0].bar(x + (ci-(len(conditions)-1)/2)*width, vals, width=width*0.9, label=cond.replace("_", " "), alpha=0.78)
    axes[0].set_xticks(x, [model_display(m) for m in models], rotation=18, ha="right")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Projected fidelity")
    axes[0].set_title("(a) Feature-grounded policy fidelity")
    axes[0].legend(frameon=False, fontsize=6.5)
    polish_axis(axes[0])

    pair = d.groupby(["model_id", "policy_id", "mask_condition"])["allocation_l1_to_reference"].mean().unstack("mask_condition")
    if "real_ticker" not in pair or "masked_ticker" not in pair:
        raise ValueError("Ticker masking plot requires real_ticker and masked_ticker conditions.")
    for model in models:
        sub = pair.loc[model] if model in pair.index.get_level_values(0) else pd.DataFrame()
        if isinstance(sub, pd.Series): sub = sub.to_frame().T
        if len(sub):
            axes[1].scatter(sub["real_ticker"], sub["masked_ticker"], marker=MODEL_MARKERS.get(model, "o"),
                            s=34, alpha=0.80, label=model_display(model))
    limmax = float(np.nanmax([pair["real_ticker"].max(), pair["masked_ticker"].max(), 0.1])) * 1.08
    axes[1].plot([0, limmax], [0, limmax], color="#777777", linestyle="--", linewidth=0.9)
    axes[1].set_xlim(0, limmax); axes[1].set_ylim(0, limmax)
    axes[1].set_xlabel("Real-ticker allocation deviation")
    axes[1].set_ylabel("Masked-ticker allocation deviation")
    axes[1].set_title("(b) Paired deviation from policy reference")
    axes[1].legend(frameon=False, fontsize=6.5)
    polish_axis(axes[1])
    fig.subplots_adjust(left=0.09, right=0.985, top=0.90, bottom=0.22, wspace=0.32)
    return save_figure(fig, output_root, "fig_04_ticker_masking", dpi=dpi)


def plot_policy_complexity(log: pd.DataFrame, output_root: Path, dpi=600):
    apply_publication_style()
    d = log[log["experiment_id"] == "policy_complexity"].copy()
    if d.empty:
        raise ValueError("No policy_complexity records found.")
    levels = _ordered(d["complexity_level"].unique(), ["L1", "L2", "L3", "L4", "L5", "L6"])
    models = _ordered(d["model_id"].unique(), list(MODEL_DISPLAY))
    agg = d.groupby(["model_id", "complexity_level"]).agg(
        fidelity=("projected_fidelity", "mean"),
        collapse=("collapse_flag", "mean"),
        repair=("repair_used", "mean"),
    ).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharex=True)
    for mi, model in enumerate(models):
        sub = agg[agg.model_id == model].set_index("complexity_level").reindex(levels)
        color = plt.get_cmap("tab10")(mi % 10)
        axes[0].plot(levels, sub["fidelity"], marker=MODEL_MARKERS.get(model, "o"), color=color, label=model_display(model))
        axes[1].plot(levels, sub["collapse"], marker=MODEL_MARKERS.get(model, "o"), color=color, label=model_display(model))
    axes[0].set_ylim(0, 1.05); axes[1].set_ylim(0, 1.05)
    axes[0].set_ylabel("Projected fidelity")
    axes[1].set_ylabel("Equal-weight collapse rate")
    axes[0].set_title("(a) Policy fidelity by complexity")
    axes[1].set_title("(b) Collapse risk by complexity")
    for ax in axes:
        ax.set_xlabel("Policy-complexity level")
        polish_axis(ax)
    axes[0].legend(frameon=False, fontsize=6.5, ncol=2)
    fig.subplots_adjust(left=0.09, right=0.985, top=0.90, bottom=0.18, wspace=0.30)
    return save_figure(fig, output_root, "fig_05_policy_complexity_ladder", dpi=dpi)


def plot_constraint_conflict(log: pd.DataFrame, output_root: Path, dpi=600):
    apply_publication_style()
    d = log[log["experiment_id"] == "constraint_conflict_stress"].copy()
    if d.empty:
        raise ValueError("No constraint_conflict_stress records found.")
    d["language_target"] = d["conflict_type"].map(extract_cap_target)
    cap = d[d["language_target"].notna()].copy()
    if cap.empty:
        raise ValueError("No 60/70/90 cap-target conflict records found.")
    agg = cap.groupby("language_target").agg(
        raw_max=("raw_max_weight", "mean"),
        projected_max=("post_max_weight", "mean"),
        projection_l1=("projection_l1", "mean"),
        feasible_cap=("post_feasible_cap", "mean"),
    ).reset_index().sort_values("language_target")
    x = np.arange(len(agg)); labels = [f"{int(v*100)}%" for v in agg.language_target]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    axes[0].bar(x-0.18, agg.raw_max, width=0.36, label="Raw proposal", alpha=0.78)
    axes[0].bar(x+0.18, agg.projected_max, width=0.36, label="After projection", alpha=0.78)
    axes[0].axhline(0.60, color="#222222", linestyle="--", linewidth=1.0, label="Hard cap")
    axes[0].set_xticks(x, labels); axes[0].set_ylim(0, max(1.0, float(agg.raw_max.max())*1.08))
    axes[0].set_xlabel("Language-level target")
    axes[0].set_ylabel("Maximum asset weight")
    axes[0].set_title("(a) Raw versus feasible concentration")
    axes[0].legend(frameon=False, fontsize=6.5)
    polish_axis(axes[0])

    axes[1].plot(labels, agg.projection_l1, marker="o", linewidth=1.6, label=r"Mean projection $L_1$")
    ax2 = axes[1].twinx()
    ax2.plot(labels, agg.feasible_cap, marker="s", linestyle="--", color="#D55E00", label="Post-projection cap feasibility")
    axes[1].set_xlabel("Language-level target")
    axes[1].set_ylabel(r"Projection $L_1$ distance")
    ax2.set_ylabel("Feasible-cap rate")
    ax2.set_ylim(0, 1.05)
    axes[1].set_title("(b) Intervention magnitude and safety")
    polish_axis(axes[1]); ax2.spines["top"].set_visible(False)
    lines = axes[1].lines + ax2.lines
    axes[1].legend(lines, [l.get_label() for l in lines], frameon=False, fontsize=6.5, loc="best")
    fig.subplots_adjust(left=0.09, right=0.91, top=0.90, bottom=0.18, wspace=0.38)
    return save_figure(fig, output_root, "fig_06_constraint_conflict", dpi=dpi)


def plot_model_family_generalization(log: pd.DataFrame, output_root: Path, dpi=600):
    apply_publication_style()
    d = log[log["experiment_id"] == "model_family_generalization"].copy()
    if d.empty:
        raise ValueError("No model_family_generalization records found.")
    models = _ordered(d["model_id"].unique(), list(MODEL_DISPLAY))
    agg = d.groupby("model_id").agg(
        fidelity=("projected_fidelity", "mean"),
        json_valid=("json_valid", "mean"),
        collapse=("collapse_flag", "mean"),
        missing=("missing_asset_count", "mean"),
        projection_l1=("projection_l1", "mean"),
    ).reindex(models)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    x = np.arange(len(models)); width=0.24
    axes[0].bar(x-width, agg.fidelity, width=width, label="Policy fidelity", alpha=0.82)
    axes[0].bar(x, agg.json_valid, width=width, label="JSON validity", alpha=0.82)
    axes[0].bar(x+width, 1-agg.collapse, width=width, label="Non-collapse rate", alpha=0.82)
    axes[0].set_ylim(0, 1.05); axes[0].set_xticks(x, [model_display(m) for m in models], rotation=18, ha="right")
    axes[0].set_ylabel("Rate")
    axes[0].set_title("(a) Cross-family reliability rates")
    axes[0].legend(frameon=False, fontsize=6.5)
    polish_axis(axes[0])

    ax2 = axes[1]
    ax2.bar(x-0.18, agg.projection_l1, width=0.36, label=r"Projection $L_1$", alpha=0.80)
    ax2b = ax2.twinx()
    ax2b.bar(x+0.18, agg.missing, width=0.36, label="Missing assets/call", color="#D55E00", alpha=0.65)
    ax2.set_xticks(x, [model_display(m) for m in models], rotation=18, ha="right")
    ax2.set_ylabel(r"Mean projection $L_1$")
    ax2b.set_ylabel("Mean missing assets")
    ax2.set_title("(b) Projection and completeness diagnostics")
    polish_axis(ax2); ax2b.spines["top"].set_visible(False)
    handles = [axes[1].patches[0], ax2b.patches[0]] if axes[1].patches and ax2b.patches else []
    if handles: axes[1].legend(handles, [r"Projection $L_1$", "Missing assets/call"], frameon=False, fontsize=6.5)
    fig.subplots_adjust(left=0.09, right=0.91, top=0.90, bottom=0.24, wspace=0.38)
    return save_figure(fig, output_root, "fig_07_model_family_generalization", dpi=dpi)


def plot_fidelity_projection_heatmap(log: pd.DataFrame, output_root: Path, dpi=600):
    apply_publication_style()
    d = log[log["experiment_id"] == "model_family_generalization"].copy()
    if d.empty:
        raise ValueError("No model_family_generalization records found.")
    models = _ordered(d["model_id"].unique(), list(MODEL_DISPLAY))
    policies = _ordered(d["policy_id"].unique(), ["P1", "P2", "P3", "P4", "P5", "P6"])
    fid = d.groupby(["model_id", "policy_id"])["raw_fidelity"].mean().unstack().reindex(index=models, columns=policies)
    proj = d.groupby(["model_id", "policy_id"])["projection_l1"].mean().unstack().reindex(index=models, columns=policies)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15))
    im1 = _heatmap(axes[0], fid, "(a) Raw policy fidelity", fmt=".2f", cmap="Blues", vmin=0, vmax=1)
    vmax = float(np.nanmax(proj.to_numpy())) if np.isfinite(proj.to_numpy()).any() else 1.0
    im2 = _heatmap(axes[1], proj, r"(b) Mean projection $L_1$ distance", fmt=".2f", cmap="Oranges", vmin=0, vmax=max(vmax, 1e-6))
    c1 = fig.colorbar(im1, ax=axes[0], fraction=0.045, pad=0.03); c1.set_label("Fidelity rate")
    c2 = fig.colorbar(im2, ax=axes[1], fraction=0.045, pad=0.03); c2.set_label(r"Projection $L_1$")
    fig.subplots_adjust(left=0.12, right=0.96, top=0.90, bottom=0.16, wspace=0.38)
    return save_figure(fig, output_root, "fig_08_fidelity_projection_heatmap", dpi=dpi)


def _parameter_size(model: str) -> float:
    s = str(model).lower()
    patterns = [(r"270m", 0.27), (r"1b", 1.0), (r"4b", 4.0), (r"8b", 8.0), (r"20b", 20.0)]
    for pat, val in patterns:
        if re.search(pat, s): return val
    return 1.0


def plot_latency_reliability(log: pd.DataFrame, output_root: Path, dpi=600):
    apply_publication_style()
    d = log[log["experiment_id"] == "model_family_generalization"].copy()
    if d.empty:
        d = log.copy()
    models = _ordered(d["model_id"].unique(), list(MODEL_DISPLAY))
    agg = d.groupby("model_id").agg(
        median_latency=("latency_sec", "median"),
        p95_latency=("latency_sec", lambda x: pd.to_numeric(x, errors="coerce").quantile(0.95)),
        fidelity=("projected_fidelity", "mean"),
        json_valid=("json_valid", "mean"),
    ).reindex(models)
    fig, ax = plt.subplots(figsize=(5.0, 3.5))
    for i, model in enumerate(models):
        row = agg.loc[model]
        x = float(row.median_latency); y = float(row.fidelity)
        xerr = max(0.0, float(row.p95_latency) - x)
        size = 42 + 28 * math.sqrt(_parameter_size(model))
        color = plt.get_cmap("tab10")(i % 10)
        ax.errorbar(x, y, xerr=[[0.0], [xerr]], fmt=MODEL_MARKERS.get(model, "o"), markersize=math.sqrt(size),
                    color=color, ecolor=color, capsize=2.5, linewidth=0.9, alpha=0.88)
        ax.annotate(f"{model_display(model)}\nJSON {row.json_valid:.0%}", (x, y), xytext=(5, 5), textcoords="offset points", fontsize=6.8)
    ax.set_xlabel("Median local-inference latency (seconds)")
    ax.set_ylabel("Projected policy fidelity")
    ax.set_ylim(0, 1.05)
    ax.set_title("Latency–reliability trade-off (error bar extends to P95 latency)")
    polish_axis(ax)
    fig.subplots_adjust(left=0.15, right=0.96, top=0.88, bottom=0.16)
    return save_figure(fig, output_root, "fig_09_latency_reliability", dpi=dpi)
