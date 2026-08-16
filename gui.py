#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NLPI Portfolio Allocation Studio (Tkinter) - Policy Interface + Constraint Projection
- Adds Stars column to Significance table and working recompute flow
- Restores WFCV plotting (fold overlays + stitched OOS) and table
- Throttles noisy "Rebalanced on ..." logs in Log tab
"""

from __future__ import annotations

import os
import io
import queue
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import re
import sys
import subprocess
import glob

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

matplotlib.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings("ignore", message="Degrees of freedom <= 0 for slice")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")

# Persona tags for NLPI prompt profiles (used in NLPI prompt profile descriptions)
PERSONA_TAGS = {
    1: "AGGRESSIVE MOMENTUM policy profile",
    2: "DEFENSIVE LOW-VOLATILITY policy profile",
    3: "CONTRARIAN MEAN-REVERSION policy profile",
    4: "EQUAL-WEIGHT CONTROL policy profile",
    5: "RISK-ADJUSTED RETURN policy profile",
}

# ---- Project imports ----
from portfolios import PORTFOLIOS, DEFAULT_PORTFOLIO_NAME, DEFAULT_TICKERS
from core import fetch_prices_yf, make_features, split_index
from llm import (
    check_ollama, build_fewshot_db, render_fewshot_block, PROMPT_PROFILES
)
from engine import run_backtest
from engine.metrics import summary
from engine.registry import build_strategies  # ★ registry 사용
from paper_canonical import (
    PAPER_PROTOCOL, PAPER_TICKERS, PAPER_MAIN_MODELS, PAPER_ROBUSTNESS_MODELS,
    PAPER_BENCHMARKS, PAPER_CODED, create_run_directory, write_initial_manifest,
    update_manifest, validate_protocol_values, finalize_run,
)
from paper_figures.generate_all import generate_all_figures

# Optional significance module
try:
    from engine.statsig import build_comparison_table as build_significance_table
except Exception:
    build_significance_table = None

# ---- Metrics columns ----
COL_DEF = [
    ("name",   "Strategy",              160),
    ("Sharpe", "Sharpe Ratio",          110),
    ("Sortino","Sortino Ratio",         110),
    ("CUM",    "Cumulative Return",     130),
    ("ANN",    "Annualized Return",     130),
    ("Vol",    "Annualized Volatility", 150),
    ("CAGR",   "CAGR",                  90),
    ("MDD",    "Maximum Drawdown",      140),
    ("TO",     "Turnover Ratio",        110),
]
COL_KEYS  = [k for k,_,_ in COL_DEF]
COL_LABEL = {k:lbl for k,lbl,_ in COL_DEF}
COL_WIDTH = {k:w   for k,_,w in COL_DEF}

MSG_INFO = "info"
MSG_DONE = "done"
MSG_ERROR= "error"
MSG_CANCELED = "canceled"

# ------------------------------ App ------------------------------

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("NLPI Portfolio Allocation Studio — GUI")
        self.geometry("2100x1200")

        # Inputs
        self.var_tickers = tk.StringVar(value=",".join(DEFAULT_TICKERS))
        self.var_selected_port = tk.StringVar(value=DEFAULT_PORTFOLIO_NAME)
        self.var_start   = tk.StringVar(value="2006-01-01")
        self.var_end     = tk.StringVar(value=str(date.today()))
        self.var_reb     = tk.IntVar(value=42)
        self.var_tcost   = tk.DoubleVar(value=0.0010)
        self.var_maxw    = tk.DoubleVar(value=0.60)
        self.var_turn    = tk.DoubleVar(value=0.25)
        self.var_use_llm = tk.BooleanVar(value=True)
        self.var_prog_stride = tk.IntVar(value=5)
        self.var_prompt_level = tk.IntVar(value=1)
        self.var_wfcv = tk.BooleanVar(value=True)
        self.var_holdout = tk.BooleanVar(value=False)  # NEW: Train/Test holdout split (for future ML)
        self.var_model_name = tk.StringVar(value=os.getenv("OLLAMA_MODEL", "gpt-oss:20b"))
        
        # --- Logging controls ---
        self.var_log_level = tk.StringVar(value="ERROR")
        self.var_gui_log_level = tk.StringVar(value="ERROR")
        self.var_log_every = tk.StringVar(value="0")
        self._allow_rebalance_spam = False

        # Significance baseline selector
        self.var_sig_base = tk.StringVar(value="EQUAL")

        # Ollama models
        self._ollama_models: List[str] = []

        # Infra
        self.exec = ThreadPoolExecutor(max_workers=1)
        self.msg_q: "queue.Queue[tuple[str,object]]" = queue.Queue()
        self.worker_future = None
        self.cancel_event = threading.Event()

        # Data caches
        self._last_prices: pd.DataFrame | None = None
        self._fewshot_text: str = ""
        self._last_res_test: Dict[str, pd.Series] = {}
        self._last_metrics: Dict[str, dict] = {}

        # benchmark selection model (name->bool)
        self._bench_defs = [
            ("EQUAL",     "Equal Weight"),
            ("RiskParity","Risk Parity"),
            ("MVP",       "Min-Variance"),
            ("LW-MVP",    "Ledoit–Wolf MVP"),
            ("HRP",       "Hierarchical Risk Parity"),
            ("BL",        "Black–Litterman"),
            ("MOM6",     "Momentum 6m"),
            ("TRND6",    "Trend 6m"),
            ("SHARPE",    "Sharpe-Weighted (60d)"),
            ("SORTINO",   "Sortino-Weighted (60d)"),
            ("CODED_P1",  "Coded Persona P1 (no LLM)"),
            ("CODED_P2",  "Coded Persona P2 (no LLM)"),
            ("CODED_P3",  "Coded Persona P3 (no LLM)"),
            ("CODED_P4",  "Coded Persona P4 (no LLM)"),
            ("CODED_P5",  "Coded Persona P5 (no LLM)"),
            ("NLPI",      "NLPI: LLM-based Natural-Language Policy Interface"),
        ]
        self._bench_vars: Dict[str, tk.BooleanVar] = {k: tk.BooleanVar(value=True) for k,_ in self._bench_defs}

        # WFCV state
        self._wfcv_overlay: Dict[str, List[Tuple[pd.DatetimeIndex, np.ndarray]]] = {}
        self._wfcv_stitched: Dict[str, Tuple[pd.DatetimeIndex, np.ndarray]] = {}
        self._last_wfcv: dict = {}
        self._last_wfcv_info: dict = {}
        
        # Per-WFCV (per-model)
        self._per_wfcv_selected_model = tk.StringVar(value="")
        self._per_wfcv_include_bench = tk.BooleanVar(value=True)
        
        # OOS export controls
        self.var_oos_segment = tk.StringVar(value="stitched_oos")
        self.var_oos_fold_sel = tk.StringVar(value="All")
        self.var_oos_show_fold = tk.BooleanVar(value=False)
        self.var_oos_include_bench = tk.BooleanVar(value=True)

        # GUI one-click export state
        self._last_payload: dict = {}
        self._last_tech_stats_df: pd.DataFrame | None = None
        self._auto_export_outdir: str | None = None
        self._auto_export_after_run: bool = False

        # Q1/Q2 reliability experiment package controls
        self.var_q2_mode = tk.StringVar(value="q2")
        self.var_q2_extra_models = tk.StringVar(value="qwen3.5:4b")
        self.var_q2_out_base = tk.StringVar(value=os.path.join(os.getcwd(), "outputs", "q2_gui_reliability"))
        self.var_q2_n_per_regime = tk.IntVar(value=6)
        self.var_q2_max_calls = tk.StringVar(value="")
        self.var_q2_use_extra = tk.BooleanVar(value=True)
        self._q2_proc = None
        self._q2_last_outdir = None

        # Canonical paper-run controls. These settings are locked to the
        # manuscript protocol and orchestrate main + coded + reliability runs.
        self.var_paper_output_base = tk.StringVar(value=os.path.join(os.getcwd(), "outputs"))
        self.var_paper_include_reliability = tk.BooleanVar(value=True)
        self.var_paper_include_qwen = tk.BooleanVar(value=True)
        self.var_paper_n_per_regime = tk.IntVar(value=int(PAPER_PROTOCOL.get("n_per_regime", 6)))
        self._paper_running = False
        self._paper_stage = "idle"
        self._paper_root = None
        self._paper_proc = None
        self.var_paper_status = tk.StringVar(value="Idle")
        self.var_paper_figure_dpi = tk.IntVar(value=600)
        self._paper_figure_thread = None

        self._build_ui()
        self._bench_desc_labels = {}
        self._probe_ollama_on_start()
        self.after(100, self._pump)
        
    def _legend_outside(self, ax, fig, right=0.78, left=0.08, bottom=0.12, top=0.92):
        leg = ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0.0,
            frameon=False
        )
        # 좌/우/아래/위 여백을 명시적으로 고정
        fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
        return leg



    # -------------------------- UI build --------------------------
    def _build_ui(self):
        top = ttk.Frame(self, padding=10); top.pack(side="top", fill="x")
        r = 0
        ttk.Label(top, text="Selected Portfolio:").grid(row=r, column=0, sticky="e")
        ttk.Label(top, textvariable=self.var_selected_port, foreground="#036").grid(row=r, column=1, sticky="w")
        ttk.Label(top, text="Tickers:").grid(row=r, column=2, sticky="e")
        ttk.Entry(top, textvariable=self.var_tickers, width=60).grid(row=r, column=3, columnspan=3, sticky="we")
        r += 1
        
        ttk.Label(top, text="Start:").grid(row=r, column=0, sticky="e")
        # Keep explicit Entry handles so we always read the latest text even if
        # the user hasn't defocused the widget (prevents stale start/end on first click).
        self.ent_start = ttk.Entry(top, textvariable=self.var_start, width=12)
        self.ent_start.grid(row=r, column=1, sticky="w")

        ttk.Label(top, text="End:").grid(row=r, column=2, sticky="e")
        self.ent_end = ttk.Entry(top, textvariable=self.var_end, width=12)
        self.ent_end.grid(row=r, column=3, sticky="w")

        
        ttk.Label(top, text="Rebalance days:").grid(row=r, column=4, sticky="e")
        ttk.Entry(top, textvariable=self.var_reb, width=8).grid(row=r, column=5, sticky="w")
        r += 1
        ttk.Label(top, text="Tcost:").grid(row=r, column=0, sticky="e")
        ttk.Entry(top, textvariable=self.var_tcost, width=8).grid(row=r, column=1, sticky="w")
        ttk.Label(top, text="Max weight:").grid(row=r, column=2, sticky="e")
        ttk.Entry(top, textvariable=self.var_maxw, width=8).grid(row=r, column=3, sticky="w")
        ttk.Label(top, text="Turnover cap:").grid(row=r, column=4, sticky="e")
        ttk.Entry(top, textvariable=self.var_turn, width=8).grid(row=r, column=5, sticky="w")
        r += 1
        ttk.Checkbutton(top, text="Use NLPI (LLM backend)", variable=self.var_use_llm).grid(row=r, column=0, sticky="w")
        ttk.Label(top, text="Model:").grid(row=r, column=1, sticky="e")
        self.cbo_model = ttk.Combobox(top, textvariable=self.var_model_name, width=22, state="readonly", values=[])
        self.cbo_model.grid(row=r, column=2, sticky="w")
        # self.cbo_model.bind("<<ComboboxSelected>>", lambda e: self._rebuild_llm_variants_for_selected_model())
        stride_box = ttk.Frame(top); stride_box.grid(row=r, column=3, sticky="w", padx=(10,0))
        ttk.Label(stride_box, text="Progress stride:").pack(side="left")
        ttk.Entry(stride_box, textvariable=self.var_prog_stride, width=6).pack(side="left", padx=(6,0))
        ttk.Checkbutton(top, text="Walk-forward CV", variable=self.var_wfcv).grid(row=r, column=4, sticky="w", padx=(10,0))
        ttk.Checkbutton(top, text="Holdout Train/Test", variable=self.var_holdout).grid(row=r, column=5, sticky="w", padx=(10,0))
        btns = ttk.Frame(top); btns.grid(row=r, column=5, sticky="e")
        ttk.Button(btns, text="Run", width=10, command=self.on_run).pack(side="left", padx=(0,6))
        ttk.Button(btns, text="Run Main+Export", width=16, command=self.on_run_main_nlpi_export).pack(side="left", padx=(0,6))
        ttk.Button(btns, text="Stop", width=10, command=self.on_stop).pack(side="left")
        r += 1
        status = ttk.Frame(self, padding=(10,0)); status.pack(fill="x")
        self.lbl_status = ttk.Label(status, text="Ready.")
        self.lbl_status.pack(side="left")
        self.lbl_ollama = ttk.Label(status, text="Ollama: checking...", foreground="#777")
        self.lbl_ollama.pack(side="left", padx=12)
        ttk.Button(status, text="Ollama Connect", command=self.on_ollama_connect).pack(side="right", padx=8)
        self.pb = ttk.Progressbar(status, mode="indeterminate", length=240)
        self.pb.pack(side="right")

        # Notebook
        self.nb = ttk.Notebook(self); self.nb.pack(fill="both", expand=True, padx=10, pady=10)
        # 1) Portfolio
        self.tab_port = ttk.Frame(self.nb); self.nb.add(self.tab_port, text="Portfolio")
        self._build_portfolio_tab(self.tab_port)
        # 2) Benchmarks (selection)
        self.tab_bench = ttk.Frame(self.nb); self.nb.add(self.tab_bench, text="Benchmarks")
        self._build_benchmarks_tab(self.tab_bench)
        # 3) Tech Stats
        self.tab_stats = ttk.Frame(self.nb); self.nb.add(self.tab_stats, text="Tech Stats")
        self._build_stats_tab(self.tab_stats)
        # 4) Train
        self.tab_train = ttk.Frame(self.nb); self.nb.add(self.tab_train, text="Train")
        self.fig_train, self.ax_train, self.canvas_train = self._create_plot_canvas(self.tab_train)
        self._add_plot_copy_buttons(self.tab_train, self.canvas_train)
        # 5) Test
        self.tab_test = ttk.Frame(self.nb); self.nb.add(self.tab_test, text="Test")
        self.fig_test, self.ax_test, self.canvas_test = self._create_plot_canvas(self.tab_test)
        self._add_plot_copy_buttons(self.tab_test, self.canvas_test)
        # 6) WFCV
        self.tab_wfcv = ttk.Frame(self.nb); self.nb.add(self.tab_wfcv, text="WFCV")
        self._build_wfcv_tab(self.tab_wfcv)
        # 7) Metrics
        self.tab_metrics = ttk.Frame(self.nb); self.nb.add(self.tab_metrics, text="Metrics")
        self._build_metrics_tab(self.tab_metrics)
        # 8) Significance
        self.tab_sig = ttk.Frame(self.nb); self.nb.add(self.tab_sig, text="Significance")
        self._build_sig_tab(self.tab_sig)
        # 9) Few-shot
        self.tab_few = ttk.Frame(self.nb); self.nb.add(self.tab_few, text="Few-shot")
        self._build_fewshot_tab(self.tab_few)
        # 10) Prompts
        self.tab_prompts = ttk.Frame(self.nb); self.nb.add(self.tab_prompts, text="Prompts")
        self._build_prompts_tab(self.tab_prompts)
        # 11) Log
        self.tab_log = ttk.Frame(self.nb); self.nb.add(self.tab_log, text="Log")
        self._build_log_tab(self.tab_log)
        self._log("GUI initialized.")
        for c in range(6): top.columnconfigure(c, weight=1)
        # 12) All Models
        self.tab_matrix = ttk.Frame(self.nb); self.nb.add(self.tab_matrix, text="All Models")
        self._build_matrix_tab(self.tab_matrix)
        # 13) Per-Model Reports
        self.tab_model = ttk.Frame(self.nb); self.nb.add(self.tab_model, text="Per-Model")
        self._build_model_reports_tab(self.tab_model)
        # 14) Per-WFCV
        self.tab_wfcv_model = ttk.Frame(self.nb); self.nb.add(self.tab_wfcv_model, text="Per-WFCV")
        self._build_wfcv_model_tab(self.tab_wfcv_model)
        # 15) OOS Export (tidy CSV + OOS-only plot)
        self.tab_oos = ttk.Frame(self.nb); self.nb.add(self.tab_oos, text="OOS Export")
        self._build_oos_export_tab(self.tab_oos)
        # 16) Diagnostics
        self.tab_diag = ttk.Frame(self.nb); self.nb.add(self.tab_diag, text="Diagnostics")
        self._build_diagnostics_tab(self.tab_diag)
        # 17) Q1/Q2 Reliability Experiments
        self.tab_q2 = ttk.Frame(self.nb); self.nb.add(self.tab_q2, text="Q1/Q2 Experiments")
        self._build_q2_experiments_tab(self.tab_q2)
        # 18) Locked one-click paper protocol
        self.tab_paper = ttk.Frame(self.nb); self.nb.add(self.tab_paper, text="Paper Canonical Run")
        self._build_paper_canonical_tab(self.tab_paper)
        
        # --- Logging controls ---
        ttk.Label(top, text="Log level:").grid(row=r, column=0, sticky="e")
        ttk.Combobox(top, textvariable=self.var_log_level,
                    values=["DEBUG","INFO","WARN","ERROR"],
                    width=10, state="readonly").grid(row=r, column=1, sticky="w")
        ttk.Label(top, text="Log every:").grid(row=r, column=2, sticky="e")
        ttk.Entry(top, textvariable=self.var_log_every, width=6).grid(row=r, column=3, sticky="w")
        ttk.Checkbutton(top, text="Show Rebalance Spam",
                        command=lambda: setattr(self, "_allow_rebalance_spam",
                                                not self._allow_rebalance_spam)).grid(row=r, column=4, sticky="w")

    def _create_plot_canvas(self, parent: tk.Widget) -> Tuple[plt.Figure, plt.Axes, FigureCanvasTkAgg]:
        fig = plt.Figure(figsize=(40, 6))
        ax = fig.add_subplot(111)
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        return fig, ax, canvas


    def _build_paper_canonical_tab(self, parent: tk.Widget):
        """Build the locked one-click manuscript experiment tab."""
        outer = ttk.Frame(parent, padding=10)
        outer.pack(fill="both", expand=True)

        guide = (
            "Paper Canonical Run locks the manuscript protocol, executes the main benchmarks, "
            "CODED_P1–P5, and the three main NLPI model families, exports all GUI tables, then "
            "runs the reliability package (prompt robustness, ticker masking, policy complexity, "
            "constraint conflict, and model-family generalization) in the same run directory. "
            "A single data snapshot and SHA-256 manifests are shared across stages. "
            "Stage 3 reconstructs nine publication-quality figures directly from canonical CSV files "
            "and saves PNG, PDF, and SVG versions."
        )
        ttk.Label(outer, text=guide, wraplength=1650, justify="left").pack(anchor="w", pady=(0, 8))

        protocol_box = ttk.LabelFrame(outer, text="Locked paper protocol", padding=8)
        protocol_box.pack(fill="x", pady=(0, 8))
        rows = [
            ("Protocol ID", PAPER_PROTOCOL["protocol_id"]),
            ("Period", f"{PAPER_PROTOCOL['start']} to {PAPER_PROTOCOL['end']}"),
            ("Universe", f"{len(PAPER_TICKERS)} ETFs: " + ", ".join(PAPER_TICKERS)),
            ("Execution", "WFCV, next-period execution, weight drift, holdout disabled"),
            ("Constraints", "42 trading days; cost=0.001; max weight=0.60; turnover cap=0.25"),
            ("Main models", ", ".join(PAPER_MAIN_MODELS)),
            ("Main strategies", ", ".join(PAPER_BENCHMARKS + PAPER_CODED) + "; NLPI P1–P5"),
            ("Robustness model", PAPER_ROBUSTNESS_MODELS[0]),
        ]
        for r, (key, value) in enumerate(rows):
            ttk.Label(protocol_box, text=key + ":", width=18, anchor="e").grid(row=r, column=0, sticky="ne", padx=(0, 8), pady=2)
            ttk.Label(protocol_box, text=value, wraplength=1450, justify="left").grid(row=r, column=1, sticky="w", pady=2)
        protocol_box.columnconfigure(1, weight=1)

        opts = ttk.LabelFrame(outer, text="Canonical run options", padding=8)
        opts.pack(fill="x", pady=(0, 8))
        ttk.Checkbutton(opts, text="Run reliability package after main experiment", variable=self.var_paper_include_reliability).grid(row=0, column=0, sticky="w", padx=(0, 16))
        ttk.Checkbutton(opts, text="Include qwen3.5:4b model-family robustness", variable=self.var_paper_include_qwen).grid(row=0, column=1, sticky="w", padx=(0, 16))
        ttk.Label(opts, text="n/regime:").grid(row=0, column=2, sticky="e")
        ttk.Entry(opts, textvariable=self.var_paper_n_per_regime, width=8).grid(row=0, column=3, sticky="w", padx=(4, 16))
        ttk.Label(opts, text="Output base:").grid(row=1, column=0, sticky="e", pady=(6, 0))
        ttk.Entry(opts, textvariable=self.var_paper_output_base, width=105).grid(row=1, column=1, columnspan=3, sticky="we", padx=(4, 6), pady=(6, 0))
        ttk.Button(opts, text="Browse", command=self._paper_choose_output_base).grid(row=1, column=4, sticky="w", pady=(6, 0))
        ttk.Label(opts, text="Figure DPI:").grid(row=0, column=4, sticky="e", padx=(12, 4))
        ttk.Combobox(opts, textvariable=self.var_paper_figure_dpi, values=[300, 600], width=7, state="readonly").grid(row=0, column=5, sticky="w")
        ttk.Label(opts, text="Status:").grid(row=2, column=0, sticky="e", pady=(6, 0))
        ttk.Label(opts, textvariable=self.var_paper_status, foreground="#075985").grid(row=2, column=1, columnspan=4, sticky="w", padx=(4, 0), pady=(6, 0))
        opts.columnconfigure(1, weight=1)

        btns = ttk.Frame(outer)
        btns.pack(fill="x", pady=(0, 8))
        ttk.Button(btns, text="Load Paper Protocol", command=self._paper_apply_protocol).pack(side="left", padx=(0, 6))
        ttk.Button(btns, text="Validate Protocol", command=lambda: self._paper_validate_protocol(show_dialog=True)).pack(side="left", padx=(0, 6))
        self.btn_paper_run = ttk.Button(btns, text="Run All Canonical Experiments", command=self.on_run_paper_canonical)
        self.btn_paper_run.pack(side="left", padx=(0, 6))
        ttk.Button(btns, text="Stop", command=self.on_stop_paper_canonical).pack(side="left", padx=(0, 6))
        ttk.Button(btns, text="Open Output Folder", command=self._paper_open_output_folder).pack(side="left", padx=(0, 6))
        self.btn_paper_figures = ttk.Button(btns, text="Generate Paper Figures", command=lambda: self._paper_generate_figures_manual(overwrite=False))
        self.btn_paper_figures.pack(side="left", padx=(0, 6))
        ttk.Button(btns, text="Regenerate Figures Only", command=lambda: self._paper_generate_figures_manual(overwrite=True)).pack(side="left", padx=(0, 6))
        ttk.Button(btns, text="Open Figures Folder", command=self._paper_open_figures_folder).pack(side="left", padx=(0, 6))
        ttk.Button(btns, text="Refresh Files", command=self._paper_refresh_files).pack(side="left", padx=(0, 6))
        ttk.Button(btns, text="Copy Log", command=lambda: self._copy_text_widget(self.txt_paper_log)).pack(side="right")

        paned = ttk.PanedWindow(outer, orient="vertical")
        paned.pack(fill="both", expand=True)
        files_frame = ttk.LabelFrame(paned, text="Canonical output files", padding=6)
        paned.add(files_frame, weight=1)
        self.tree_paper_files = ttk.Treeview(files_frame, show="headings", columns=("file", "size"), height=9, selectmode="extended")
        self.tree_paper_files.heading("file", text="File")
        self.tree_paper_files.heading("size", text="Size")
        self.tree_paper_files.column("file", width=1050, anchor="w")
        self.tree_paper_files.column("size", width=120, anchor="e")
        self.tree_paper_files.pack(side="left", fill="both", expand=True)
        vs = ttk.Scrollbar(files_frame, orient="vertical", command=self.tree_paper_files.yview)
        vs.pack(side="right", fill="y")
        self.tree_paper_files.configure(yscrollcommand=vs.set)
        self._attach_tree_copy_handlers(self.tree_paper_files, header_labels=["File", "Size"])

        log_frame = ttk.LabelFrame(paned, text="Canonical run log", padding=6)
        paned.add(log_frame, weight=3)
        self.txt_paper_log = scrolledtext.ScrolledText(log_frame, height=18, wrap="word")
        self.txt_paper_log.pack(fill="both", expand=True)
        self._paper_log("Ready. Click Load Paper Protocol, Validate Protocol, then Run All Canonical Experiments. Stage 3 will create nine paper figures in PNG/PDF/SVG.")

    def _paper_log(self, msg: str):
        try:
            self.txt_paper_log.insert("end", str(msg).rstrip() + "\n")
            self.txt_paper_log.see("end")
        except Exception:
            pass
        try:
            self._log("[PAPER] " + str(msg).rstrip())
        except Exception:
            pass

    def _paper_choose_output_base(self):
        d = filedialog.askdirectory(title="Choose canonical paper-run output base")
        if d:
            self.var_paper_output_base.set(d)

    def _paper_apply_protocol(self):
        """Load the exact manuscript settings and strategy list into the GUI."""
        self._apply_main_nlpi_gemma4_preset()
        self.var_tickers.set(",".join(PAPER_TICKERS))
        self.var_selected_port.set("ETF Multi-Asset Main (22)")
        self.var_start.set(PAPER_PROTOCOL["start"])
        self.var_end.set(PAPER_PROTOCOL["end"])
        self.var_reb.set(PAPER_PROTOCOL["rebalance_days"])
        self.var_tcost.set(PAPER_PROTOCOL["transaction_cost"])
        self.var_maxw.set(PAPER_PROTOCOL["max_weight"])
        self.var_turn.set(PAPER_PROTOCOL["turnover_cap"])
        self.var_use_llm.set(True)
        self.var_wfcv.set(True)
        self.var_holdout.set(False)
        # Main preset deliberately omits coded rules; canonical mode includes them.
        for key in PAPER_CODED:
            if key in self._bench_vars:
                self._bench_vars[key].set(True)
        self._rebuild_benchmarks_tree()
        self._paper_log("Loaded locked protocol NLPI-PAPER-CANONICAL-V1 into the GUI.")

    def _paper_current_values(self):
        tickers = [t.strip() for t in self.var_tickers.get().split(",") if t.strip()]
        selected = [k for k, _ in self._bench_defs if self._bench_vars.get(k, tk.BooleanVar(value=False)).get()]
        return {
            "start": (self.ent_start.get() if hasattr(self, "ent_start") else self.var_start.get()).strip(),
            "end": (self.ent_end.get() if hasattr(self, "ent_end") else self.var_end.get()).strip(),
            "tickers": tickers,
            "rebalance_days": int(self.var_reb.get()),
            "transaction_cost": float(self.var_tcost.get()),
            "max_weight": float(self.var_maxw.get()),
            "turnover_cap": float(self.var_turn.get()),
            "wfcv": bool(self.var_wfcv.get()),
            "holdout": bool(self.var_holdout.get()),
            "selected_strategies": selected,
            "installed_models": list(self._ollama_models or []),
        }

    def _paper_validate_protocol(self, show_dialog: bool = False):
        values = self._paper_current_values()
        errors, warnings = validate_protocol_values(
            values,
            include_qwen=bool(self.var_paper_include_reliability.get() and self.var_paper_include_qwen.get()),
        )
        out_base = self.var_paper_output_base.get().strip()
        if not out_base:
            errors.append("Output base is empty.")
        else:
            try:
                os.makedirs(out_base, exist_ok=True)
                test = os.path.join(out_base, ".nlpi_write_test")
                with open(test, "w", encoding="utf-8") as f:
                    f.write("ok")
                os.remove(test)
            except Exception as exc:
                errors.append(f"Output base is not writable: {exc}")
        for required_file in ["core.py", "llm.py", "paper_canonical.py", os.path.join("q1_experiments", "runner.py")]:
            if not os.path.exists(os.path.join(os.getcwd(), required_file)):
                errors.append(f"Missing source file: {required_file}")

        self._paper_log("Protocol validation:")
        if errors:
            for x in errors:
                self._paper_log("  [ERROR] " + x)
        else:
            self._paper_log("  [OK] Locked dates, universe, constraints, models, and strategies match.")
        for x in warnings:
            self._paper_log("  [WARN] " + x)
        if show_dialog:
            if errors:
                messagebox.showerror("Canonical protocol validation", "Validation failed:\n\n" + "\n".join(errors))
            else:
                messagebox.showinfo("Canonical protocol validation", "Protocol validation passed." + ("\n\n" + "\n".join(warnings) if warnings else ""))
        return not errors

    def on_run_paper_canonical(self):
        if self._paper_running or (self.worker_future and not self.worker_future.done()):
            messagebox.showinfo("Paper Canonical Run", "A canonical or main run is already in progress.")
            return
        self._paper_apply_protocol()
        if not self._paper_validate_protocol(show_dialog=True):
            return

        root = create_run_directory(self.var_paper_output_base.get())
        self._paper_root = str(root)
        self._paper_running = True
        self._paper_stage = "main"
        self.var_paper_status.set("Stage 1/3 — Main OOS/WFCV experiments")
        write_initial_manifest(
            root, Path(os.getcwd()),
            include_reliability=bool(self.var_paper_include_reliability.get()),
            include_qwen=bool(self.var_paper_include_qwen.get()),
        )
        update_manifest(root, status="running", stage="main", stage_event={"stage": "main", "status": "started"})

        # All stages in this canonical run share one immutable price snapshot.
        snapshot = root / "data" / "adjusted_close.csv"
        os.environ["NLPI_DATA_SNAPSHOT"] = str(snapshot)
        os.environ["NLPI_USE_SNAPSHOT"] = "1"
        os.environ["NLPI_SAVE_SNAPSHOT"] = "1"

        selected_models, personas_by_model, selected_keys = self._selected_nlpi_models_personas()
        self._auto_export_outdir = str(root / "main")
        self._auto_export_after_run = True
        self._auto_export_selected_models = selected_models
        self._auto_export_personas_by_model = personas_by_model
        self._auto_export_selected_keys = selected_keys
        self.btn_paper_run.configure(state="disabled")
        self._paper_log(f"Canonical run root: {root}")
        self._paper_log("Stage 1/3: running main benchmarks, coded references, and NLPI WFCV experiment.")
        self.on_run()

    def _paper_after_main_export(self, main_outdir: str):
        if not self._paper_running or not self._paper_root:
            return
        root = Path(self._paper_root)
        update_manifest(root, stage="main-complete", stage_event={"stage": "main", "status": "completed", "output": main_outdir})
        self._paper_log("Stage 1 complete: main experiment exported.")
        self._paper_refresh_files()
        if bool(self.var_paper_include_reliability.get()):
            self._paper_start_reliability()
        else:
            self._paper_start_figure_generation(automatic=True, overwrite=True, include_reliability=False)

    def _paper_start_reliability(self):
        if not self._paper_root:
            return
        root = Path(self._paper_root)
        self._paper_stage = "reliability"
        self.var_paper_status.set("Stage 2/3 — Reliability experiments")
        models = list(PAPER_MAIN_MODELS)
        if self.var_paper_include_qwen.get():
            models.extend(m for m in PAPER_ROBUSTNESS_MODELS if m not in models)
        outdir = root / "reliability"
        nreg = max(1, int(self.var_paper_n_per_regime.get() or PAPER_PROTOCOL["n_per_regime"]))
        args = [
            sys.executable, "-m", "q1_experiments.runner",
            "--start", PAPER_PROTOCOL["start"], "--end", PAPER_PROTOCOL["end"],
            "--rebalance", str(PAPER_PROTOCOL["rebalance_days"]),
            "--tcost", str(PAPER_PROTOCOL["transaction_cost"]),
            "--maxw", str(PAPER_PROTOCOL["max_weight"]),
            "--turncap", str(PAPER_PROTOCOL["turnover_cap"]),
            "--prompt-cap", str(PAPER_PROTOCOL["prompt_cap_pct"]),
            "--tickers", *PAPER_TICKERS,
            "--models", *models,
            "--outdir", str(outdir),
            "--experiments", *PAPER_PROTOCOL["reliability_experiments"],
            "--decision-sample", PAPER_PROTOCOL["reliability_decision_sample"],
            "--n-per-regime", str(nreg),
            "--policies", *PAPER_PROTOCOL["reliability_policies"],
            "--seed", str(PAPER_PROTOCOL["seed"]),
        ]
        update_manifest(root, stage="reliability", stage_event={"stage": "reliability", "status": "started", "models": models})
        self._paper_log("Stage 2/3: running reliability and model-family robustness package.")
        self._paper_log("Running: " + " ".join(args))
        self.pb.start(10)
        env = os.environ.copy()

        def _task():
            code = 1
            try:
                self._paper_proc = subprocess.Popen(
                    args, cwd=os.getcwd(), env=env,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1,
                )
                assert self._paper_proc.stdout is not None
                for line in self._paper_proc.stdout:
                    self.after(0, self._paper_log, line.rstrip())
                code = self._paper_proc.wait()
            except Exception as exc:
                self.after(0, self._paper_log, f"[ERROR] Reliability stage failed: {exc}")
            finally:
                if code == 0:
                    self.after(0, self._paper_start_figure_generation, True, True, True)
                else:
                    self.after(0, self._paper_finish, "failed", code)
        threading.Thread(target=_task, daemon=True).start()

    def _paper_start_figure_generation(self, automatic=True, overwrite=True, include_reliability=True):
        if not self._paper_root:
            return
        root = Path(self._paper_root)
        self._paper_stage = "figures"
        self.var_paper_status.set("Stage 3/3 — Publication-quality figures")
        update_manifest(root, stage="figures", stage_event={"stage": "figures", "status": "started"})
        self._paper_log("Stage 3/3: validating canonical CSVs and generating nine paper figures as PNG/PDF/SVG.")
        self.pb.start(10)

        def _task():
            try:
                manifest = generate_all_figures(
                    root,
                    dpi=int(self.var_paper_figure_dpi.get() or 600),
                    overwrite=bool(overwrite),
                    strict=bool(automatic),
                    include_reliability=bool(include_reliability),
                    log_fn=lambda msg: self.after(0, self._paper_log, msg),
                )
                update_manifest(root, stage="figures-complete", stage_event={"stage": "figures", "status": "completed", "generated": sorted(manifest.get("generated", {}))})
                self.after(0, self._paper_refresh_files)
                if automatic:
                    self.after(0, self._paper_finish, "completed", 0 if include_reliability else None)
                else:
                    self.after(0, self._paper_manual_figures_done, True, "")
            except Exception as exc:
                self.after(0, self._paper_log, f"[ERROR] Figure generation failed: {exc}")
                if automatic:
                    self.after(0, self._paper_finish, "failed", 1)
                else:
                    self.after(0, self._paper_manual_figures_done, False, str(exc))

        self._paper_figure_thread = threading.Thread(target=_task, daemon=True)
        self._paper_figure_thread.start()

    def _paper_generate_figures_manual(self, overwrite=False):
        if self._paper_running:
            messagebox.showinfo("Paper Figures", "A canonical run is currently active. Figures will be generated automatically in Stage 3.")
            return
        root = self._paper_root
        if not root or not os.path.isdir(root):
            root = filedialog.askdirectory(title="Choose completed canonical run folder")
            if not root:
                return
            self._paper_root = root
        self.btn_paper_figures.configure(state="disabled")
        self.var_paper_status.set("Generating paper figures from canonical CSVs")
        self._paper_start_figure_generation(automatic=False, overwrite=overwrite, include_reliability=True)

    def _paper_manual_figures_done(self, success, detail):
        self.pb.stop()
        self.var_paper_status.set("Figures completed" if success else "Figure generation failed")
        if hasattr(self, "btn_paper_figures"):
            self.btn_paper_figures.configure(state="normal")
        self._paper_refresh_files()
        if success:
            messagebox.showinfo("Paper Figures", f"Publication figures were generated.\n\n{Path(self._paper_root) / 'figures'}")
        else:
            messagebox.showerror("Paper Figures", detail)

    def _paper_open_figures_folder(self):
        if not self._paper_root:
            messagebox.showinfo("Paper Figures", "No canonical run folder is selected.")
            return
        d = str(Path(self._paper_root) / "figures")
        if not os.path.isdir(d):
            messagebox.showinfo("Paper Figures", "The figures folder does not exist yet.")
            return
        try:
            if sys.platform == "darwin":
                subprocess.Popen(["open", d])
            elif os.name == "nt":
                os.startfile(d)  # type: ignore[attr-defined]
            else:
                subprocess.Popen(["xdg-open", d])
        except Exception as exc:
            messagebox.showwarning("Paper Figures", f"Could not open folder: {exc}")

    def _paper_finish(self, status: str, reliability_exit_code=None):
        if not self._paper_root:
            return
        root = Path(self._paper_root)
        try:
            finalize_run(root, status=status, reliability_exit_code=reliability_exit_code)
        except Exception as exc:
            self._paper_log(f"[WARN] Final manifest/checksum generation failed: {exc}")
        self._paper_running = False
        self._paper_stage = "complete" if status == "completed" else "failed"
        self.var_paper_status.set("Completed — figures and tables ready" if status == "completed" else "Failed — inspect canonical log")
        self.btn_paper_run.configure(state="normal")
        self.pb.stop()
        self._paper_refresh_files()
        if status == "completed":
            self._paper_log(f"Canonical paper run completed: {root}")
            messagebox.showinfo("Paper Canonical Run", f"All selected canonical experiments completed.\n\n{root}")
        else:
            self._paper_log(f"Canonical paper run failed or stopped: {root}")
            messagebox.showerror("Paper Canonical Run", f"Canonical run did not complete successfully.\n\n{root}")

    def on_stop_paper_canonical(self):
        self.cancel_event.set()
        proc = self._paper_proc
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass
        if self._paper_running and self._paper_root:
            try:
                finalize_run(Path(self._paper_root), status="stopped", reliability_exit_code=None)
            except Exception:
                pass
        self._paper_running = False
        self._paper_stage = "stopped"
        self.var_paper_status.set("Stopped")
        if hasattr(self, "btn_paper_run"):
            self.btn_paper_run.configure(state="normal")
        self._paper_log("Stop requested for canonical paper run.")

    def _paper_open_output_folder(self):
        d = self._paper_root or self.var_paper_output_base.get()
        if not d or not os.path.exists(d):
            messagebox.showinfo("Paper Canonical Run", "No output folder exists yet.")
            return
        try:
            if sys.platform == "darwin":
                subprocess.Popen(["open", d])
            elif os.name == "nt":
                os.startfile(d)  # type: ignore[attr-defined]
            else:
                subprocess.Popen(["xdg-open", d])
        except Exception as exc:
            messagebox.showwarning("Paper Canonical Run", f"Could not open folder: {exc}")

    def _paper_refresh_files(self):
        if not hasattr(self, "tree_paper_files"):
            return
        for iid in self.tree_paper_files.get_children():
            self.tree_paper_files.delete(iid)
        if not self._paper_root or not os.path.exists(self._paper_root):
            return
        root = Path(self._paper_root)
        for p in sorted(x for x in root.rglob("*") if x.is_file()):
            size = p.stat().st_size
            size_s = f"{size/1024:.1f} KB" if size < 1024*1024 else f"{size/(1024*1024):.2f} MB"
            self.tree_paper_files.insert("", "end", values=(str(p.relative_to(root)), size_s))


    def _build_q2_experiments_tab(self, parent: tk.Widget):
        """GUI wrapper for Q1/Q2 NLPI reliability/safety experiment package."""
        outer = ttk.Frame(parent, padding=10)
        outer.pack(fill="both", expand=True)

        guide = (
            "Q1/Q2 추가 실험은 성과 우월성 검증이 아니라 NLPI reliability 검증입니다: "
            "prompt robustness, ticker masking, policy-complexity ladder, constraint-conflict stress, "
            "model-family generalization, sensitivity template. 현재 상단의 날짜/티커/비용/제약 설정을 그대로 사용합니다."
        )
        ttk.Label(outer, text=guide, wraplength=1600, justify="left").pack(anchor="w", pady=(0, 8))

        cfg = ttk.LabelFrame(outer, text="Experiment configuration", padding=8)
        cfg.pack(fill="x", pady=(0, 8))

        ttk.Label(cfg, text="Mode:").grid(row=0, column=0, sticky="e", padx=(0, 4), pady=2)
        ttk.Combobox(
            cfg, textvariable=self.var_q2_mode, state="readonly", width=18,
            values=["dry", "smoke", "q2", "q2_extended", "core", "full"]
        ).grid(row=0, column=1, sticky="w", padx=(0, 12), pady=2)

        ttk.Checkbutton(cfg, text="Use extra models", variable=self.var_q2_use_extra).grid(row=0, column=2, sticky="w", padx=(0,4), pady=2)
        ttk.Label(cfg, text="Extra models:").grid(row=0, column=3, sticky="e", padx=(12, 4), pady=2)
        ttk.Entry(cfg, textvariable=self.var_q2_extra_models, width=38).grid(row=0, column=4, sticky="w", padx=(0, 12), pady=2)

        ttk.Label(cfg, text="n/regime:").grid(row=1, column=0, sticky="e", padx=(0, 4), pady=2)
        ttk.Entry(cfg, textvariable=self.var_q2_n_per_regime, width=8).grid(row=1, column=1, sticky="w", padx=(0, 12), pady=2)
        ttk.Label(cfg, text="Max calls (optional):").grid(row=1, column=2, sticky="e", padx=(12, 4), pady=2)
        ttk.Entry(cfg, textvariable=self.var_q2_max_calls, width=12).grid(row=1, column=3, sticky="w", padx=(0, 12), pady=2)

        ttk.Label(cfg, text="Output base:").grid(row=2, column=0, sticky="e", padx=(0, 4), pady=2)
        ttk.Entry(cfg, textvariable=self.var_q2_out_base, width=80).grid(row=2, column=1, columnspan=4, sticky="we", padx=(0, 6), pady=2)
        ttk.Button(cfg, text="Browse", command=self._q2_choose_out_base).grid(row=2, column=5, sticky="w", pady=2)
        for c in range(6):
            cfg.columnconfigure(c, weight=1 if c == 4 else 0)

        btns = ttk.Frame(outer)
        btns.pack(fill="x", pady=(0, 8))
        self.btn_q2_run = ttk.Button(btns, text="Run Q1/Q2 Package", command=self.on_run_q2_package)
        self.btn_q2_run.pack(side="left", padx=(0, 6))
        self.btn_q2_stop = ttk.Button(btns, text="Stop Q1/Q2", command=self.on_stop_q2_package)
        self.btn_q2_stop.pack(side="left", padx=(0, 6))
        ttk.Button(btns, text="Open Output Folder", command=self._q2_open_output_folder).pack(side="left", padx=(0, 6))
        ttk.Button(btns, text="Refresh Output Files", command=self._q2_refresh_output_files).pack(side="left", padx=(0, 6))
        ttk.Button(btns, text="Copy Log", command=lambda: self._copy_text_widget(self.txt_q2_log)).pack(side="right")

        paned = ttk.PanedWindow(outer, orient="vertical")
        paned.pack(fill="both", expand=True)

        files_frame = ttk.LabelFrame(paned, text="Generated result tables / files", padding=6)
        paned.add(files_frame, weight=1)
        self.tree_q2_files = ttk.Treeview(files_frame, show="headings", columns=("file", "size"), height=8, selectmode="extended")
        self.tree_q2_files.heading("file", text="File")
        self.tree_q2_files.heading("size", text="Size")
        self.tree_q2_files.column("file", width=950, anchor="w")
        self.tree_q2_files.column("size", width=120, anchor="e")
        self.tree_q2_files.pack(side="left", fill="both", expand=True)
        vs = ttk.Scrollbar(files_frame, orient="vertical", command=self.tree_q2_files.yview)
        vs.pack(side="right", fill="y"); self.tree_q2_files.configure(yscrollcommand=vs.set)
        self._attach_tree_copy_handlers(self.tree_q2_files, header_labels=["File", "Size"])

        log_frame = ttk.LabelFrame(paned, text="Q1/Q2 run log", padding=6)
        paned.add(log_frame, weight=3)
        self.txt_q2_log = scrolledtext.ScrolledText(log_frame, height=18, wrap="word")
        self.txt_q2_log.pack(fill="both", expand=True)
        self._q2_log("Ready. Recommended first run: mode=dry. Then run mode=smoke. For paper robustness, run mode=q2 or q2_extended.")

    def _q2_choose_out_base(self):
        d = filedialog.askdirectory(title="Choose Q1/Q2 output base folder")
        if d:
            self.var_q2_out_base.set(os.path.join(d, "q2_gui_reliability"))

    def _q2_log(self, msg: str):
        try:
            self.txt_q2_log.insert("end", str(msg).rstrip() + "\n")
            self.txt_q2_log.see("end")
        except Exception:
            pass
        try:
            self._log("[Q1/Q2] " + str(msg).rstrip())
        except Exception:
            pass

    def _q2_build_command(self):
        import datetime
        mode = self.var_q2_mode.get().strip() or "q2"
        tickers = [t.strip() for t in self.var_tickers.get().split(",") if t.strip()]
        start = (self.ent_start.get() if hasattr(self, "ent_start") else self.var_start.get()).strip()
        end = (self.ent_end.get() if hasattr(self, "ent_end") else self.var_end.get()).strip()
        reb = str(int(self.var_reb.get()))
        tcost = str(float(self.var_tcost.get()))
        maxw = str(float(self.var_maxw.get()))
        turn = str(float(self.var_turn.get()))
        nreg = str(int(self.var_q2_n_per_regime.get() or 6))
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        outdir = f"{self.var_q2_out_base.get().rstrip(os.sep)}_{mode}_{ts}"

        base_models = ["gemma3:270m", "gemma3:1b", "llama3.1:8b"]
        extra = []
        if self.var_q2_use_extra.get() or mode == "q2_extended":
            extra = [x.strip() for x in self.var_q2_extra_models.get().split() if x.strip()]
        models = base_models + [m for m in extra if m not in base_models]

        common = [
            sys.executable, "-m", "q1_experiments.runner",
            "--start", start, "--end", end,
            "--rebalance", reb, "--tcost", tcost, "--maxw", maxw, "--turncap", turn,
            "--prompt-cap", "60",
            "--tickers", *tickers,
            "--models", *models,
            "--outdir", outdir,
        ]
        max_calls = self.var_q2_max_calls.get().strip()
        if max_calls:
            common += ["--max-calls", max_calls]

        if mode == "dry":
            args = common + ["--experiments", "all", "--decision-sample", "first", "--n-per-regime", "2", "--max-calls", max_calls or "20", "--dry-run", "--synthetic-data"]
        elif mode == "smoke":
            args = common + ["--experiments", "prompt_robustness", "ticker_masking", "constraint_stress", "sensitivity_template", "--decision-sample", "first", "--n-per-regime", "2", "--max-calls", max_calls or "30"]
        elif mode == "core":
            args = common + ["--experiments", "prompt_robustness", "ticker_masking", "policy_complexity", "constraint_stress", "model_generalization", "sensitivity_template", "--decision-sample", "stratified", "--n-per-regime", "10"]
        elif mode == "full":
            args = common + ["--experiments", "prompt_robustness", "ticker_masking", "policy_complexity", "constraint_stress", "model_generalization", "sensitivity_template", "--decision-sample", "full"]
        else:  # q2 or q2_extended
            args = common + ["--experiments", "prompt_robustness", "ticker_masking", "policy_complexity", "constraint_stress", "model_generalization", "sensitivity_template", "--decision-sample", "stratified", "--n-per-regime", nreg, "--policies", "P1", "P2", "P3", "P4", "P5", "P6"]
        return args, outdir

    def on_run_q2_package(self):
        if getattr(self, "_q2_proc", None) is not None and self._q2_proc.poll() is None:
            messagebox.showinfo("Q1/Q2", "A Q1/Q2 experiment is already running.")
            return
        args, outdir = self._q2_build_command()
        self._q2_last_outdir = outdir
        os.makedirs(outdir, exist_ok=True)
        self._q2_log("Running: " + " ".join(args))
        self._q2_log("Output folder: " + outdir)
        self.btn_q2_run.configure(state="disabled")
        self.pb.start(10)

        def _task():
            try:
                self._q2_proc = subprocess.Popen(
                    args, cwd=os.getcwd(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1
                )
                assert self._q2_proc.stdout is not None
                for line in self._q2_proc.stdout:
                    self.after(0, self._q2_log, line.rstrip())
                code = self._q2_proc.wait()
                self.after(0, self._q2_log, f"Finished with exit code {code}")
                self.after(0, self._q2_refresh_output_files)
            except Exception as e:
                self.after(0, self._q2_log, f"[ERROR] {e}")
            finally:
                self.after(0, lambda: self.btn_q2_run.configure(state="normal"))
                self.after(0, self.pb.stop)
        threading.Thread(target=_task, daemon=True).start()

    def on_stop_q2_package(self):
        proc = getattr(self, "_q2_proc", None)
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                self._q2_log("Terminate requested for Q1/Q2 process.")
            except Exception as e:
                self._q2_log(f"[WARN] terminate failed: {e}")

    def _q2_open_output_folder(self):
        d = getattr(self, "_q2_last_outdir", None) or self.var_q2_out_base.get()
        if not d or not os.path.exists(d):
            messagebox.showinfo("Q1/Q2", "No output folder exists yet.")
            return
        try:
            if sys.platform == "darwin":
                subprocess.Popen(["open", d])
            elif os.name == "nt":
                os.startfile(d)  # type: ignore[attr-defined]
            else:
                subprocess.Popen(["xdg-open", d])
        except Exception as e:
            messagebox.showwarning("Q1/Q2", f"Could not open folder: {e}")

    def _q2_refresh_output_files(self):
        if not hasattr(self, "tree_q2_files"):
            return
        for iid in self.tree_q2_files.get_children():
            self.tree_q2_files.delete(iid)
        d = getattr(self, "_q2_last_outdir", None)
        if not d or not os.path.exists(d):
            return
        files = []
        for pat in ["*.json", "*.csv", "logs/*.csv", "logs/*.jsonl", "tables/*.csv", "figures/*.png"]:
            files.extend(glob.glob(os.path.join(d, pat)))
        for f in sorted(files):
            try:
                size = os.path.getsize(f)
                size_s = f"{size/1024:.1f} KB" if size < 1024*1024 else f"{size/(1024*1024):.2f} MB"
            except Exception:
                size_s = ""
            self.tree_q2_files.insert("", "end", values=(os.path.relpath(f, d), size_s))

    def _build_portfolio_tab(self, parent: tk.Widget):
        frm = ttk.Frame(parent, padding=8); frm.pack(fill="both", expand=True)
        left = ttk.Frame(frm); left.pack(side="left", fill="y", padx=(0,8))
        ttk.Label(left, text="Predefined Portfolios").pack(anchor="w")
        self.lst_ports = tk.Listbox(left, height=14, exportselection=False)
        self.lst_ports.pack(fill="y", expand=False)
        for name in PORTFOLIOS.keys(): self.lst_ports.insert("end", name)
        btns = ttk.Frame(left); btns.pack(pady=6, anchor="w")
        ttk.Button(btns, text="Apply to Tickers", command=self.on_apply_portfolio).pack(side="left")
        right = ttk.Frame(frm); right.pack(side="left", fill="both", expand=True)
        ttk.Label(right, text="Portfolio Tickers").pack(anchor="w")
        self.lst_tickers = tk.Listbox(right, height=12, selectmode="extended", exportselection=False)
        self.lst_tickers.pack(fill="both", expand=True)
        copy_fr = ttk.Frame(frm); copy_fr.pack(fill="x", pady=6)
        ttk.Button(copy_fr, text="Copy Tickers", command=self._copy_current_tickers).pack(side="left")

        # Auto-select the revised main experimental universe so the GUI opens
        # with the ETF-only multi-asset universe requested for reviewer response.
        self._select_default_portfolio_in_gui()

    def _select_default_portfolio_in_gui(self):
        """Populate Portfolio tab and top ticker field with the default universe."""
        try:
            default_name = DEFAULT_PORTFOLIO_NAME
            if default_name not in PORTFOLIOS:
                return
            names = list(PORTFOLIOS.keys())
            idx = names.index(default_name)
            self.lst_ports.selection_clear(0, "end")
            self.lst_ports.selection_set(idx)
            self.lst_ports.activate(idx)
            self.var_selected_port.set(default_name)
            ticks = PORTFOLIOS[default_name]
            self.var_tickers.set(",".join(ticks))
            self.lst_tickers.delete(0, "end")
            for t in ticks:
                self.lst_tickers.insert("end", t)
        except Exception as e:
            try:
                self._log(f"Default portfolio auto-selection skipped: {e}")
            except Exception:
                pass

    def _build_benchmarks_tab(self, parent: tk.Widget):
        frm = ttk.Frame(parent, padding=8); frm.pack(fill="both", expand=True)
        ttk.Label(frm, text="Select benchmarks to run.").pack(anchor="w", pady=(0,6))
        self.tree_bm = ttk.Treeview(frm, show="headings", columns=("Use","Key","Description"), height=14, selectmode="none")
        self.tree_bm.heading("Use", text="Use"); self.tree_bm.column("Use", width=60, anchor="center")
        self.tree_bm.heading("Key", text="Key"); self.tree_bm.column("Key", width=220, anchor="w")
        self.tree_bm.heading("Description", text="Description"); self.tree_bm.column("Description", width=520, anchor="w")
        self.tree_bm.pack(side="left", fill="both", expand=True)
        self.tree_bm.bind("<Button-1>", self._on_benchmark_click)
        self._rebuild_benchmarks_tree()
        vs = ttk.Scrollbar(frm, orient="vertical", command=self.tree_bm.yview)
        vs.pack(side="right", fill="y"); self.tree_bm.configure(yscrollcommand=vs.set)
        btns = ttk.Frame(parent, padding=(8,0)); btns.pack(fill="x")
        ttk.Button(btns, text="Select All", command=lambda: self._bench_set_all(True)).pack(side="left", padx=(0,6))
        ttk.Button(btns, text="Select None", command=lambda: self._bench_set_all(False)).pack(side="left")
        ttk.Button(btns, text="Preset Main: G3+G3-1B+Llama", command=self._apply_main_nlpi_gemma4_preset).pack(side="left", padx=(12,0))
        ttk.Button(btns, text="Copy Grid",
                  command=lambda: self._tree_copy(self.tree_bm, only_selected=False)).pack(side="right")

    def _build_stats_tab(self, parent: tk.Widget):
        frm = ttk.Frame(parent, padding=8); frm.pack(fill="both", expand=True)
        self.tree_stats = ttk.Treeview(frm, show="headings", columns=("Ticker",), height=20, selectmode="extended")
        self.tree_stats.pack(side="left", fill="both", expand=True)
        self.tree_stats.heading("Ticker", text="Ticker"); self.tree_stats.column("Ticker", width=140, anchor="w", stretch=True)
        vs = ttk.Scrollbar(frm, orient="vertical", command=self.tree_stats.yview)
        vs.pack(side="right", fill="y"); self.tree_stats.configure(yscrollcommand=vs.set)
        self._attach_tree_copy_handlers(self.tree_stats, header_labels=[])
        ctrl = ttk.Frame(parent, padding=(8,0)); ctrl.pack(fill="x")
        ttk.Button(ctrl, text="Copy Table", command=lambda: self._copy_tree_as_tsv(self.tree_stats)).pack(side="left")

    def _build_wfcv_tab(self, parent: tk.Widget):
        frm = ttk.Frame(parent, padding=8); frm.pack(fill="both", expand=True)
        self.fig_wfcv, self.ax_wfcv, self.canvas_wfcv = self._create_plot_canvas(frm)
        self._add_plot_copy_buttons(parent, self.canvas_wfcv)
        cols = ("Fold","Strategy","Sharpe","CAGR","MDD")
        self.tree_wfcv = ttk.Treeview(parent, show="headings", columns=cols, height=10, selectmode="extended")
        for c in cols:
            self.tree_wfcv.heading(c, text=c)
            self.tree_wfcv.column(c, width=140 if c!="Fold" else 80,
                                  anchor="w" if c in ("Fold","Strategy") else "e", stretch=True)
        self.tree_wfcv.pack(fill="x", padx=8, pady=(4,8))
        self._attach_tree_copy_handlers(self.tree_wfcv, header_labels=list(cols))
        ctrl = ttk.Frame(parent, padding=(8,0)); ctrl.pack(fill="x")
        ttk.Button(ctrl, text="Copy Table", command=lambda: self._copy_tree_as_tsv(self.tree_wfcv)).pack(side="left")
        
    def _build_metrics_tab(self, parent: tk.Widget):
        self.tree = ttk.Treeview(parent, show="headings", columns=COL_KEYS, height=18, selectmode="extended")
        self.tree.pack(side="left", fill="both", expand=True, padx=(0,6), pady=6)
        for key, label, width in COL_DEF:
            self.tree.heading(key, text=label)
            self.tree.column(key, width=width, anchor="w" if key == "name" else "e", stretch=True)
        vsb = ttk.Scrollbar(parent, orient="vertical", command=self.tree.yview)
        vsb.pack(side="right", fill="y"); self.tree.configure(yscrollcommand=vsb.set)
        self._attach_tree_copy_handlers(self.tree, header_labels=[lbl for _, lbl, _ in COL_DEF])
        ctrl = ttk.Frame(parent, padding=(8,0)); ctrl.pack(fill="x")
        ttk.Button(ctrl, text="Copy Table", command=lambda: self._copy_tree_as_tsv(self.tree)).pack(side="left")

    def _build_sig_tab(self, parent: tk.Widget):
        top = ttk.Frame(parent, padding=8); top.pack(fill="x")
        ttk.Label(top, text="Comparator (baseline):").pack(side="left")
        self.cbo_sig = ttk.Combobox(top, textvariable=self.var_sig_base, width=24, state="readonly", values=[])
        self.cbo_sig.pack(side="left", padx=(6,6))
        ttk.Button(top, text="Recompute", command=self._recompute_significance).pack(side="left")
        frm = ttk.Frame(parent, padding=8); frm.pack(fill="both", expand=True)
        cols = ["Group","Algo","Comparator","N","Mean Diff (Ann.)","t_HAC","p_HAC","Wilcoxon p","JK z","MBB p_two","Stars"]
        self.tree_sig = ttk.Treeview(frm, show="headings", columns=cols, height=18, selectmode="extended")
        self.tree_sig.pack(side="left", fill="both", expand=True)
        for c in cols:
            self.tree_sig.heading(c, text=c)
            anchor = "w" if c in ("Group","Algo","Comparator","Stars") else "e"
            width  = 120 if c not in ("Mean Diff (Ann.)","t_HAC","JK z","MBB p_two") else 140
            self.tree_sig.column(c, width=width, anchor=anchor, stretch=True)
        vs = ttk.Scrollbar(frm, orient="vertical", command=self.tree_sig.yview)
        vs.pack(side="right", fill="y"); self.tree_sig.configure(yscrollcommand=vs.set)
        self._attach_tree_copy_handlers(self.tree_sig, header_labels=cols)
        ctrl = ttk.Frame(parent, padding=(8,0)); ctrl.pack(fill="x")
        ttk.Button(ctrl, text="Copy Table", command=lambda: self._copy_tree_as_tsv(self.tree_sig)).pack(side="left")

    def _build_fewshot_tab(self, parent: tk.Widget):
        frm = ttk.Frame(parent, padding=8); frm.pack(fill="both", expand=True)
        self.txt_few = tk.Text(frm, height=24, wrap="word")
        self.txt_few.pack(fill="both", expand=True)
        btns = ttk.Frame(parent, padding=(8,4)); btns.pack(fill="x")
        ttk.Button(btns, text="Copy", command=lambda: self._copy_text_widget(self.txt_few)).pack(side="left")
        ttk.Button(btns, text="Clear", command=lambda: self.txt_few.delete("1.0","end")).pack(side="left", padx=(6,0))

    def _build_prompts_tab(self, parent: tk.Widget):
        frm = ttk.Frame(parent, padding=8); frm.pack(fill="both", expand=True)
        sel = ttk.Frame(frm); sel.pack(fill="x", pady=(0,6))
        ttk.Label(sel, text="Policy persona (for NLPI benchmark):").pack(side="left")
        for i in PROMPT_PROFILES.keys():
            ttk.Radiobutton(sel, text=str(i), variable=self.var_prompt_level, value=i,
                            command=self._refresh_prompt_preview).pack(side="left", padx=(6,0))
        self.txt_prompt_preview = tk.Text(frm, height=18, wrap="word")
        self.txt_prompt_preview.pack(fill="both", expand=True, pady=(4,4))
        btns = ttk.Frame(frm); btns.pack(fill="x")
        ttk.Button(btns, text="Copy", command=lambda: self._copy_text_widget(self.txt_prompt_preview)).pack(side="left")
        ttk.Button(btns, text="Reset to default", command=self._reset_prompt_to_default).pack(side="left", padx=(6,0))
        ttk.Label(frm, foreground="#555",
                  text="Note: Dynamic NLPI benchmarks use their own policy persona independent of this preview.").pack(anchor="w", pady=(6,0))
        self._refresh_prompt_preview()

    def _build_log_tab(self, parent: tk.Widget):
        frm = ttk.Frame(parent, padding=8); frm.pack(fill="both", expand=True)
        self.txt_log = tk.Text(frm, height=22, wrap="word", undo=True)
        self.txt_log.pack(fill="both", expand=True)
        menu = tk.Menu(self.txt_log, tearoff=0)
        menu.add_command(label="Copy", command=lambda: self._log_copy())
        menu.add_command(label="Select All", command=lambda: self._log_select_all())
        def popup(e):
            try: menu.tk_popup(e.x_root, e.y_root)
            finally: menu.grab_release()
        self.txt_log.bind("<Button-3>", popup)
        for key in ("<Control-c>", "<Command-c>"): self.txt_log.bind(key, lambda e: self._log_copy() or "break")
        for key in ("<Control-a>", "<Command-a>"): self.txt_log.bind(key, lambda e: self._log_select_all() or "break")
        btns = ttk.Frame(parent, padding=(8,4)); btns.pack(fill="x")
        ttk.Button(btns, text="Copy All", command=lambda: self._log_copy(all_text=True)).pack(side="left")
        ttk.Button(btns, text="Clear", command=lambda: self.txt_log.delete("1.0", "end")).pack(side="left", padx=6)

    # -------------------- Benchmarks table helpers --------------------

    def _on_benchmark_click(self, event):
        iid = self.tree_bm.identify_row(event.y)
        if not iid: return
        try:
            values = self.tree_bm.item(iid, "values"); key = values[1]
            if key in self._bench_vars:
                var = self._bench_vars[key]; var.set(not var.get())
            self._rebuild_benchmarks_tree()
        except (IndexError, KeyError): pass

    def _rebuild_benchmarks_tree(self):
        for iid in self.tree_bm.get_children():
            self.tree_bm.delete(iid)
        persona = PERSONA_TAGS.get(int(self.var_prompt_level.get() or 1), "")
        for key, desc in self._bench_defs:
            var = self._bench_vars.get(key)
            if var is None:
                var = tk.BooleanVar(value=True)
                self._bench_vars[key] = var
            d = desc
            if key == "NLPI" and var.get():
                d = f"{desc} | persona={persona}"
            self.tree_bm.insert("", "end", values=("✓" if var.get() else " ", key, d))

    def _ensure_llm_variants(self):
        added = 0
        if not PROMPT_PROFILES: return
        for m in self._ollama_models:
            if not m: continue
            for p in PROMPT_PROFILES.keys():
                key = f"NLPI[{m}|P{p}]"
                if not any(k == key for k,_ in self._bench_defs):
                    self._bench_defs.append((key, f"NLPI {m} (Prompt {p})"))
                    if key not in self._bench_vars: self._bench_vars[key] = tk.BooleanVar(value=False)
                    added += 1
        if added > 0: self._rebuild_benchmarks_tree(); self._log(f"Added {added} NLPI variants (model × prompt).")

    def _rebuild_llm_variants_for_selected_model(self):
        """
        Keep ONLY NLPI variants for the currently selected dropdown model.
        This prevents mixed-model runs in a single experiment.
        """
        sel_model = (self.var_model_name.get() or "").strip()
        if not sel_model:
            return

        # 1) remove existing NLPI[...|P#] entries from bench_defs and bench_vars
        keep_defs = []
        for k, d in self._bench_defs:
            if k.startswith("NLPI[") and "|" in k:
                # drop variant entries
                continue
            keep_defs.append((k, d))
        self._bench_defs = keep_defs

        # delete corresponding vars (NLPI[...] only)
        for k in list(self._bench_vars.keys()):
            if k.startswith("NLPI[") and "|" in k:
                del self._bench_vars[k]

        # 2) add variants ONLY for selected model
        added = 0
        for p in PROMPT_PROFILES.keys():
            key = f"NLPI[{sel_model}|P{p}]"
            if not any(k == key for k, _ in self._bench_defs):
                self._bench_defs.append((key, f"NLPI {sel_model} (Prompt {p})"))
                # default off to avoid accidental multi-variant runs
                self._bench_vars[key] = tk.BooleanVar(value=False)
                added += 1

        if added > 0:
            self._rebuild_benchmarks_tree()
            self._log(f"Rebuilt NLPI variants for selected model only: {sel_model} (added {added}).")



    def _ensure_benchmark_key(self, key: str, desc: str, default: bool = False):
        """Ensure a benchmark/NLPI key exists in the GUI benchmark table."""
        if not any(k == key for k, _ in self._bench_defs):
            self._bench_defs.append((key, desc))
        if key not in self._bench_vars:
            self._bench_vars[key] = tk.BooleanVar(value=default)

    def _selected_nlpi_models_personas(self):
        """Return selected NLPI models and personas from the visible checkbox table.

        Run Main+Export intentionally uses the currently checked rows on the
        Benchmarks tab. This prevents the export button from silently replacing
        the user's visible selection with a hard-coded preset.
        """
        selected_models = []
        personas_by_model = {}
        selected_keys = []
        pat = re.compile(r"^NLPI\[(.+?)\|P(\d+)\]$")
        for key, _desc in self._bench_defs:
            var = self._bench_vars.get(key)
            if var is None or not var.get():
                continue
            selected_keys.append(key)
            m = pat.match(key)
            if not m:
                continue
            model = m.group(1)
            persona = int(m.group(2))
            if model not in selected_models:
                selected_models.append(model)
            personas_by_model.setdefault(model, [])
            if persona not in personas_by_model[model]:
                personas_by_model[model].append(persona)
        for model in personas_by_model:
            personas_by_model[model] = sorted(personas_by_model[model])
        return selected_models, personas_by_model, selected_keys

    def _apply_main_nlpi_gemma4_preset(self):
        """Select the current main-model preset used for the NLPI paper.

        Current main models:
        - llama3.1:8b
        - gemma3:270m
        - gemma3:1b

        Personas:
        - P1~P5 only

        Qwen, Gemma4, gpt-oss, and DeepSeek are intentionally excluded from
        this preset and should be run separately as robustness or diagnostic
        models. The Run Main+Export button applies this preset automatically
        before launching the run.
        """
        main_models = ["llama3.1:8b", "gemma3:270m", "gemma3:1b"]
        main_personas = [1, 2, 3, 4, 5]
        base_benchmarks = ["EQUAL", "RiskParity", "MVP", "MOM6", "SHARPE"]

        # Make sure selected-model variants exist even when Ollama model probing
        # has not yet populated _ollama_models or when a model is installed but
        # not listed due to an intermittent Ollama API issue.
        for model in main_models:
            for p in main_personas:
                key = f"NLPI[{model}|P{p}]"
                self._ensure_benchmark_key(key, f"NLPI {model} (Prompt {p})", default=False)

        # Clear all selections first.
        for key in list(self._bench_vars.keys()):
            self._bench_vars[key].set(False)

        # Select benchmark controls.
        for key in base_benchmarks:
            if key in self._bench_vars:
                self._bench_vars[key].set(True)

        # Select main NLPI model × persona variants.
        for model in main_models:
            for p in main_personas:
                key = f"NLPI[{model}|P{p}]"
                if key in self._bench_vars:
                    self._bench_vars[key].set(True)

        # Paper-consistent defaults for GUI one-click main runs.
        self.var_start.set("2010-01-01")
        self.var_end.set("2025-12-29")
        self.var_reb.set(42)
        self.var_tcost.set(0.001)
        self.var_maxw.set(0.60)
        self.var_turn.set(0.25)
        self.var_use_llm.set(True)
        self.var_wfcv.set(True)
        self.var_holdout.set(False)
        if "llama3.1:8b" in (self._ollama_models or ["llama3.1:8b"]):
            self.var_model_name.set("llama3.1:8b")
        self._rebuild_benchmarks_tree()
        self._log("Applied main export preset: dates=2010-01-01~2025-12-29; benchmarks=EQUAL/RiskParity/MVP/MOM6/SHARPE; NLPI=llama3.1:8b, gemma3:270m, gemma3:1b × P1~P5.")

    def _default_gui_output_dir(self) -> str:
        import datetime
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_start = (self.ent_start.get() if hasattr(self, "ent_start") else self.var_start.get()).replace("-", "")
        safe_end = (self.ent_end.get() if hasattr(self, "ent_end") else self.var_end.get()).replace("-", "")
        outdir = os.path.join(os.getcwd(), "outputs", f"gui_main_nlpi_checked_{safe_start}_{safe_end}_{ts}")
        return outdir

    def on_run_main_nlpi_export(self):
        """Apply the paper main preset, run it, and export results.

        One-click main export uses the paper main configuration:
        - Start/End: 2010-01-01 ~ 2025-12-29
        - Benchmarks: EQUAL, RiskParity, MVP, MOM6, SHARPE
        - NLPI: llama3.1:8b, gemma3:270m, gemma3:1b × P1~P5
        - WFCV enabled, holdout disabled
        """
        if self.worker_future and not self.worker_future.done():
            messagebox.showinfo("Run", "A run is already in progress.")
            return

        # Force the main-paper setting before collecting selected rows.
        self._apply_main_nlpi_gemma4_preset()

        selected_models, personas_by_model, selected_keys = self._selected_nlpi_models_personas()
        if selected_models:
            self.var_use_llm.set(True)

        outdir = self._default_gui_output_dir()
        os.makedirs(outdir, exist_ok=True)
        self._auto_export_outdir = outdir
        self._auto_export_after_run = True
        self._auto_export_selected_models = selected_models
        self._auto_export_personas_by_model = personas_by_model
        self._auto_export_selected_keys = selected_keys
        self._log(f"Run Main+Export requested with forced paper main preset. NLPI models={selected_models}, personas={personas_by_model}. Results will be saved to: {outdir}")
        self.on_run()

    def _export_gui_outputs_to_folder(self, outdir: str, payload: dict | None = None):
        """Save the latest GUI results to an output folder without a file dialog."""
        os.makedirs(outdir, exist_ok=True)
        payload = payload or getattr(self, "_last_payload", {}) or {}

        def _safe_to_csv(df, filename, index=False):
            if isinstance(df, pd.DataFrame) and not df.empty:
                df.to_csv(os.path.join(outdir, filename), index=index)
                return True
            return False

        # Experiment configuration snapshot.
        try:
            import json, datetime
            cfg = {
                "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
                "source": "GUI Run Main+Export",
                "tickers": [t.strip() for t in self.var_tickers.get().split(",") if t.strip()],
                "start": (self.ent_start.get() if hasattr(self, "ent_start") else self.var_start.get()).strip(),
                "end": (self.ent_end.get() if hasattr(self, "ent_end") else self.var_end.get()).strip(),
                "rebalance_days": int(self.var_reb.get()),
                "tcost": float(self.var_tcost.get()),
                "max_weight": float(self.var_maxw.get()),
                "turnover_cap": float(self.var_turn.get()),
                "wfcv_on": bool(self.var_wfcv.get()),
                "holdout_on": bool(self.var_holdout.get()),
                "selected_benchmarks": [k for k, _ in self._bench_defs if self._bench_vars.get(k, tk.BooleanVar(value=False)).get()],
                "run_main_export_mode": "forced_paper_main_preset",
                "selected_nlpi_models": getattr(self, "_auto_export_selected_models", []),
                "selected_personas_by_model": getattr(self, "_auto_export_personas_by_model", {}),
                "selected_strategy_keys": getattr(self, "_auto_export_selected_keys", []),
                "preset_note": "Run Main+Export automatically applies the paper main preset: dates 2010-01-01 to 2025-12-29; benchmarks EQUAL/RiskParity/MVP/MOM6/SHARPE; NLPI llama3.1:8b, gemma3:270m, gemma3:1b x P1-P5.",
            }
            with open(os.path.join(outdir, "experiment_config_gui.json"), "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)
        except Exception as e:
            self._log(f"[WARN] Could not write experiment_config_gui.json: {e}")

        # Main performance table.
        try:
            met = getattr(self, "_last_metrics", {}) or {}
            if isinstance(met, dict) and met:
                dfm = pd.DataFrame(list(met.values()))
                _safe_to_csv(dfm, "performance_main.csv")
        except Exception as e:
            self._log(f"[WARN] performance_main.csv export failed: {e}")

        # WFCV fold and stitched performance table.
        try:
            wfcv = payload.get("wfcv", None) or getattr(self, "_last_wfcv", {}) or {}
            if isinstance(wfcv, dict):
                _safe_to_csv(wfcv.get("table"), "metrics_wfcv_folds.csv")
        except Exception as e:
            self._log(f"[WARN] metrics_wfcv_folds.csv export failed: {e}")

        # OOS tidy CSV.
        try:
            _safe_to_csv(getattr(self, "_oos_tidy_df", None), "oos_tidy.csv")
        except Exception as e:
            self._log(f"[WARN] oos_tidy.csv export failed: {e}")

        # Diagnostics tables.
        try:
            dsum = getattr(self, "_diag_summary_df", None)
            dts = getattr(self, "_diag_ts_df", None)
            if _safe_to_csv(dsum, "run_diagnostics.csv"):
                cols = [c for c in ["strategy_key", "strategy", "fold", "Avg_latency_sec", "n_calls", "JSON_valid_rate", "Parse_fail_rate", "Repair_rate", "Missing_asset_per_call", "Equal_fallback_rate"] if c in dsum.columns]
                if cols:
                    dsum[cols].to_csv(os.path.join(outdir, "model_latency.csv"), index=False)
            if _safe_to_csv(dts, "diagnostics_timeseries.csv") and "event" in dts.columns:
                dts[dts["event"] == "projection"].to_csv(os.path.join(outdir, "projection_diagnostics.csv"), index=False)
                dts[dts["event"] == "llm_call"].to_csv(os.path.join(outdir, "llm_call_diagnostics.csv"), index=False)
                dts[dts["event"] == "prompt_fidelity"].to_csv(os.path.join(outdir, "prompt_fidelity.csv"), index=False)
        except Exception as e:
            self._log(f"[WARN] diagnostics export failed: {e}")

        # Tech stats.
        try:
            _safe_to_csv(getattr(self, "_last_tech_stats_df", None), "tech_stats.csv")
        except Exception as e:
            self._log(f"[WARN] tech_stats.csv export failed: {e}")

        # Figures already visible in GUI.
        try:
            self.fig_wfcv.savefig(os.path.join(outdir, "fig_equity_curves_wfcv.png"), dpi=180, bbox_inches="tight")
            self.fig_oos.savefig(os.path.join(outdir, "fig_oos_equity.png"), dpi=180, bbox_inches="tight")
            self.fig_train.savefig(os.path.join(outdir, "fig_train_equity.png"), dpi=180, bbox_inches="tight")
            self.fig_test.savefig(os.path.join(outdir, "fig_test_equity.png"), dpi=180, bbox_inches="tight")
        except Exception as e:
            self._log(f"[WARN] figure export failed: {e}")

        self._log(f"GUI outputs exported to: {outdir}")
        return outdir



    def _bench_set_all(self, val: bool):
        for key, _ in self._bench_defs: self._bench_vars[key].set(val)
        self._rebuild_benchmarks_tree()

    # -------------------- Prompts helpers --------------------

    def _reset_prompt_to_default(self):
        self.var_prompt_level.set(1); self._refresh_prompt_preview()

    def _refresh_prompt_preview(self):
        lvl = int(self.var_prompt_level.get() or 1)
        txt = PROMPT_PROFILES.get(lvl, "")
        self.txt_prompt_preview.delete("1.0", "end")
        self.txt_prompt_preview.insert("1.0", txt)
        
    def _update_llm_bench_desc(self):
        try:
            key = "NLPI"
            if key in self._bench_desc_labels:
                base = "NLPI (LLM-based policy interface via Ollama)"
                persona = PERSONA_TAGS.get(int(self.var_prompt_level.get() or 1), "")
                var = self._bench_vars.get(key)
                if var and var.get():
                    desc = f"{base} | persona={persona}"
                else:
                    desc = base
                self._bench_desc_labels[key].configure(text=desc)
        except Exception:
            pass

    def _on_bench_toggle(self, key: str):
        if key == "NLPI":
            self._update_llm_bench_desc()

    # -------------------- Tree copy helpers --------------------

    def _attach_tree_copy_handlers(self, tree: ttk.Treeview, header_labels: List[str]):
        tree._header_labels = header_labels
        menu = tk.Menu(tree, tearoff=0)
        menu.add_command(label="Copy", command=lambda: self._tree_copy(tree, only_selected=True))
        menu.add_command(label="Copy All", command=lambda: self._tree_copy(tree, only_selected=False))
        menu.add_separator(); menu.add_command(label="Select All", command=lambda: self._tree_select_all(tree))
        def popup(e):
            try: menu.tk_popup(e.x_root, e.y_root)
            finally: menu.grab_release()
        tree.bind("<Button-3>", popup)
        for key in ("<Control-c>", "<Command-c>"): tree.bind(key, lambda e: (self._tree_copy(tree, only_selected=True), "break"))
        for key in ("<Control-a>", "<Command-a>"): tree.bind(key, lambda e: (self._tree_select_all(tree), "break"))

    def _tree_copy(self, tree: ttk.Treeview, only_selected: bool = True):
        try:
            cols = tree["columns"]; header = getattr(tree, "_header_labels", None)
            if not header or len(header) != len(cols): header = list(cols)
            items = tree.selection() if only_selected else tree.get_children()
            if not items: return
            rows = ["\t".join(str(h) for h in header)]
            for iid in items: rows.append("\t".join(str(v) for v in tree.item(iid, "values")))
            self.clipboard_clear(); self.clipboard_append("\n".join(rows))
        except Exception: pass

    def _copy_tree_as_tsv(self, tree: ttk.Treeview): self._tree_copy(tree, only_selected=False)
    def _tree_select_all(self, tree: ttk.Treeview):
        try: tree.selection_set(tree.get_children())
        except Exception: pass

    # -------------------- Plot copy helpers --------------------

    def _add_plot_copy_buttons(self, parent: tk.Widget, canvas: FigureCanvasTkAgg):
        btns = ttk.Frame(parent, padding=(8,4)); btns.pack(fill="x")
        ttk.Button(btns, text="Copy Plot", command=lambda: self._copy_plot_to_clipboard(canvas)).pack(side="left")

    def _copy_plot_to_clipboard(self, canvas: FigureCanvasTkAgg):
        try:
            buf = io.BytesIO()
            canvas.figure.tight_layout()
            canvas.figure.savefig(buf, format="png", dpi=300, bbox_inches="tight", pad_inches=0.3)
            tmp = os.path.join(os.path.abspath(os.path.expanduser("~")), "plot_export_tmp.png")
            with open(tmp, "wb") as f: f.write(buf.getvalue())
            self.clipboard_clear(); self.clipboard_append(tmp)
            self._log(f"Plot saved to {tmp} (path copied).")
        except Exception as e: self._log(f"[WARN] Copy plot failed: {e}")

    def _copy_text_widget(self, widget: tk.Text):
        try:
            text = widget.get("1.0", "end-1c")
            if text: self.clipboard_clear(); self.clipboard_append(text)
        except Exception: pass

    def _copy_current_tickers(self):
        try: self.clipboard_clear(); self.clipboard_append(self.var_tickers.get())
        except Exception: pass

    # -------------------- All Models / Per-Model tabs --------------------

    def _build_matrix_tab(self, parent: tk.Widget):
        frm = ttk.Frame(parent, padding=8); frm.pack(fill="both", expand=True)
        ttk.Label(frm, text="All strategies (benchmarks + NLPI variants) in a matrix view.").pack(anchor="w", pady=(0,6))
        cols = ("Model","Prompt","Strategy","Sharpe","Sortino","CAGR","Vol","MDD","TO")
        self.tree_matrix = ttk.Treeview(frm, show="headings", columns=cols, height=18, selectmode="extended")
        self.tree_matrix.pack(side="left", fill="both", expand=True)
        for c in cols:
            self.tree_matrix.heading(c, text=c)
            anchor = "w" if c in ("Model","Prompt","Strategy") else "e"
            width  = {"Strategy":220,"Model":160,"Prompt":80}.get(c, 110)
            self.tree_matrix.column(c, width=width, anchor=anchor, stretch=True)
        vs = ttk.Scrollbar(frm, orient="vertical", command=self.tree_matrix.yview)
        vs.pack(side="right", fill="y"); self.tree_matrix.configure(yscrollcommand=vs.set)
        self._attach_tree_copy_handlers(self.tree_matrix, header_labels=list(cols))
        ctrl = ttk.Frame(parent, padding=(8,0)); ctrl.pack(fill="x")
        ttk.Button(ctrl, text="Copy Table", command=lambda: self._copy_tree_as_tsv(self.tree_matrix)).pack(side="left")

    def _build_model_reports_tab(self, parent: tk.Widget):
        wrap = ttk.Frame(parent, padding=8); wrap.pack(fill="both", expand=True)
        sel = ttk.Frame(wrap); sel.pack(fill="x", pady=(0,8))
        ttk.Label(sel, text="NLPI Model:").pack(side="left")
        self.cbo_model_filter = ttk.Combobox(sel, state="readonly", width=24, values=[])
        self.cbo_model_filter.pack(side="left", padx=(6,12))
        self.cbo_model_filter.bind("<<ComboboxSelected>>", lambda e: self._refresh_model_report())

        ttk.Label(sel, text="Include Benchmarks").pack(side="left", padx=(6,6))
        self.var_model_view_bench = tk.BooleanVar(value=True)
        ttk.Checkbutton(sel, variable=self.var_model_view_bench).pack(side="left")
        ttk.Button(sel, text="Refresh", command=self._refresh_model_report).pack(side="left", padx=(12,0))

        chart = ttk.Frame(wrap); chart.pack(fill="both", expand=True)
        self.fig_model, self.ax_model, self.canvas_model = self._create_plot_canvas(chart)
        self._add_plot_copy_buttons(wrap, self.canvas_model)

        cols = ("Strategy","Sharpe","Sortino","CAGR","Vol","MDD","TO")
        self.tree_model = ttk.Treeview(wrap, show="headings", columns=cols, height=10, selectmode="extended")
        self.tree_model.pack(fill="both", expand=False, padx=(0,6), pady=(6,0))
        for c in cols:
            self.tree_model.heading(c, text=c)
            self.tree_model.column(c, width=140 if c=="Strategy" else 100, anchor="w" if c=="Strategy" else "e", stretch=True)
        self._attach_tree_copy_handlers(self.tree_model, header_labels=list(cols))
        ctrl = ttk.Frame(wrap, padding=(8,6)); ctrl.pack(fill="x")
        ttk.Button(ctrl, text="Copy Table", command=lambda: self._copy_tree_as_tsv(self.tree_model)).pack(side="left")
                 
    def _build_wfcv_model_tab(self, parent: tk.Widget):
        wrap = ttk.Frame(parent, padding=8); wrap.pack(fill="both", expand=True)
        sel = ttk.Frame(wrap); sel.pack(fill="x", pady=(0,8))
        ttk.Label(sel, text="NLPI Model:").pack(side="left")
        self.cbo_wfcv_model = ttk.Combobox(sel, state="readonly", width=24, values=[])
        self.cbo_wfcv_model.pack(side="left", padx=(6,12))
        self.cbo_wfcv_model.bind("<<ComboboxSelected>>", lambda e: self._refresh_wfcv_model_report())

        ttk.Label(sel, text="Include Benchmarks").pack(side="left", padx=(6,6))
        self.var_wfcv_view_bench = tk.BooleanVar(value=True)
        ttk.Checkbutton(sel, variable=self.var_wfcv_view_bench,
                        command=self._refresh_wfcv_model_report).pack(side="left")
        ttk.Button(sel, text="Refresh", command=self._refresh_wfcv_model_report).pack(side="left", padx=(12,0))

        chart = ttk.Frame(wrap); chart.pack(fill="both", expand=True)
        self.fig_wfcv_model, self.ax_wfcv_model, self.canvas_wfcv_model = self._create_plot_canvas(chart)
        self._add_plot_copy_buttons(wrap, self.canvas_wfcv_model)

        cols = ("Strategy","Sharpe","Sortino","CAGR","Vol","MDD","TO")
        self.tree_wfcv_model = ttk.Treeview(wrap, show="headings", columns=cols, height=14, selectmode="extended")
        self.tree_wfcv_model.pack(fill="both", expand=True, padx=(0,6), pady=(6,0))
        for c in cols:
            self.tree_wfcv_model.heading(c, text=c)
            self.tree_wfcv_model.column(c, width=140 if c=="Strategy" else 100,
                                        anchor="w" if c=="Strategy" else "e", stretch=True)
        self._attach_tree_copy_handlers(self.tree_wfcv_model, header_labels=list(cols))
        ctrl = ttk.Frame(wrap, padding=(8,6)); ctrl.pack(fill="x")
        ttk.Button(ctrl, text="Copy Table", command=lambda: self._copy_tree_as_tsv(self.tree_wfcv_model)).pack(side="left")

    def _refresh_wfcv_model_report(self):
        info = self._last_wfcv_info or {}
        stitched = info.get("stitched", {}) or {}
        table_df = info.get("table", None)

        def parse_model(name: str) -> str:
            if name.startswith("NLPI[") and "|" in name and name.endswith("]"):
                return name[4:-1].split("|", 1)[0]
            return ""

        models = sorted({parse_model(nm) for nm in stitched.keys() if nm.startswith("NLPI[")})
        if getattr(self, "cbo_wfcv_model", None):
            curvals = list(self.cbo_wfcv_model["values"]) if self.cbo_wfcv_model["values"] else []
            if models != list(curvals):
                self.cbo_wfcv_model["values"] = models
            if not self.cbo_wfcv_model.get() and models:
                self.cbo_wfcv_model.set(models[0])

        model = self.cbo_wfcv_model.get().strip() if getattr(self, "cbo_wfcv_model", None) else ""
        if not model:
            self.ax_wfcv_model.clear(); self.canvas_wfcv_model.draw()
            for iid in self.tree_wfcv_model.get_children(): self.tree_wfcv_model.delete(iid)
            return

        include_bench = bool(self.var_wfcv_view_bench.get()) if hasattr(self, "var_wfcv_view_bench") else True

        def is_target_llm(name: str) -> bool:
            return name.startswith("NLPI[") and f"NLPI[{model}|" in name

        target_names = sorted([nm for nm in stitched.keys() if is_target_llm(nm)])
        bench_names  = []
        if include_bench:
            bench_names = [nm for nm in stitched.keys()
                           if nm in ("EQUAL","RiskParity","MVP","LW-MVP","HRP","BL","MOM6","TRND6","SHARPE","SORTINO","CODED_P1","CODED_P2","CODED_P3","CODED_P4","CODED_P5")]            

        ax = self.ax_wfcv_model
        ax.clear()

        def plot_stitched(name: str, lw=2.2):
            idx_eq = stitched.get(name)
            if not idx_eq: return
            idx, eq = idx_eq
            ax.plot(idx, eq, label=name, linewidth=lw)

        for nm in target_names:
            plot_stitched(nm, lw=2.4)
        for nm in bench_names:
            plot_stitched(nm, lw=1.8)

        ax.set_title(f"Per-WFCV Out-of-Sample Equity (Concatenated Folds) — {model}")
        ax.set_xlabel("Date"); ax.set_ylabel("Equity (initial=1)")
        if target_names or bench_names: 
            #ax.legend(loc="best")
            self._legend_outside(ax, ax.figure , right=0.78)   # ← 교체
        self.canvas_wfcv_model.draw()

        self.tree_wfcv_model.delete(*self.tree_wfcv_model.get_children())
        if isinstance(table_df, pd.DataFrame) and not table_df.empty:
            df_all = table_df.copy()
            df_all = df_all[df_all["Fold"].astype(str) == "ALL"]
            show_names = target_names + bench_names
            df_all = df_all[df_all["Strategy"].isin(show_names)]
            for _, row in df_all.iterrows():
                vals = (
                    row.get("Strategy",""),
                    f"{float(row.get('Sharpe',np.nan)):.4f}" if pd.notna(row.get('Sharpe')) else "",
                    f"{float(row.get('Sortino',np.nan)):.4f}"   if pd.notna(row.get('Sortino'))   else "",
                    f"{float(row.get('CAGR',np.nan)):.4f}"  if pd.notna(row.get('CAGR'))  else "",
                    f"{float(row.get('Vol',np.nan)):.4f}"   if pd.notna(row.get('Vol'))   else "",
                    f"{float(row.get('MDD',np.nan)):.4f}"   if pd.notna(row.get('MDD'))   else "",
                    f"{float(row.get('TO',np.nan)):.4f}"   if pd.notna(row.get('TO'))   else "",
                )
                self.tree_wfcv_model.insert("", "end", values=vals)
    # -------------------------
    # OOS Export: tidy CSV + OOS-only plot
    # -------------------------
    def _build_oos_export_tab(self, parent: tk.Widget):
        wrap = ttk.Frame(parent, padding=8); wrap.pack(fill="both", expand=True)

        top = ttk.Frame(wrap); top.pack(fill="x", pady=(0,8))
        ttk.Label(top, text="View:").pack(side="left")
        # ---------- OOS View Controls (NEW) ----------
        ctrl = ttk.Frame(wrap)
        ctrl.pack(fill="x", pady=5)

        ttk.Label(ctrl, text="Fold:").pack(side="left", padx=(5, 2))

        #self.var_oos_fold_sel = tk.StringVar(value="All")
        self.cbo_oos_fold = ttk.Combobox(
            ctrl,
            textvariable=self.var_oos_fold_sel,
            values=["All", "1", "2", "3", "4", "5"],
            width=6,
            state="readonly"
        )
        self.cbo_oos_fold.pack(side="left", padx=5)

        self.var_oos_show_stitched = tk.BooleanVar(value=True)
        #self.var_oos_show_fold = tk.BooleanVar(value=False)

        ttk.Checkbutton(
            ctrl,
            text="Show stitched only",
            variable=self.var_oos_show_stitched,
            command=self._sync_oos_export_controls
        ).pack(side="left", padx=10)

        ttk.Checkbutton(
            ctrl,
            text="Show fold only",
            variable=self.var_oos_show_fold,
            command=self._sync_oos_export_controls
        ).pack(side="left", padx=10)

        self.cbo_oos_fold.bind("<<ComboboxSelected>>", lambda e: self._sync_oos_export_controls())

        ttk.Radiobutton(top, text="Stitched OOS", variable=self.var_oos_segment,
                        value="stitched_oos", command=self._plot_oos_filtered).pack(side="left", padx=(8,0))
        ttk.Radiobutton(top, text="Fold OOS", variable=self.var_oos_segment,
                        value="fold_oos", command=self._plot_oos_filtered).pack(side="left", padx=(8,0))

        ttk.Label(top, text="Model:").pack(side="left", padx=(16,0))
        self.cbo_oos_model = ttk.Combobox(top, state="readonly", width=22, values=[])
        self.cbo_oos_model.pack(side="left", padx=(6,0))
        self.cbo_oos_model.bind("<<ComboboxSelected>>", lambda e: self._plot_oos_filtered())

        #self.var_oos_include_bench = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text="Include Benchmarks", variable=self.var_oos_include_bench,
                        command=self._plot_oos_filtered).pack(side="left", padx=(12,0))

        ttk.Button(top, text="Plot", command=self._plot_oos_filtered).pack(side="right")
        ttk.Button(top, text="Save tidy CSV", command=self._save_oos_csv).pack(side="right", padx=(0,8))
        ttk.Button(top, text="Copy tidy CSV", command=self._copy_oos_csv).pack(side="right", padx=(0,8))

        chart = ttk.Frame(wrap); chart.pack(fill="both", expand=True)
        self.fig_oos, self.ax_oos, self.canvas_oos = self._create_plot_canvas(chart)
        self._add_plot_copy_buttons(wrap, self.canvas_oos)

        prev = ttk.LabelFrame(wrap, text="Tidy CSV preview (first 200 rows)")
        prev.pack(fill="both", expand=False, pady=(8,0))
        self.txt_oos_preview = scrolledtext.ScrolledText(prev, height=10, wrap="none")
        self.txt_oos_preview.pack(fill="both", expand=True)

        self._oos_tidy_df = None



    def _parse_strategy_meta(self, name: str):
        """Return (strategy, model, persona) for a plotted series key."""
        if not isinstance(name, str):
            return ("NA","NA","NA")

        # NLPI[model|P1]
        m = re.match(r"^NLPI\[(?P<model>[^\]|]+)\|(?P<persona>P\d)\]$", name)
        if m:
            return ("NLPI", m.group("model"), m.group("persona"))

        if name == "NLPI":
            return ("NLPI", "", "")

        # CODED_P1
        m = re.match(r"^CODED_P(?P<p>\d)$", name)
        if m:
            return (name, "NA", f"P{m.group('p')}")

        return (name, "NA", "NA")

    def _build_oos_tidy_from_payload(self, payload: dict):
        """Build a single tidy DataFrame for OOS equity curves (fold + stitched)."""
        info = (payload or {}).get("wfcv", {}) or {}
        overlay = info.get("overlay", {}) or {}
        stitched = info.get("stitched", {}) or {}

        rows = []
        universe = (payload or {}).get("universe", "") or ""

        # Fold-level OOS (overlay)
        for name, lst in overlay.items():
            if not lst:
                continue
            strategy, model, persona = self._parse_strategy_meta(name)
            for fold_idx, pair in enumerate(lst, start=1):
                try:
                    idx, eq = pair
                except Exception:
                    continue
                if idx is None or eq is None:
                    continue
                for d, v in zip(idx, eq):
                    rows.append({
                        "date": pd.to_datetime(d),
                        "strategy": strategy,
                        "model": model if model else "NA",
                        "persona": persona if persona else "NA",
                        "fold": fold_idx,
                        "segment": "fold_oos",
                        "equity": float(v),
                        "is_oos": 1,
                        "universe": universe if universe else "NA",
                    })

        # Stitched OOS
        for name, pair in stitched.items():
            try:
                idx, eq = pair
            except Exception:
                continue
            if idx is None or eq is None:
                continue
            strategy, model, persona = self._parse_strategy_meta(name)
            for d, v in zip(idx, eq):
                rows.append({
                    "date": pd.to_datetime(d),
                    "strategy": strategy,
                    "model": model if model else "NA",
                    "persona": persona if persona else "NA",
                    "fold": 0,
                    "segment": "stitched_oos",
                    "equity": float(v),
                    "is_oos": 1,
                    "universe": universe if universe else "NA",
                })

        if rows:
            df = pd.DataFrame(rows)
            df = df.sort_values(["segment","fold","strategy","model","persona","date"]).reset_index(drop=True)
        else:
            df = pd.DataFrame(columns=["date","strategy","model","persona","fold","segment","equity","is_oos","universe"])

        self._oos_tidy_df = df

        # Update model list for OOS plotting
        models = sorted([m for m in df["model"].dropna().unique().tolist() if m not in ("NA", "")])
        if getattr(self, "cbo_oos_model", None):
            self.cbo_oos_model["values"] = ["(all)"] + models
            if not self.cbo_oos_model.get():
                self.cbo_oos_model.set("(all)")

    def _refresh_oos_export_tab(self):
        if getattr(self, "txt_oos_preview", None) is None:
            return
        df = getattr(self, "_oos_tidy_df", None)
        self.txt_oos_preview.delete("1.0", "end")
        if isinstance(df, pd.DataFrame) and not df.empty:
            self.txt_oos_preview.insert("end", df.head(200).to_csv(index=False))
        else:
            self.txt_oos_preview.insert("end", "(no OOS data yet)\nRun a backtest to populate this table.")
        self._plot_oos_filtered()
        
    def _sync_oos_export_controls(self):
        fold_sel = self.var_oos_fold_sel.get()

        if fold_sel != "All":
            # 특정 fold 선택 → 자동 fold-only
            self.var_oos_show_stitched.set(False)
            self.var_oos_show_fold.set(True)
        else:
            # All 선택 시 stitched / fold 토글 상호배타 유지
            if self.var_oos_show_stitched.get():
                self.var_oos_show_fold.set(False)
            elif self.var_oos_show_fold.get():
                self.var_oos_show_stitched.set(False)
            else:
                self.var_oos_show_stitched.set(True)

        self._plot_oos_filtered()


    def _plot_oos_filtered(self):
        df = getattr(self, "_oos_tidy_df", None)
        ax = getattr(self, "ax_oos", None)
        if ax is None:
            return
        ax.clear()

        if not isinstance(df, pd.DataFrame) or df.empty:
            self.canvas_oos.draw()
            return

        fold_sel = self.var_oos_fold_sel.get()
        show_fold = self.var_oos_show_fold.get()

        # --- NEW fold selection logic ---

        if fold_sel != "All":
            # 특정 fold 선택 → 해당 fold의 fold_oos만 표시
            segment = "fold_oos"
            fold_filter = int(fold_sel)
        else:
            # All 선택
            if show_fold:
                segment = "fold_oos"
            else:
                segment = "stitched_oos"
            fold_filter = None

        include_bench = bool(self.var_oos_include_bench.get()) if hasattr(self, "var_oos_include_bench") else True
        model_sel = self.cbo_oos_model.get().strip() if getattr(self, "cbo_oos_model", None) else "(all)"

        dfx = df[df["segment"] == segment].copy()
        
        if fold_filter is not None:
            dfx = dfx[dfx["fold"] == fold_filter]
            
        if model_sel and model_sel != "(all)":
            # keep selected model's NLPI curves; keep non-NLPI baselines
            is_llm = dfx["strategy"].eq("NLPI")
            dfx = dfx[(~is_llm) | (dfx["model"].eq(model_sel))]
            
        if dfx.empty:
            ax.set_title("No data available for selected configuration")
            self.canvas_oos.draw()
            return


        if not include_bench:
            dfx = dfx[dfx["strategy"].eq("NLPI")]

        if segment == "fold_oos":
            for (strategy, model, persona, fold), g in dfx.groupby(["strategy","model","persona","fold"], dropna=False):
                lbl = f"{strategy}[{model}|{persona}] (F{int(fold)})" if strategy == "NLPI" else f"{strategy} (F{int(fold)})"
                ax.plot(g["date"], g["equity"], label=lbl, linewidth=1.0)
        else:
            for (strategy, model, persona), g in dfx.groupby(["strategy","model","persona"], dropna=False):
                lbl = f"NLPI[{model}|{persona}]" if strategy == "NLPI" else strategy
                ax.plot(g["date"], g["equity"], label=lbl, linewidth=2.2)

        title_model = model_sel if model_sel and model_sel != "(all)" else "all models"
        ax.set_title(f"OOS Equity ({segment.replace('_',' ')}) — {title_model}")
        ax.set_xlabel("Date"); ax.set_ylabel("Equity (initial=1)")
        self._legend_outside(ax, ax.figure, right=0.78)
        self.canvas_oos.draw()

    def _copy_oos_csv(self):
        df = getattr(self, "_oos_tidy_df", None)
        if not isinstance(df, pd.DataFrame) or df.empty:
            messagebox.showinfo("Copy", "No OOS data to copy yet.")
            return
        s = df.to_csv(index=False)
        if len(s) > 2_000_000:
            s = df.head(5000).to_csv(index=False)
            messagebox.showinfo("Copy", "CSV is very large. Copied first 5,000 rows.")
        self.clipboard_clear()
        self.clipboard_append(s)
        self.update()

    def _save_oos_csv(self):
        df = getattr(self, "_oos_tidy_df", None)
        if not isinstance(df, pd.DataFrame) or df.empty:
            messagebox.showinfo("Save", "No OOS data to save yet.")
            return
        fpath = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("All files", "*")],
            title="Save tidy OOS equity CSV"
        )
        if not fpath:
            return
        df.to_csv(fpath, index=False)
        self._log(f"Saved tidy OOS CSV: {fpath}")

    # -------------------------
    # Diagnostics tab (NEW)
    # -------------------------
    def _build_diagnostics_tab(self, parent: tk.Widget):
        wrap = ttk.Frame(parent, padding=8)
        wrap.pack(fill="both", expand=True)

        top = ttk.Frame(wrap)
        top.pack(fill="x", pady=(0, 6))

        ttk.Button(top, text="Refresh", command=self._refresh_diagnostics_tab).pack(side="left")
        ttk.Button(top, text="Copy Summary", command=self._copy_diag_summary).pack(side="left", padx=(6, 0))
        ttk.Button(top, text="Export Reviewer Response Tables", command=self._export_reviewer_response_tables).pack(side="left", padx=(6, 0))

        ttk.Button(top, text="Save run_diagnostics.csv", command=self._save_diag_summary).pack(side="right")
        ttk.Button(top, text="Save diagnostics_timeseries.csv", command=self._save_diag_ts).pack(side="right", padx=(0, 8))

        # Summary preview
        frm1 = ttk.LabelFrame(wrap, text="run_diagnostics (summary)")
        frm1.pack(fill="both", expand=False, pady=(6, 6))
        self.txt_diag_summary = scrolledtext.ScrolledText(frm1, height=10, wrap="none")
        self.txt_diag_summary.pack(fill="both", expand=True)

        # Timeseries preview (head)
        frm2 = ttk.LabelFrame(wrap, text="diagnostics_timeseries (head)")
        frm2.pack(fill="both", expand=True)
        self.txt_diag_ts = scrolledtext.ScrolledText(frm2, height=14, wrap="none")
        self.txt_diag_ts.pack(fill="both", expand=True)

        # storage
        self._diag_summary_df = None
        self._diag_ts_df = None

        self._refresh_diagnostics_tab()


    def _refresh_diagnostics_tab(self):
        # summary
        self.txt_diag_summary.delete("1.0", "end")
        df1 = getattr(self, "_diag_summary_df", None)
        if isinstance(df1, pd.DataFrame) and not df1.empty:
            self.txt_diag_summary.insert("end", df1.to_csv(index=False))
        else:
            self.txt_diag_summary.insert("end", "(no diagnostics summary yet)\nRun backtest with diagnostics enabled.\n")

        # timeseries
        self.txt_diag_ts.delete("1.0", "end")
        df2 = getattr(self, "_diag_ts_df", None)
        if isinstance(df2, pd.DataFrame) and not df2.empty:
            self.txt_diag_ts.insert("end", df2.head(200).to_csv(index=False))
        else:
            self.txt_diag_ts.insert("end", "(no diagnostics timeseries yet)\n")


    def _copy_diag_summary(self):
        df1 = getattr(self, "_diag_summary_df", None)
        if not isinstance(df1, pd.DataFrame) or df1.empty:
            messagebox.showinfo("Copy", "No diagnostics summary to copy yet.")
            return
        s = df1.to_csv(index=False)
        self.clipboard_clear()
        self.clipboard_append(s)
        self.update()


    def _save_diag_summary(self):
        df1 = getattr(self, "_diag_summary_df", None)
        if not isinstance(df1, pd.DataFrame) or df1.empty:
            messagebox.showinfo("Save", "No diagnostics summary to save yet.")
            return
        fpath = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("All files", "*")],
            title="Save run_diagnostics.csv"
        )
        if not fpath:
            return
        df1.to_csv(fpath, index=False)
        self._log(f"Saved run_diagnostics.csv: {fpath}")


    def _save_diag_ts(self):
        df2 = getattr(self, "_diag_ts_df", None)
        if not isinstance(df2, pd.DataFrame) or df2.empty:
            messagebox.showinfo("Save", "No diagnostics timeseries to save yet.")
            return
        fpath = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("All files", "*")],
            title="Save diagnostics_timeseries.csv"
        )
        if not fpath:
            return
        df2.to_csv(fpath, index=False)
        self._log(f"Saved diagnostics_timeseries.csv: {fpath}")




    def _export_reviewer_response_tables(self):
        """Export reviewer-response CSVs from the latest GUI run.

        Creates tables for projection diagnostics, NLPI-call diagnostics,
        prompt fidelity, latency, and clean performance comparison.
        """
        outdir = filedialog.askdirectory(title="Choose folder for reviewer-response tables")
        if not outdir:
            return
        os.makedirs(outdir, exist_ok=True)

        try:
            # Performance comparison
            met = getattr(self, "_last_metrics", {}) or {}
            if isinstance(met, dict) and met:
                dfm = pd.DataFrame(list(met.values()))
                dfm.to_csv(os.path.join(outdir, "performance_main.csv"), index=False)
                keep = ["NLPI", "NLPI[gpt-oss:20b|P5]", "CODED_P5", "EQUAL", "EQ", "MVP", "LW-MVP", "HRP", "BL", "RiskParity", "RP"]
                if "Strategy" in dfm.columns:
                    dfm[dfm["Strategy"].isin(keep)].to_csv(os.path.join(outdir, "performance_clean_comparison.csv"), index=False)

            dsum = getattr(self, "_diag_summary_df", None)
            dts = getattr(self, "_diag_ts_df", None)
            if isinstance(dsum, pd.DataFrame) and not dsum.empty:
                dsum.to_csv(os.path.join(outdir, "run_diagnostics.csv"), index=False)
                cols = [c for c in ["strategy_key", "strategy", "Avg_latency_sec", "n_calls", "JSON_valid_rate", "Parse_fail_rate", "Repair_rate"] if c in dsum.columns]
                dsum[cols].to_csv(os.path.join(outdir, "model_latency.csv"), index=False)
            if isinstance(dts, pd.DataFrame) and not dts.empty:
                dts.to_csv(os.path.join(outdir, "diagnostics_timeseries.csv"), index=False)
                if "event" in dts.columns:
                    dts[dts["event"] == "projection"].to_csv(os.path.join(outdir, "projection_diagnostics.csv"), index=False)
                    dts[dts["event"] == "llm_call"].to_csv(os.path.join(outdir, "llm_call_diagnostics.csv"), index=False)
                    dts[dts["event"] == "prompt_fidelity"].to_csv(os.path.join(outdir, "prompt_fidelity.csv"), index=False)

            self._log(f"Exported reviewer-response tables to: {outdir}")
            messagebox.showinfo("Export", f"Reviewer-response tables exported to:\n{outdir}")
        except Exception as e:
            messagebox.showerror("Export error", str(e))


    def _update_all_models_matrix(self):
        try:
            tree = self.tree_matrix
        except Exception:
            return

        for iid in tree.get_children():
            tree.delete(iid)

        def parse_name(name: str):
            model, prompt = "", ""
            if name.startswith("NLPI[") and "|" in name and name.endswith("]"):
                body = name[4:-1]
                model, pr = body.split("|", 1)
                prompt = pr.replace("P", "").strip()
            return model, prompt

        for name, s in (self._last_metrics or {}).items():
            model, prompt = parse_name(name)
            if not isinstance(s, dict):
                continue

            def get(k, default=np.nan):
                try:
                    return float(s.get(k, default))
                except Exception:
                    return default

            row = (model or "-", prompt or "-", name,
                get("Sharpe"), get("Sortino"), get("CAGR"),
                get("Vol"), get("MDD"), get("TO"))
            tree.insert("", "end", values=row)

        models = sorted({parse_name(n)[0] for n in (self._last_metrics or {}).keys() if n.startswith("NLPI[")})
        if getattr(self, "cbo_model_filter", None):
            self.cbo_model_filter["values"] = models
            if not self.cbo_model_filter.get() and models:
                self.cbo_model_filter.set(models[0])

    def _refresh_model_report(self):
        model = self.cbo_model_filter.get().strip() if getattr(self, "cbo_model_filter", None) else ""
        if not model:
            return
        include_bench = bool(self.var_model_view_bench.get()) if hasattr(self, "var_model_view_bench") else True

        target_rows = []
        bench_rows  = []
        for name, s in (self._last_metrics or {}).items():
            if not isinstance(s, dict):
                continue
            if name.startswith("NLPI[") and f"NLPI[{model}|" in name:
                target_rows.append((name, s))
            elif include_bench and name in ("EQUAL","RiskParity","MVP","LW-MVP","HRP","BL",
                                "MOM6","TRND6","SHARPE","SORTINO",
                                "CODED_P1","CODED_P2","CODED_P3","CODED_P4","CODED_P5"):
                bench_rows.append((name, s))

        tree = self.tree_model
        for iid in tree.get_children():
            tree.delete(iid)

        def getf(s, k):
            try:
                return float(s.get(k, np.nan))
            except Exception:
                return np.nan

        def add_row(name, s):
            tree.insert("", "end",
                        values=(name, getf(s,"Sharpe"), getf(s,"Sortino"), getf(s,"CAGR"),
                                getf(s,"Vol"), getf(s,"MDD"), getf(s,"TO")))

        for name, s in sorted(target_rows, key=lambda x: x[0]):
            add_row(name, s)
        for name, s in bench_rows:
            add_row(name, s)

        ax = self.ax_model
        ax.clear()

        def plot_equity(name):
            r = (self._last_res_test or {}).get(name, None)
            if isinstance(r, pd.Series) and not r.empty:
                eq = (1.0 + r).cumprod()
                ax.plot(eq.index, eq.values, label=name)

        for name, _ in sorted(target_rows, key=lambda x: x[0]):
            plot_equity(name)
        if include_bench:
            for name, _ in bench_rows:
                plot_equity(name)

        ax.set_title(f"Per-Model Test Equity Curves — {model}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Equity (initial=1)")
        #ax.legend(loc="best")
        self._legend_outside(ax, ax.figure , right=0.78)   # ← ax.figure 사용
        self.canvas_model.draw()

    # -------------------- Ollama probe --------------------

    def _probe_ollama_on_start(self): self.after(50, self._async_probe_models)
    def on_ollama_connect(self): self._async_probe_models()

    def _async_probe_models(self):
        def task():
            base = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
            ok, msg = check_ollama(base); self.msg_q.put((MSG_INFO, f"Ollama: {msg}"))
            if ok:
                try:
                    import requests
                    r = requests.get(base + "/api/tags", timeout=5.0)
                    models = [m.get("model") or m.get("name") for m in r.json().get("models", [])] if r.status_code == 200 else []
                    self.msg_q.put(("models", models)); self.msg_q.put(("ollama_ok", True))
                except Exception as e:
                    self.msg_q.put(("models", [])); self.msg_q.put(("ollama_ok", False)); self.msg_q.put((MSG_INFO, f"Model fetch failed: {e}"))
            else:
                self.msg_q.put(("models", [])); self.msg_q.put(("ollama_ok", False))
        threading.Thread(target=task, daemon=True).start()

    # -------------------- Portfolio actions --------------------

    def on_apply_portfolio(self):
        sel = self.lst_ports.curselection()
        if not sel: return messagebox.showinfo("Portfolio", "Select a portfolio on the left.")
        name = self.lst_ports.get(sel[0]); ticks = PORTFOLIOS.get(name, [])
        self.var_selected_port.set(name); self.var_tickers.set(",".join(ticks))
        self.lst_tickers.delete(0, "end")
        for t in ticks: self.lst_tickers.insert("end", t)
        self._log(f"Applied portfolio: {name} ({len(ticks)} tickers)")

    # -------------------- Run / Stop --------------------

    def on_run(self):
        if self.worker_future and not self.worker_future.done(): return
        tickers = [t.strip() for t in self.var_tickers.get().split(",") if t.strip()]
        if not tickers: return messagebox.showwarning("Input", "Please enter at least one ticker.")
        try:
            self.update_idletasks()
        except Exception:
            pass
        start_str = (self.ent_start.get() if hasattr(self, "ent_start") else self.var_start.get()).strip()
        end_str = (self.ent_end.get() if hasattr(self, "ent_end") else self.var_end.get()).strip()


        params = {
                    "tickers": tickers, "start": start_str, "end": end_str,
                    "reb": int(self.var_reb.get()), "tcost": float(self.var_tcost.get()), "maxw": float(self.var_maxw.get()),
                    "turn": float(self.var_turn.get()), "use_llm": bool(self.var_use_llm.get()),
                    "model": self.var_model_name.get().strip() or os.getenv("OLLAMA_MODEL","gpt-oss:20b"),
                    "stride": max(1, int(self.var_prog_stride.get() or 1)), "wfcv_on": bool(self.var_wfcv.get()), 
                    "holdout_on": bool(self.var_holdout.get()),
                    "selected_bench": [k for k,_ in self._bench_defs if self._bench_vars.get(k, tk.BooleanVar(value=False)).get()],
                }

        self.lbl_status.configure(text="Running..."); self.pb.start(50)
        # clear plots/tables
        self.ax_train.clear(); self.canvas_train.draw()
        self.ax_test.clear();  self.canvas_test.draw()
        self.ax_wfcv.clear();  self.canvas_wfcv.draw()
        for tree in (self.tree, self.tree_wfcv, self.tree_stats, self.tree_sig): 
            tree.delete(*tree.get_children())
        if hasattr(self, "tree_all_models"):
            self.tree_all_models.delete(*self.tree_all_models.get_children())
        if hasattr(self, "txt_model_report"):
            self.txt_model_report.delete("1.0", "end")
        self.tree_stats["columns"] = ("Ticker",); self.tree_stats.heading("Ticker", text="Ticker")
        self.tree_stats.column("Ticker", width=140, anchor="w", stretch=True); self.tree_stats._header_labels = ["Ticker"]
        self.txt_few.delete("1.0","end")
        self.cancel_event.clear()
        self._last_res_test = {}
        self._log(f"Run requested. Tickers={len(params['tickers'])}, benchmarks={params['selected_bench']}")
        self.worker_future = self.exec.submit(self._worker, **params)
        
        # clear diagnostics
        self._diag_summary_df = None
        self._diag_ts_df = None
        if hasattr(self, "txt_diag_summary"):
            self._refresh_diagnostics_tab()
            
        self._log(
            f"Run config: wfcv_on={params['wfcv_on']} | holdout_on={params['holdout_on']} | "
            f"reb={params['reb']} | start={params['start']} | end={params['end']}"
        )


    def on_stop(self):
        self.cancel_event.set(); self.lbl_status.configure(text="Cancel requested…"); self._log("Cancel requested by user.")

    # -------------------- Worker --------------------

    def _worker(self, tickers, start, end, reb, tcost, maxw, turn, use_llm, model, stride, wfcv_on, holdout_on, selected_bench):
        try:
            if self.cancel_event.is_set(): return self.msg_q.put((MSG_CANCELED, None))

            self.msg_q.put((MSG_INFO, f"Downloading prices: {tickers}"))
            
            prices = None
            for attempt in range(3):
                prices = fetch_prices_yf(tickers, start, end)
                if prices is not None and not prices.empty:
                    break
                self.msg_q.put((MSG_INFO, f"[WARN] Price fetch empty. retry {attempt+1}/3"))
                import time; time.sleep(1.5 * (attempt + 1))

            if prices is None or prices.empty or len(prices) < 60:
                return self.msg_q.put((MSG_ERROR, f"Not enough data ({0 if prices is None else len(prices)} rows)."))

            self.msg_q.put((MSG_INFO, "Computing features..."))
            feats = make_features(prices, tickers); prices = prices.loc[feats.index]

            # Tech stats quick view
            try:
                rets = (prices[tickers].ffill().pct_change(fill_method=None)
                        .replace([np.inf, -np.inf], np.nan).dropna(how="all").fillna(0.0))
                desc_full = rets.describe().T.join(pd.DataFrame({"Skew": rets.skew(), "Kurtosis": rets.kurt()})).round(6).reset_index().rename(columns={"index": "Ticker"})
                self.msg_q.put(("tech_stats_df", desc_full))
            except Exception: 
                desc_full = None

            n = len(prices)
            train_indices = None
            test_indices  = None
            if holdout_on:
                split = split_index(n, 0.7)
                train_indices = (0, split)
                test_indices  = (split, n)
                self.msg_q.put((MSG_INFO, f"Train {feats.index[0].date()} ~ {feats.index[split-1].date()} / "
                                        f"Test {feats.index[split].date()} ~ {feats.index[-1].date()}"))
            else:
                self.msg_q.put((MSG_INFO, "Holdout disabled: skipping Train/Test; using WFCV as primary evaluation."))


            # Few-shot build source (future ML: train split / now: full history; WFCV에서는 fold-train으로 재구성 가능)
            if holdout_on and train_indices is not None:
                train_feats = feats.iloc[train_indices[0]:train_indices[1]].copy()
            else:
                train_feats = feats.copy()

            # GUI log wrapper — filter noisy rebalance spam
            def gui_log_filtered(m: str):
                if isinstance(m, str) and (m.startswith("Rebalanced on ") and not self._allow_rebalance_spam):
                    return
                self.msg_q.put((MSG_INFO, m))

            base_cfg = {
                "rebalance_days": reb, "tcost": tcost, "max_weight": maxw, "turnover_cap": turn,
                "ollama_url": os.getenv("OLLAMA_URL","http://localhost:11434"), "model_name": model,
                "use_ollama": use_llm, "log_fn": gui_log_filtered,
                "prompt_profile": int(self.var_prompt_level.get() or 1),
                "log_level": self.var_log_level.get(),
                "gui_log_level": self.var_gui_log_level.get(),
                "log_every": int(self.var_log_every.get() or 0),
            }

            # --- Build strategies via registry ---
            all_strats = build_strategies(
                selected_bench, tickers, prices, feats, base_cfg,
                fewshot_or_train_feats=train_feats
            )

            # Few-shot text preview (from any NLPI strategy)
            try:
                any_strat = next(v for v in all_strats.values() if hasattr(v, "fewshot_block"))
                fs_text = getattr(any_strat, "fewshot_block", "") or ""
                self.msg_q.put(("fewshot_text", fs_text))
            except StopIteration:
                pass

            # Progress callback (stride)
            def make_progress_cb(phase_name: str, strat_name: str):
                def _cb(i: int, n_total: int, dt):
                    if reb <= 0 or (i % reb) != 0: return
                    step_idx = i // reb
                    if (step_idx % max(1, int(stride))) != 0: return
                    pct = int((i + 1) * 100 / max(n_total, 1))
                    ts = dt.date() if hasattr(dt, "date") else dt
                    self.msg_q.put((MSG_INFO, f"[{phase_name}] {strat_name}: {pct}% at {ts}"))
                return _cb

            payload = {"metrics": {}, "desc": desc_full}
            all_fold_strats = []

            if holdout_on and test_indices is not None:
                # Train
                self.msg_q.put((MSG_INFO, "Backtesting (Train)…"))
                res_train: Dict[str, pd.Series] = {}
                for name, strat in all_strats.items():
                    if self.cancel_event.is_set(): return self.msg_q.put((MSG_CANCELED, None))
                    ret, w = run_backtest(
                        prices, tickers, feats, strat,
                        train_indices[0], train_indices[1],
                        reb,
                        cancel_event=self.cancel_event
                    )
                    res_train[name] = ret
                payload["train"] = res_train

                # Test
                self.msg_q.put((MSG_INFO, "Backtesting (Test)…"))
                res_test: Dict[str, pd.Series] = {}
                metrics: Dict[str, dict] = {}
                for name, strat in all_strats.items():
                    if self.cancel_event.is_set(): return self.msg_q.put((MSG_CANCELED, None))
                    ret, w = run_backtest(
                        prices, tickers, feats, strat,
                        test_indices[0], test_indices[1],
                        reb,
                        cancel_event=self.cancel_event
                    )
                    res_test[name] = ret
                    metrics[name]  = _make_metrics_row(name, summary(ret, w))
                payload["test"] = res_test
                payload["metrics"] = metrics
            else:
                # Holdout off: metrics는 WFCV의 stitched(ALL) 기준으로 만들거나, full-sample을 참고용으로 한 번만 돌릴 수도 있음
                payload["metrics"] = {}  # WFCV 끝나고 채우도록 아래에서 처리

            # WFCV (optional)
            if wfcv_on:
                # ----------------------------
                # Adaptive WFCV windows (FIX)
                # ----------------------------
                min_test = max(21, reb)                 # 최소 OOS 길이(거래일): 1개월(21) or reb 중 큰 값
                min_train_floor = max(126, reb * 3)     # 최소 Train 하한: 6개월(126) or 3*reb
                min_train = min(252, int(n * 0.70))     # 1년(252) 강제 대신, 데이터의 70% 이내로 상한
                min_train = max(min_train_floor, min_train)

                # OOS가 최소 1폴드라도 만들어질 수 있는지 체크
                if n <= (min_train + min_test):
                    # 마지막 수단: 데이터가 1년 미만이면 min_train을 더 완화(하지만 reb*3은 유지)
                    relaxed_train = max(min_train_floor, int(n * 0.50))
                    if n <= (relaxed_train + min_test):
                        return self.msg_q.put((MSG_ERROR,
                            f"WFCV cannot run: n={n} is too small for min_train={min_train} (relaxed={relaxed_train}) "
                            f"and min_test={min_test} with reb={reb}. Extend date range or reduce rebalance_days."
                        ))
                    min_train = relaxed_train

                # k를 고정(5)하지 말고, 가능한 OOS 길이에 따라 자동 설정
                oos_start = min_train
                oos_n = n - oos_start
                max_k = max(2, min(5, oos_n // min_test))  # 폴드당 최소 min_test는 확보
                k = max_k

                # 폴드 step
                step = max(min_test, oos_n // k)

                # 각 fold의 (train_end, test_end) 생성
                fold_bounds = []
                for i in range(k):
                    te_start = oos_start + i * step
                    te_end = min(oos_start + (i + 1) * step, n)
                    if te_end - te_start < min_test:
                        continue
                    tr_start = 0
                    tr_end = te_start
                    # train도 최소 길이 보장
                    if (tr_end - tr_start) < min_train:
                        continue
                    fold_bounds.append((tr_start, tr_end, te_start, te_end))

                if not fold_bounds:
                    return self.msg_q.put((MSG_ERROR,
                        "WFCV produced 0 folds after applying adaptive window rules. "
                        "Extend date range or reduce rebalance_days."
                    ))


                # OOS 구간을 k개로 분할 (min_train 이후 구간만 대상으로)
                oos_start = min_train
                oos_n = n - oos_start
                step = max(max(21, reb), oos_n // k)

                # 각 fold의 (train_end, test_end) 생성
                fold_bounds = []
                for i in range(k):
                    te_start = oos_start + i * step
                    te_end = min(oos_start + (i + 1) * step, n)
                    if te_end - te_start < max(21, reb):
                        continue
                    tr_start = 0
                    tr_end = te_start
                    fold_bounds.append((tr_start, tr_end, te_start, te_end))

                if not fold_bounds:
                    return self.msg_q.put((MSG_ERROR,
                        "WFCV produced 0 folds after applying minimum window rules. "
                        "Extend date range or reduce rebalance_days / k."
                    ))

                # 이후 for 루프는 fold_bounds 기반으로 변경

                overlay: Dict[str, List[Tuple[pd.DatetimeIndex, np.ndarray]]] = {nm: [] for nm in all_strats.keys()}
                stitched: Dict[str, Tuple[pd.DatetimeIndex, np.ndarray]] = {}
                table_rows = []
                eq_concat: Dict[str, List[pd.Series]] = {nm: [] for nm in all_strats.keys()}
                to_by_strat: Dict[str, List[float]] = {nm: [] for nm in all_strats.keys()}
                stitched_ret_map: Dict[str, pd.Series] = {}

                for fold_id, (tr_start, tr_end, te_start, te_end) in enumerate(fold_bounds, start=1):
                    if self.cancel_event.is_set():
                        return self.msg_q.put((MSG_CANCELED, None))

                    self.msg_q.put((MSG_INFO,
                        f"[WFCV] Fold {fold_id}: train {prices.index[tr_start].date()}~{prices.index[tr_end-1].date()} / "
                        f"test {prices.index[te_start].date()}~{prices.index[te_end-1].date()}"
                    ))

                    # (권장) fold별 train 구간으로 few-shot/컨텍스트를 재구성
                    fold_train_feats = feats.iloc[tr_start:tr_end].copy()

                    fold_strats = build_strategies(
                        selected_bench, tickers, prices, feats, base_cfg,
                        fewshot_or_train_feats=fold_train_feats
                    )
                    all_fold_strats.append(fold_strats)

                    for name, strat in fold_strats.items():
                        if self.cancel_event.is_set():
                            return self.msg_q.put((MSG_CANCELED, None))

                        ret, w = run_backtest(
                            prices, tickers, feats, strat,
                            te_start, te_end, reb,
                            cancel_event=self.cancel_event
                        )

                        eq = (1 + ret).cumprod()
                        overlay[name].append((eq.index, eq.values))
                        eq_concat[name].append(ret)

                        sm = summary(ret, w)
                        to_val = sm.get("AvgTurnover", np.nan)
                        to_by_strat[name].append(to_val)

                        table_rows.append({
                            "Fold": str(fold_id),
                            "Strategy": name,
                            "Sharpe": sm.get("Sharpe", np.nan),
                            "Sortino": sm.get("Sortino", np.nan),
                            "CAGR": sm.get("CAGR", np.nan),
                            "Vol": sm.get("Vol", np.nan),
                            "MDD": sm.get("MDD", np.nan),
                            "TO": to_val,
                        })

                # stitched(ALL)
                for name in all_strats.keys():
                    if eq_concat[name]:
                        stitched_ret = pd.concat(eq_concat[name]).sort_index()
                        stitched_ret_map[name] = stitched_ret
                        stitched_eq = (1 + stitched_ret).cumprod()
                        stitched[name] = (stitched_eq.index, stitched_eq.values)

                        sm_all = summary(stitched_ret, None)
                        to_all = float(np.nanmean(to_by_strat[name])) if to_by_strat[name] else np.nan

                        table_rows.append({
                            "Fold": "ALL",
                            "Strategy": name,
                            "Sharpe": sm_all.get("Sharpe", np.nan),
                            "Sortino": sm_all.get("Sortino", np.nan),
                            "CAGR": sm_all.get("CAGR", np.nan),
                            "Vol": sm_all.get("Vol", np.nan),
                            "MDD": sm_all.get("MDD", np.nan),
                            "TO": to_all,
                        })

                payload["wfcv"] = {
                    "overlay": overlay,
                    "stitched": stitched,
                    "stitched_ret": stitched_ret_map,
                    "table": pd.DataFrame(table_rows)
                }
                
                # ------------------------------------------------------
                # Metrics tab population when Holdout is disabled
                # ------------------------------------------------------
                # Metrics 탭은 payload["metrics"]를 기반으로 그려집니다.
                # WFCV-only(holdout_off)일 때 payload["metrics"]가 비어있어
                # Metrics 탭이 공백이 되는 문제를 방지하기 위해,
                # stitched(전체 구간 연결) 성과를 '전체 요약'으로 채웁니다.
                if not holdout_on:
                    metrics = {}
                    for name, stitched_ret in stitched_ret_map.items():
                        sm_all = summary(stitched_ret, None)
                        # summary()에는 turnover가 없으므로 주입
                        sm_all = summary(stitched_ret, None)

                        # fold별 AvgTurnover를 모아둔 to_by_strat 기반으로 ALL의 TO를 계산
                        vals = to_by_strat.get(name, [])
                        to_all = float(np.nanmean(vals)) if vals else float("nan")
                        sm_all["AvgTurnover"] = to_all
                        metrics[name] = _make_metrics_row(name, sm_all)
                    payload["metrics"] = metrics



            # ==========================================================
            # Diagnostics aggregation (NEW)
            # ==========================================================
            try:
                diag_summary_rows = []
                diag_ts_rows = []

                # full-sample strategies
                if holdout_on:
                    for name, strat in all_strats.items():
                        if hasattr(strat, "diagnostics_summary"):
                            row = strat.diagnostics_summary()
                            if isinstance(row, dict):
                                row = dict(row)
                                row["strategy_key"] = name
                                row["fold"] = 0
                                diag_summary_rows.append(row)

                        if hasattr(strat, "diagnostics_timeseries"):
                            ts = strat.diagnostics_timeseries()
                            if isinstance(ts, pd.DataFrame) and not ts.empty:
                                ts = ts.copy()
                                ts["strategy_key"] = name
                                ts["fold"] = 0
                                diag_ts_rows.append(ts)

                # WFCV fold strategies (if exists)
                if wfcv_on and isinstance(all_fold_strats, list):
                    for fold_idx, fs in enumerate(all_fold_strats, start=1):
                        if not isinstance(fs, dict):
                            continue
                        for name, strat in fs.items():
                            # summary
                            if hasattr(strat, "diagnostics_summary"):
                                row = strat.diagnostics_summary()
                                if isinstance(row, dict):
                                    row = dict(row)
                                    row["strategy_key"] = name
                                    row["fold"] = fold_idx
                                    diag_summary_rows.append(row)

                            # timeseries
                            if hasattr(strat, "diagnostics_timeseries"):
                                ts = strat.diagnostics_timeseries()
                                if isinstance(ts, pd.DataFrame) and not ts.empty:
                                    ts = ts.copy()
                                    ts["strategy_key"] = name
                                    ts["fold"] = fold_idx
                                    diag_ts_rows.append(ts)

                payload["diag_summary"] = (
                    pd.DataFrame(diag_summary_rows)
                    if diag_summary_rows else pd.DataFrame()
                )

                payload["diag_timeseries"] = (
                    pd.concat(diag_ts_rows, ignore_index=True)
                    if diag_ts_rows else pd.DataFrame()
                )

            except Exception:
                payload["diag_summary"] = pd.DataFrame()
                payload["diag_timeseries"] = pd.DataFrame()

            self.msg_q.put((MSG_DONE, payload))

        except Exception as e:
            import traceback
            self.msg_q.put((MSG_ERROR, f"{e}\n{traceback.format_exc()}"))

    # -------------------- Pump --------------------

    def _pump(self):
        try:
            while True:
                msg, payload = self.msg_q.get_nowait()
                if msg == MSG_INFO:
                    if isinstance(payload, str) and payload.startswith("Rebalanced on ") and not self._allow_rebalance_spam:
                        continue
                    self.lbl_status.configure(text=str(payload)); self._log(str(payload))
                elif msg == MSG_ERROR:
                    self.pb.stop(); self.lbl_status.configure(text="Error")
                    self._log(f"[ERROR] {payload}")
                    if getattr(self, "_paper_running", False):
                        self._paper_log(f"[ERROR] Main canonical stage failed: {payload}")
                        self._paper_finish("failed", reliability_exit_code=None)
                    else:
                        messagebox.showerror("Error", str(payload))
                elif msg == MSG_DONE:
                    self.pb.stop(); self.lbl_status.configure(text="Done.")
                    self._last_res_test = payload.get("test", {}) or {}
                    # If Holdout Train/Test is off, fall back to WFCV concatenated OOS returns
                    if (not self._last_res_test) and isinstance(payload.get("wfcv"), dict):
                        stitched_ret = payload["wfcv"].get("stitched_ret", {})
                        if isinstance(stitched_ret, dict) and stitched_ret:
                            self._last_res_test = stitched_ret
                    self._last_metrics  = payload.get("metrics", {}) or {}
                    self._last_wfcv     = payload.get("wfcv", {}) or {}
                    self._build_oos_tidy_from_payload(payload)
                    self._refresh_oos_export_tab()
                    self._refresh_wfcv_model_report()
                    self._refresh_sig_combo()
                    self._render(payload); 
                    # ---- Diagnostics payload (NEW) ----
                    try:
                        dsum = payload.get("diag_summary", None)
                        dts  = payload.get("diag_timeseries", None)

                        self._diag_summary_df = dsum if isinstance(dsum, pd.DataFrame) else None
                        self._diag_ts_df = dts if isinstance(dts, pd.DataFrame) else None

                        if hasattr(self, "txt_diag_summary"):
                            self._refresh_diagnostics_tab()
                    except Exception:
                        pass
                    self._last_payload = payload or {}
                    self._update_all_models_matrix()
                    self._refresh_model_report()
                    self._log("Run completed.")
                    if getattr(self, "_auto_export_after_run", False) and getattr(self, "_auto_export_outdir", None):
                        try:
                            outdir = self._export_gui_outputs_to_folder(self._auto_export_outdir, payload)
                            self._auto_export_after_run = False
                            self._auto_export_outdir = None
                            if getattr(self, "_paper_running", False) and getattr(self, "_paper_stage", "") == "main":
                                self._paper_after_main_export(outdir)
                            else:
                                messagebox.showinfo("Export complete", f"GUI results exported to:\n{outdir}")
                        except Exception as e:
                            self._log(f"[ERROR] Auto export failed: {e}")
                            if getattr(self, "_paper_running", False):
                                self._paper_log(f"[ERROR] Canonical export failed: {e}")
                                self._paper_finish("failed", reliability_exit_code=None)
                            else:
                                messagebox.showerror("Auto export failed", str(e))
                elif msg == MSG_CANCELED:
                    self.pb.stop(); self.lbl_status.configure(text="Canceled."); self._log("Run canceled.")
                    if getattr(self, "_paper_running", False):
                        self._paper_finish("stopped", reliability_exit_code=None)
                elif msg == "models":
                    self._ollama_models = list(payload or [])
                    self.cbo_model.configure(values=self._ollama_models)
                    if self._ollama_models and self.var_model_name.get() not in self._ollama_models:
                        self.var_model_name.set(self._ollama_models[0])
                    self._ensure_llm_variants()
                elif msg == "ollama_ok":
                    ok = bool(payload)
                    self.lbl_ollama.configure(text="Ollama: Connected" if ok else "Ollama: Disconnected",
                                              foreground="#0a7" if ok else "#c33")
                    self.var_use_llm.set(ok)
                elif msg == "tech_stats_df":
                    if isinstance(payload, pd.DataFrame):
                        self._last_tech_stats_df = payload.copy()
                        self._render_df_describe(payload)
                elif msg == "fewshot_text":
                    self._fewshot_text = str(payload or "")
                    self.txt_few.delete("1.0", "end")
                    if self._fewshot_text: self.txt_few.insert("1.0", self._fewshot_text)
        except queue.Empty:
            pass
        finally:
            self.after(100, self._pump)

    # -------------------- Rendering --------------------

    def _render(self, payload: dict):
        if isinstance(payload.get("desc"), pd.DataFrame): self._render_df_describe(payload["desc"])

        if 'train' in payload and isinstance(payload['train'], dict):
            self.ax_train.clear()
            for strat_name, ret in payload['train'].items():
                if isinstance(ret, pd.Series) and not ret.empty:
                    (1+ret).cumprod().plot(ax=self.ax_train, label=strat_name)
            self.ax_train.set_title("Train Period Equity Curves")
            self.ax_train.set_xlabel("Date"); self.ax_train.set_ylabel("Equity (Initial=1)")
            self._legend_outside(self.ax_train, self.fig_train)
            self.canvas_train.draw()

        if 'test' in payload and isinstance(payload['test'], dict):
            self.ax_test.clear()
            for strat_name, ret in payload['test'].items():
                if isinstance(ret, pd.Series) and not ret.empty:
                    (1+ret).cumprod().plot(ax=self.ax_test, label=strat_name)
            self.ax_test.set_title("Test Period Equity Curves")
            self.ax_test.set_xlabel("Date"); self.ax_test.set_ylabel("Equity (Initial=1)")
            self._legend_outside(self.ax_test, self.fig_test)
            self.canvas_test.draw()

        if isinstance(payload.get("wfcv"), dict):
            info = payload["wfcv"]
            overlay = info.get("overlay", {}) or {}
            stitched = info.get("stitched", {}) or {}
            self.ax_wfcv.clear()
            for name, segs in overlay.items():
                for (idx, eq) in segs:
                    self.ax_wfcv.plot(idx, eq, linewidth=1, alpha=0.3)
            for name, (idx, eq) in stitched.items():
                self.ax_wfcv.plot(idx, eq, linewidth=2.4, label=name)
            self.ax_wfcv.set_title("WFCV: Fold Out-of-Sample (thin) + Stitched Out-of-Sample (bold)")
            self.ax_wfcv.set_xlabel("Date"); self.ax_wfcv.set_ylabel("Equity (Initial=1)")
            self._legend_outside(self.ax_wfcv, self.fig_wfcv)
            self.canvas_wfcv.draw()

            self.tree_wfcv.delete(*self.tree_wfcv.get_children())
            df = info.get("table")
            if isinstance(df, pd.DataFrame) and not df.empty:
                for _, row in df.iterrows():
                    vals = [row.get("Fold",""), row.get("Strategy",""),
                            f"{row.get('Sharpe',np.nan):.4f}" if pd.notna(row.get('Sharpe')) else "",
                            f"{row.get('CAGR',np.nan):.4f}" if pd.notna(row.get('CAGR')) else "",
                            f"{row.get('MDD',np.nan):.4f}" if pd.notna(row.get('MDD')) else ""]
                    self.tree_wfcv.insert("", "end", values=vals)
            self._last_wfcv_info = info
            self._refresh_wfcv_model_report()

        self.tree.delete(*self.tree.get_children())
        if 'metrics' in payload and isinstance(payload['metrics'], dict):
            # payload['metrics'] = {strategy_key: metrics_row_dict}
            # -> Tree 첫 컬럼(name)에 key를 채워 넣어야 화면에 정상 표시됨

            def _fmt(v):
                """Robust formatter for GUI cells (handles None/NaN)."""
                import math
                if v is None:
                    return ""
                if isinstance(v, float) and math.isnan(v):
                    return ""
                if isinstance(v, int):
                    return str(v)
                if isinstance(v, float):
                    return f"{v:.4f}"
                return str(v)

            rows2 = [dict(v, name=k) if isinstance(v, dict) else {"name": k}
                    for k, v in payload["metrics"].items()]
            rows2 = sorted(rows2, key=lambda r: r.get("Sharpe", -np.inf), reverse=True)

            for row in rows2:
                vals = []
                for k, _, _w in COL_DEF:     # ✅ 3번째는 default가 아니라 width로 취급
                    vals.append(_fmt(row.get(k)))
                self.tree.insert("", "end", values=vals)

                
        self._last_metrics = payload.get("metrics", {}) or {}
        self._update_all_models_matrix()
        self._refresh_model_report()

        self._refresh_sig_combo()
        self._recompute_significance(auto=True)

    def _render_df_describe(self, desc: pd.DataFrame):
        for item in self.tree_stats.get_children(): self.tree_stats.delete(item)
        cols = list(desc.columns); self.tree_stats["columns"] = cols; self.tree_stats["show"] = "headings"
        for c in cols:
            self.tree_stats.heading(c, text=c)
            self.tree_stats.column(c, width=140 if c == "Ticker" else 110,
                                   anchor="w" if c == "Ticker" else "e", stretch=True)
        for _, row in desc.iterrows():
            vals = []
            for c in cols:
                v = row[c]
                if c == "Ticker": vals.append(str(v))
                elif pd.isna(v):  vals.append("")
                else:             vals.append(f"{float(v):.6f}")
            self.tree_stats.insert("", "end", values=vals)
        self.tree_stats._header_labels = cols

    # -------------------- Significance helpers --------------------

    def _refresh_sig_combo(self):
        keys = list(self._last_res_test.keys())
        if not keys: return
        self.cbo_sig.configure(values=keys)
        if self.var_sig_base.get() not in keys:
            self.var_sig_base.set(keys[0])

    def _recompute_significance(self, auto: bool=False):
        if build_significance_table is None: 
            return
        if not self._last_res_test: 
            return
        baseline = self.var_sig_base.get().strip()
        if not baseline or baseline not in self._last_res_test:
            return
        try:
            df = build_significance_table(
                group="Multi", algo="DAA",
                returns_dict=self._last_res_test, baseline=baseline,
                comparators=None, ann_factor=252, hac_lags=5,
                mbb_block=7, mbb_B=1000, mbb_seed=7
            )
        except Exception as e:
            self._log(f"Significance build failed: {e}")
            return

        self.tree_sig.delete(*self.tree_sig.get_children())
        cols = list(self.tree_sig["columns"])
        for _, row in df.iterrows():
            vals = []
            for c in cols:
                v = row.get(c, "")
                if c in ("N",): vals.append(str(int(v)) if pd.notna(v) else "")
                elif c in ("Mean Diff (Ann.)","t_HAC","p_HAC","Wilcoxon p","JK z","MBB p_two"):
                    vals.append("" if pd.isna(v) else f"{float(v):.4g}")
                else:
                    vals.append("" if pd.isna(v) else str(v))
            self.tree_sig.insert("", "end", values=vals)

    # -------------------- Log helpers --------------------

    def _log(self, msg: str):
        try: self.txt_log.insert("end", msg + "\n"); self.txt_log.see("end")
        except Exception: pass

    def _log_copy(self, all_text: bool = False):
        try:
            text = self.txt_log.get("1.0", "end-1c") if all_text else self.txt_log.selection_get()
            if text: self.clipboard_clear(); self.clipboard_append(text)
        except Exception: pass

    def _log_select_all(self):
        try: self.txt_log.tag_add("sel", "1.0", "end"); self.txt_log.see("end")
        except Exception: pass

# --------------------------- helpers ---------------------------

def _make_metrics_row(name: str, stats: dict) -> dict:
    """
    summary() 결과를 GUI 테이블 컬럼에 맞게 매핑하는 수정된 함수
    """
    d = {"name": name}
    d["Sharpe"]  = stats.get("Sharpe", np.nan)
    d["Sortino"] = stats.get("Sortino", np.nan)
    d["Vol"]     = stats.get("Vol", np.nan)
    d["CAGR"]    = stats.get("CAGR", np.nan)
    d["MDD"]     = stats.get("MDD", np.nan)

    terminal_val = stats.get("Terminal", 1.0)
    d["CUM"] = terminal_val - 1.0 if terminal_val is not None else np.nan
    d["ANN"] = stats.get("CAGR", np.nan)
    d["TO"]  = stats.get("AvgTurnover", np.nan)

    for key, value in d.items():
        if key != "name":
            try:
                d[key] = float(value)
            except (ValueError, TypeError):
                d[key] = np.nan
    return d

# ------------------------------ main ------------------------------

def main():
    App().mainloop()

if __name__ == "__main__":
    main()
