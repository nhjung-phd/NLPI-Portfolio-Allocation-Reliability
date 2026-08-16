#!/usr/bin/env python3
"""Live monitor for resumable NLPI reliability experiments.

Runs continuously by default and exits cleanly with Ctrl+C.  Uses only the
Python standard library so it can run outside the project virtual environment.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


DEFAULT_OUTDIR = Path(
    "outputs/NLPI_PAPER_CANONICAL_V1_20260717_152949/"
    "reliability_resumable_20260727"
)
DEFAULT_LOG = Path("reliability_resumable_20260727.log")
CALL_RE = re.compile(r"\[CALL (?:START|DONE|FAILED)\]\s+(\d+)/(\d+)")
MODEL_RE = re.compile(r"\[MODEL (START|DONE)\]\s+(\S+)")
RUN_RE = re.compile(r"\[RUN\]\s+(\S+)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Continuously monitor NLPI reliability experiments (Ctrl+C exits)"
    )
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    p.add_argument("--db", type=Path, default=None)
    p.add_argument("--log", type=Path, default=DEFAULT_LOG)
    p.add_argument("--interval", type=float, default=10.0, help="Refresh seconds")
    p.add_argument(
        "--stale-minutes", type=float, default=25.0,
        help="Warn when no checkpoint completion occurs for this many minutes",
    )
    p.add_argument("--once", action="store_true", help="Print once and exit")
    p.add_argument("--no-clear", action="store_true", help="Do not clear between refreshes")
    p.add_argument("--recent", type=int, default=8, help="Recent log events to show")
    return p.parse_args()


def run_command(cmd: list[str], timeout: float = 3.0) -> tuple[int, str]:
    try:
        cp = subprocess.run(
            cmd, text=True, capture_output=True, timeout=timeout, check=False
        )
        text = (cp.stdout or "").strip()
        if cp.stderr and not text:
            text = cp.stderr.strip()
        return cp.returncode, text
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 127, str(exc)


def fmt_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "-"
    seconds = int(seconds)
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if days:
        return f"{days}d {hours:02d}h {minutes:02d}m"
    if hours:
        return f"{hours:02d}h {minutes:02d}m"
    return f"{minutes:02d}m {seconds:02d}s"


def parse_utc(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(
            timezone.utc
        )
    except ValueError:
        return None


def tail_lines(path: Path, count: int = 500) -> list[str]:
    if not path.exists():
        return []
    # Logs can become large. Seek near the end rather than reading the whole file.
    with path.open("rb") as fh:
        fh.seek(0, os.SEEK_END)
        size = fh.tell()
        fh.seek(max(0, size - 256_000))
        data = fh.read().decode("utf-8", errors="replace")
    return data.splitlines()[-count:]


def parse_log(lines: Iterable[str]) -> dict:
    current_model = "-"
    current_experiment = "-"
    ordinal = None
    plan_total = None
    events: list[str] = []
    warnings: list[str] = []
    for line in lines:
        model = MODEL_RE.search(line)
        if model:
            current_model = model.group(2)
            if model.group(1) == "DONE":
                current_model = f"{current_model} (done)"
        run = RUN_RE.search(line)
        if run:
            current_experiment = run.group(1)
        call = CALL_RE.search(line)
        if call:
            ordinal, plan_total = int(call.group(1)), int(call.group(2))
        if any(
            token in line
            for token in (
                "[CALL START]", "[CALL DONE]", "[CALL FAILED]",
                "[WARN]", "[MODEL START]", "[MODEL DONE]", "[ALL DONE]",
            )
        ):
            events.append(line)
        if any(token in line.lower() for token in ("timeout", "failed", "error")):
            warnings.append(line)
    return {
        "model": current_model,
        "experiment": current_experiment,
        "ordinal": ordinal,
        "plan_total": plan_total,
        "events": events,
        "warnings": warnings,
    }


def read_db(db: Path) -> dict:
    empty = {
        "recorded": 0, "completed": 0, "failed": 0, "timed_out": 0,
        "running": 0, "groups": [], "current": None, "last_done": None,
        "avg_sec": None, "recent_avg_sec": None, "error": None,
    }
    if not db.exists():
        empty["error"] = f"Checkpoint DB not found: {db}"
        return empty
    try:
        conn = sqlite3.connect(f"file:{db.resolve()}?mode=ro", uri=True, timeout=2)
        conn.execute("PRAGMA query_only=ON")
        statuses = dict(
            conn.execute("SELECT status, COUNT(*) FROM calls GROUP BY status")
        )
        groups = conn.execute(
            """
            SELECT experiment_id, model_id,
                   SUM(status='completed'),
                   SUM(status IN ('failed','timed_out')),
                   SUM(status='running'),
                   COUNT(*)
            FROM calls
            GROUP BY experiment_id, model_id
            ORDER BY MIN(started_at), experiment_id, model_id
            """
        ).fetchall()
        current = conn.execute(
            """
            SELECT experiment_id, model_id, condition_id, decision_date,
                   started_at, attempt
            FROM calls WHERE status='running'
            ORDER BY started_at DESC LIMIT 1
            """
        ).fetchone()
        last_done = conn.execute(
            """
            SELECT MAX(completed_at) FROM calls
            WHERE status IN ('completed','failed','timed_out')
            """
        ).fetchone()[0]
        avg_sec = conn.execute(
            "SELECT AVG(elapsed_sec) FROM calls WHERE status='completed'"
        ).fetchone()[0]
        recent_avg = conn.execute(
            """
            SELECT AVG(elapsed_sec) FROM (
              SELECT elapsed_sec FROM calls
              WHERE status='completed' AND elapsed_sec IS NOT NULL
              ORDER BY completed_at DESC LIMIT 30
            )
            """
        ).fetchone()[0]
        conn.close()
        return {
            "recorded": sum(statuses.values()),
            "completed": statuses.get("completed", 0),
            "failed": statuses.get("failed", 0),
            "timed_out": statuses.get("timed_out", 0),
            "running": statuses.get("running", 0),
            "groups": groups,
            "current": current,
            "last_done": last_done,
            "avg_sec": avg_sec,
            "recent_avg_sec": recent_avg,
            "error": None,
        }
    except sqlite3.Error as exc:
        empty["error"] = f"Cannot read checkpoint DB: {exc}"
        return empty


def runner_status() -> tuple[bool, list[str]]:
    rc, output = run_command(["pgrep", "-fl", "q1_experiments.runner"])
    lines = (
        [
            line
            for line in output.splitlines()
            if line.strip()
            and "pgrep" not in line
            and "q1_checkpoint_status.py" not in line
        ]
        if rc == 0
        else []
    )
    return bool(lines), lines


def ollama_status() -> tuple[bool, str]:
    rc, output = run_command(["ollama", "ps"], timeout=5)
    if rc == 0:
        return True, output or "(no model currently loaded)"
    return False, output or "ollama command unavailable"


def render(args: argparse.Namespace) -> tuple[str, bool]:
    db = args.db or args.outdir / "logs" / "q1_call_checkpoint.sqlite3"
    db_state = read_db(db)
    log_state = parse_log(tail_lines(args.log))
    runner_alive, runners = runner_status()
    ollama_ok, ollama_text = ollama_status()
    now = datetime.now(timezone.utc)
    last_done = parse_utc(db_state["last_done"])
    stale_sec = (now - last_done).total_seconds() if last_done else None

    ordinal = log_state["ordinal"]
    plan_total = log_state["plan_total"]
    recent_avg = db_state["recent_avg_sec"] or db_state["avg_sec"]
    eta = None
    if ordinal is not None and plan_total and recent_avg:
        eta = max(0, plan_total - ordinal) * recent_avg

    alerts: list[str] = []
    if not runner_alive:
        alerts.append("Runner process is not running.")
    if db_state["error"]:
        alerts.append(db_state["error"])
    if stale_sec is not None and runner_alive and stale_sec >= args.stale_minutes * 60:
        alerts.append(
            f"No completed/failed checkpoint for {fmt_duration(stale_sec)} "
            f"(threshold {args.stale_minutes:g}m)."
        )
    if db_state["failed"] or db_state["timed_out"]:
        alerts.append(
            f"Recorded failures: {db_state['failed']} failed, "
            f"{db_state['timed_out']} timed out."
        )
    if db_state["running"] > 1:
        alerts.append(f"Multiple running checkpoint rows: {db_state['running']}.")
    if not ollama_ok:
        alerts.append("Could not query Ollama.")

    width = min(max(shutil.get_terminal_size((100, 30)).columns, 80), 140)
    out: list[str] = []
    out.append("=" * width)
    out.append(
        "NLPI RELIABILITY LIVE MONITOR".ljust(width - 25)
        + datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    )
    out.append("=" * width)
    out.append(
        f"Runner: {'RUNNING' if runner_alive else 'STOPPED':8} | "
        f"Model: {log_state['model']} | Experiment: {log_state['experiment']}"
    )
    if ordinal is not None and plan_total:
        pct = 100.0 * ordinal / plan_total
        out.append(
            f"Current module: {ordinal:,}/{plan_total:,} ({pct:5.1f}%) | "
            f"Recent avg/call: {fmt_duration(recent_avg)} | ETA: {fmt_duration(eta)}"
        )
    out.append(
        f"Checkpoint: completed={db_state['completed']:,} | "
        f"failed={db_state['failed']:,} | timed_out={db_state['timed_out']:,} | "
        f"running={db_state['running']:,} | recorded={db_state['recorded']:,}"
    )
    out.append(
        f"Last checkpoint: "
        f"{fmt_duration(stale_sec) + ' ago' if stale_sec is not None else '-'} | "
        f"All-time avg/call: {fmt_duration(db_state['avg_sec'])}"
    )

    current = db_state["current"]
    if current:
        started = parse_utc(current[4])
        elapsed = (now - started).total_seconds() if started else None
        out.append(
            f"Active call: {current[1]} | {current[0]} | {current[2]} | "
            f"{current[3]} | attempt={current[5]} | elapsed={fmt_duration(elapsed)}"
        )

    out.append("")
    out.append("CHECKPOINT BY EXPERIMENT / MODEL")
    if db_state["groups"]:
        out.append(
            f"{'EXPERIMENT':29} {'MODEL':16} {'DONE':>7} {'FAIL':>7} "
            f"{'RUN':>5} {'TOTAL':>7}"
        )
        out.append("-" * min(width, 78))
        for experiment, model, done, fail, running, total in db_state["groups"]:
            out.append(
                f"{experiment[:29]:29} {model[:16]:16} {done:7d} "
                f"{fail:7d} {running:5d} {total:7d}"
            )
    else:
        out.append("(no checkpoint rows yet)")

    out.append("")
    out.append("OLLAMA")
    out.extend(f"  {line}" for line in ollama_text.splitlines()[:8])
    if runners:
        out.append("")
        out.append("RUNNER PROCESS")
        out.extend(f"  {line}" for line in runners[:4])

    out.append("")
    out.append("HEALTH")
    if alerts:
        out.extend(f"  [!] {alert}" for alert in alerts)
    else:
        out.append("  [OK] Runner, checkpoint writes, and Ollama look healthy.")

    recent_events = log_state["events"][-max(0, args.recent):]
    if recent_events:
        out.append("")
        out.append("RECENT EVENTS")
        out.extend(f"  {line[:width-2]}" for line in recent_events)

    out.append("")
    out.append(
        f"Auto-refresh: {args.interval:g}s | Ctrl+C: stop monitor only "
        "(experiment keeps running)"
    )
    return "\n".join(out), bool(alerts)


def main() -> None:
    args = parse_args()
    if args.interval <= 0:
        raise SystemExit("--interval must be greater than zero")
    try:
        while True:
            screen, _ = render(args)
            if not args.no_clear and not args.once and sys.stdout.isatty():
                print("\033[2J\033[H", end="")
            print(screen, flush=True)
            if args.once:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nMonitor stopped. The experiment runner was not terminated.")


if __name__ == "__main__":
    main()
