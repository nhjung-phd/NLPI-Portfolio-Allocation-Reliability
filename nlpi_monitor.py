#!/usr/bin/env python3
"""Terminal dashboard for NLPI reliability experiments on macOS.

Uses only the Python standard library. Quit with Ctrl-C or q.
"""

from __future__ import annotations

import argparse
import curses
import datetime as dt
import os
from pathlib import Path
import subprocess
import time


DEFAULT_PROJECT = Path(
    "/Users/nhjung/Data/git/LLM_Portfolios/"
    "LLM_Portfolio_NLPI_PaperCanonical"
)
DEFAULT_LOG = DEFAULT_PROJECT / "reliability_prompt_robustness_20260723.log"
DEFAULT_OUTPUT = (
    DEFAULT_PROJECT
    / "outputs/NLPI_PAPER_CANONICAL_V1_20260717_152949/"
    "reliability_retry_20260723/prompt_robustness"
)


def run(command: list[str], timeout: float = 3.0) -> str:
    try:
        result = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
        return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return f"[unavailable: {exc}]"


def runner_pids() -> list[str]:
    output = run(["pgrep", "-f", "q1_experiments.runner"])
    return [line.strip() for line in output.splitlines() if line.strip().isdigit()]


def parse_etime(value: str) -> float | None:
    """Convert ps etime ([[dd-]hh:]mm:ss) to seconds."""
    try:
        day_split = value.strip().split("-", 1)
        days = int(day_split[0]) if len(day_split) == 2 else 0
        clock = day_split[-1].split(":")
        if len(clock) == 3:
            hours, minutes, seconds = map(int, clock)
        elif len(clock) == 2:
            hours = 0
            minutes, seconds = map(int, clock)
        else:
            return None
        return float(days * 86400 + hours * 3600 + minutes * 60 + seconds)
    except (TypeError, ValueError):
        return None


def runner_elapsed(pids: list[str]) -> float | None:
    """Return elapsed seconds for a single runner; avoid guessing for duplicates."""
    if len(pids) != 1:
        return None
    output = run(["ps", "-p", pids[0], "-o", "etime="])
    return parse_etime(output)


def duration_text(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes = remainder // 60
    if hours:
        return f"{hours}h {minutes:02d}m"
    return f"{minutes}m"


def estimate_rows(
    elapsed: float | None,
    total_min_hours: float,
    total_max_hours: float,
) -> list[str]:
    if elapsed is None:
        return ["Unavailable: a single runner process is required."]

    lower = total_min_hours * 3600
    upper = total_max_hours * 3600
    now = dt.datetime.now()
    started = now - dt.timedelta(seconds=elapsed)
    earliest = started + dt.timedelta(seconds=lower)
    latest = started + dt.timedelta(seconds=upper)
    remaining_low = max(0.0, lower - elapsed)
    remaining_high = max(0.0, upper - elapsed)

    rows = [
        f"Elapsed: {duration_text(elapsed)}",
        f"Estimated total: {duration_text(lower)}–{duration_text(upper)} "
        "(planning range; not measured progress)",
    ]
    if elapsed < lower:
        rows.append(
            f"Estimated remaining: {duration_text(remaining_low)}–"
            f"{duration_text(remaining_high)}"
        )
        rows.append(
            "Estimated completion: "
            f"{earliest:%Y-%m-%d %H:%M}–{latest:%Y-%m-%d %H:%M}"
        )
    elif elapsed <= upper:
        rows.append(f"Estimated remaining: 0m–{duration_text(remaining_high)}")
        rows.append(f"Estimated completion: now–{latest:%Y-%m-%d %H:%M}")
    else:
        rows.append(
            f"OVER ESTIMATE by {duration_text(elapsed - upper)}; "
            "check activity rather than treating this as progress."
        )
    return rows


def process_rows(pids: list[str]) -> list[str]:
    if not pids:
        return []
    output = run(
        [
            "ps",
            "-p",
            ",".join(pids),
            "-o",
            "pid=,etime=,state=,%cpu=,%mem=,command=",
        ]
    )
    rows = []
    for line in output.splitlines():
        parts = line.strip().split(None, 5)
        if len(parts) >= 5:
            command = parts[5] if len(parts) == 6 else ""
            rows.append(
                f"PID {parts[0]:>6}  elapsed {parts[1]:>11}  "
                f"state {parts[2]:<4} CPU {parts[3]:>5}%  "
                f"MEM {parts[4]:>5}%  {command}"
            )
    return rows


def ollama_rows() -> list[str]:
    output = run(["ollama", "ps"], timeout=5.0)
    if not output:
        return ["No model currently loaded."]
    return output.splitlines()


def server_rows() -> list[str]:
    output = run(
        [
            "ps",
            "ax",
            "-o",
            "pid=,etime=,%cpu=,%mem=,command=",
        ]
    )
    rows = []
    for line in output.splitlines():
        if "llama-server" in line and "grep" not in line:
            parts = line.strip().split(None, 4)
            if len(parts) == 5:
                rows.append(
                    f"PID {parts[0]:>6}  elapsed {parts[1]:>11}  "
                    f"CPU {parts[2]:>5}%  MEM {parts[3]:>5}%  {parts[4]}"
                )
    return rows or ["No llama-server process found."]


def file_age(path: Path) -> tuple[str, float | None]:
    try:
        stat = path.stat()
    except OSError:
        return "not found", None
    age = max(0.0, time.time() - stat.st_mtime)
    stamp = dt.datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    return f"{stamp} ({human_age(age)} ago, {stat.st_size:,} bytes)", age


def human_age(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m {seconds % 60}s"
    if seconds < 86400:
        return f"{seconds // 3600}h {(seconds % 3600) // 60}m"
    return f"{seconds // 86400}d {(seconds % 86400) // 3600}h"


def tail(path: Path, lines: int) -> list[str]:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - 64 * 1024))
            text = handle.read().decode("utf-8", errors="replace")
        return text.splitlines()[-lines:] or ["(log is empty)"]
    except OSError as exc:
        return [f"(cannot read log: {exc})"]


def latest_files(directory: Path, limit: int = 6) -> list[tuple[Path, float, int]]:
    if not directory.exists():
        return []
    found: list[tuple[Path, float, int]] = []
    try:
        for path in directory.rglob("*"):
            if path.is_file() and path.name != ".DS_Store":
                stat = path.stat()
                found.append((path, stat.st_mtime, stat.st_size))
    except OSError:
        pass
    return sorted(found, key=lambda item: item[1], reverse=True)[:limit]


def shorten(text: str, width: int) -> str:
    if width <= 1:
        return ""
    return text if len(text) <= width else text[: max(1, width - 1)] + "…"


def add_line(
    screen: curses.window,
    row: int,
    text: str,
    style: int = 0,
) -> int:
    height, width = screen.getmaxyx()
    if row < height - 1:
        try:
            screen.addstr(row, 0, shorten(text, width - 1), style)
        except curses.error:
            pass
    return row + 1


def section(screen: curses.window, row: int, title: str) -> int:
    return add_line(screen, row, f"─ {title} " + "─" * 60, curses.A_BOLD)


def calculate_health(
    pids: list[str],
    servers: list[str],
    log_age: float | None,
    output_age: float | None,
    stale_minutes: int,
) -> tuple[str, int]:
    stale = stale_minutes * 60
    if len(pids) > 1:
        return f"WARNING: {len(pids)} duplicate runners detected", 3
    if not pids:
        return "STOPPED: no q1_experiments.runner process", 2
    if not any("PID " in row for row in servers):
        return "WAITING: runner exists, but no active llama-server", 2
    most_recent = min(age for age in (log_age, output_age) if age is not None) \
        if any(age is not None for age in (log_age, output_age)) else None
    if most_recent is not None and most_recent > stale:
        return (
            f"CHECK: no log/output update for {human_age(most_recent)} "
            f"(threshold {stale_minutes}m)",
            2,
        )
    return "RUNNING: runner and Ollama inference process detected", 1


def dashboard(
    screen: curses.window,
    log_path: Path,
    output_dir: Path,
    interval: float,
    stale_minutes: int,
    estimate_min_hours: float,
    estimate_max_hours: float,
) -> None:
    try:
        curses.curs_set(0)
    except curses.error:
        pass
    screen.nodelay(True)
    screen.timeout(max(100, int(interval * 1000)))
    if curses.has_colors():
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_GREEN, -1)
        curses.init_pair(2, curses.COLOR_YELLOW, -1)
        curses.init_pair(3, curses.COLOR_RED, -1)

    paused = False
    while True:
        key = screen.getch()
        if key in (ord("q"), ord("Q"), 3):
            return
        if key in (ord("p"), ord("P"), ord(" ")):
            paused = not paused
        if paused:
            time.sleep(0.1)
            continue

        pids = runner_pids()
        elapsed = runner_elapsed(pids)
        processes = process_rows(pids)
        ollama = ollama_rows()
        servers = server_rows()
        log_status, log_age = file_age(log_path)
        output_status, output_age = file_age(output_dir)
        recent = latest_files(output_dir)
        if recent:
            output_age = max(0.0, time.time() - recent[0][1])
            output_status = (
                f"latest file {human_age(output_age)} ago; "
                f"{len(recent)} recent file(s) shown"
            )
        health, color = calculate_health(
            pids, servers, log_age, output_age, stale_minutes
        )

        screen.erase()
        row = 0
        now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row = add_line(
            screen,
            row,
            f" NLPI RELIABILITY MONITOR   {now}   refresh {interval:g}s",
            curses.A_REVERSE | curses.A_BOLD,
        )
        style = curses.color_pair(color) | curses.A_BOLD if curses.has_colors() else curses.A_BOLD
        row = add_line(screen, row, f" STATUS  {health}", style)
        row = add_line(
            screen,
            row,
            " Ctrl-C/q: quit   p/space: pause   "
            "(monitor exit does NOT stop the experiment)",
        )

        row = section(screen, row, "Time estimate")
        for item in estimate_rows(
            elapsed, estimate_min_hours, estimate_max_hours
        ):
            row = add_line(screen, row, item)

        row = section(screen, row, "Q1 runner")
        if processes:
            for item in processes:
                row = add_line(screen, row, item)
        else:
            row = add_line(screen, row, "No runner found.")

        row = section(screen, row, "Ollama active models")
        for item in ollama[:6]:
            row = add_line(screen, row, item)

        row = section(screen, row, "Ollama inference process")
        for item in servers[:3]:
            row = add_line(screen, row, item)

        row = section(screen, row, "Files")
        row = add_line(screen, row, f"Log:    {log_path}")
        row = add_line(screen, row, f"         {log_status}")
        row = add_line(screen, row, f"Output: {output_dir}")
        row = add_line(screen, row, f"         {output_status}")
        for path, modified, size in recent:
            try:
                relative = path.relative_to(output_dir)
            except ValueError:
                relative = path
            age = human_age(max(0.0, time.time() - modified))
            row = add_line(screen, row, f"  {relative}  {size:,} B  {age} ago")

        remaining = max(3, screen.getmaxyx()[0] - row - 2)
        row = section(screen, row, f"Log tail ({min(remaining, 12)} lines)")
        for item in tail(log_path, min(remaining, 12)):
            row = add_line(screen, row, item)

        screen.refresh()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Monitor an NLPI q1_experiments.runner job in a terminal UI."
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=DEFAULT_LOG,
        help=f"log file (default: {DEFAULT_LOG})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"experiment output directory (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="refresh interval in seconds (default: 2)",
    )
    parser.add_argument(
        "--stale-minutes",
        type=int,
        default=60,
        help="warn after this many minutes without file updates (default: 60)",
    )
    parser.add_argument(
        "--estimate-min-hours",
        type=float,
        default=8.0,
        help="optimistic total runtime estimate in hours (default: 8)",
    )
    parser.add_argument(
        "--estimate-max-hours",
        type=float,
        default=16.0,
        help="conservative total runtime estimate in hours (default: 16)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.interval < 0.2:
        raise SystemExit("--interval must be at least 0.2 seconds")
    if (
        args.estimate_min_hours <= 0
        or args.estimate_max_hours < args.estimate_min_hours
    ):
        raise SystemExit(
            "--estimate-max-hours must be >= --estimate-min-hours > 0"
        )
    try:
        curses.wrapper(
            dashboard,
            args.log.expanduser(),
            args.output_dir.expanduser(),
            args.interval,
            args.stale_minutes,
            args.estimate_min_hours,
            args.estimate_max_hours,
        )
    except KeyboardInterrupt:
        pass
    print("NLPI monitor closed. The experiment process was not stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
