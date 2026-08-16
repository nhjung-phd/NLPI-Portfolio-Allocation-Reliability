"""Crash-safe task ledger and atomic per-task result storage."""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class TaskLedger:
    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.result_dir = self.run_dir / "checkpoints" / "completed"
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / "task_ledger.sqlite"
        self.db = sqlite3.connect(self.path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("""CREATE TABLE IF NOT EXISTS tasks (
            task_id TEXT PRIMARY KEY, experiment TEXT NOT NULL, payload TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending', attempts INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL, started_at TEXT, finished_at TEXT,
            result_path TEXT, error TEXT)""")
        self.db.commit()

    def recover_interrupted(self) -> int:
        cur = self.db.execute(
            "UPDATE tasks SET status='pending', started_at=NULL, error='recovered after interruption' WHERE status='running'"
        )
        self.db.commit()
        return cur.rowcount

    def add(self, task_id: str, experiment: str, payload: dict) -> None:
        self.db.execute(
            "INSERT OR IGNORE INTO tasks(task_id,experiment,payload,status,created_at) VALUES(?,?,?,?,?)",
            (task_id, experiment, json.dumps(payload, ensure_ascii=False, sort_keys=True), "pending", utcnow()),
        )

    def commit(self) -> None:
        self.db.commit()

    def pending(self, retry_failed: bool = False):
        states = ("pending", "failed") if retry_failed else ("pending",)
        marks = ",".join("?" for _ in states)
        for task_id, experiment, payload in self.db.execute(
            f"SELECT task_id,experiment,payload FROM tasks WHERE status IN ({marks}) ORDER BY rowid", states
        ):
            yield task_id, experiment, json.loads(payload)

    def start(self, task_id: str) -> None:
        self.db.execute(
            "UPDATE tasks SET status='running', attempts=attempts+1, started_at=?, finished_at=NULL, error=NULL WHERE task_id=?",
            (utcnow(), task_id),
        )
        self.db.commit()

    def complete(self, task_id: str, result: dict) -> None:
        target = self.result_dir / f"{task_id}.json"
        tmp = target.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(result, fh, ensure_ascii=False, allow_nan=False)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, target)
        self.db.execute(
            "UPDATE tasks SET status='completed',finished_at=?,result_path=?,error=NULL WHERE task_id=?",
            (utcnow(), str(target.relative_to(self.run_dir)), task_id),
        )
        self.db.commit()

    def fail(self, task_id: str, error: str) -> None:
        self.db.execute(
            "UPDATE tasks SET status='failed',finished_at=?,error=? WHERE task_id=?",
            (utcnow(), error[-4000:], task_id),
        )
        self.db.commit()

    def results(self):
        for (rel,) in self.db.execute(
            "SELECT result_path FROM tasks WHERE status='completed' AND result_path IS NOT NULL ORDER BY rowid"
        ):
            path = self.run_dir / rel
            if path.exists():
                yield json.loads(path.read_text(encoding="utf-8"))

    def counts(self) -> dict:
        out = {"pending": 0, "running": 0, "completed": 0, "failed": 0}
        out.update(dict(self.db.execute("SELECT status,COUNT(*) FROM tasks GROUP BY status")))
        return out

    def close(self):
        self.db.close()
