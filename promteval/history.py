"""A local, free history of past Improve/Compare runs.

Uses Python's built-in `sqlite3` -- no server, no extra dependency, no account,
no cost. One file on disk (~/.promteval/history.db by default) so it works no
matter which directory `prompteval` gets launched from, and survives closing
the terminal or the browser tab.
"""

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_DB_PATH = Path.home() / ".promteval" / "history.db"


@dataclass
class HistoryEntry:
    id: int
    kind: str  # "improve" | "compare"
    created_at: str  # ISO 8601, UTC
    summary: str
    model: str
    request: dict[str, Any]
    response: dict[str, Any]


@contextmanager
def _connection(db_path: Path) -> Iterator[sqlite3.Connection]:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                kind TEXT NOT NULL CHECK (kind IN ('improve', 'compare')),
                created_at TEXT NOT NULL,
                summary TEXT NOT NULL,
                model TEXT NOT NULL,
                request_json TEXT NOT NULL,
                response_json TEXT NOT NULL
            )
            """
        )
        yield conn
        conn.commit()
    finally:
        conn.close()


def save_run(
    kind: str,
    summary: str,
    model: str,
    request: dict[str, Any],
    response: dict[str, Any],
    db_path: Path = DEFAULT_DB_PATH,
) -> int:
    """Records one completed run. Never raises on its own account -- callers
    decide whether a history-save failure should interrupt anything (it
    shouldn't: losing history is much better than losing a real result)."""
    with _connection(db_path) as conn:
        cursor = conn.execute(
            "INSERT INTO runs (kind, created_at, summary, model, request_json, response_json) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                kind,
                datetime.now(UTC).isoformat(),
                summary[:200],
                model,
                json.dumps(request),
                json.dumps(response),
            ),
        )
        return cursor.lastrowid


def list_runs(limit: int = 50, db_path: Path = DEFAULT_DB_PATH) -> list[HistoryEntry]:
    """Newest first, without the (potentially large) request/response bodies --
    use get_run() for the full record of one entry."""
    with _connection(db_path) as conn:
        rows = conn.execute(
            "SELECT id, kind, created_at, summary, model, request_json, response_json "
            "FROM runs ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [_row_to_entry(row) for row in rows]


def get_run(run_id: int, db_path: Path = DEFAULT_DB_PATH) -> HistoryEntry | None:
    with _connection(db_path) as conn:
        row = conn.execute(
            "SELECT id, kind, created_at, summary, model, request_json, response_json FROM runs WHERE id = ?",
            (run_id,),
        ).fetchone()
    return _row_to_entry(row) if row else None


def delete_run(run_id: int, db_path: Path = DEFAULT_DB_PATH) -> bool:
    with _connection(db_path) as conn:
        cursor = conn.execute("DELETE FROM runs WHERE id = ?", (run_id,))
        return cursor.rowcount > 0


def clear_history(db_path: Path = DEFAULT_DB_PATH) -> int:
    with _connection(db_path) as conn:
        cursor = conn.execute("DELETE FROM runs")
        return cursor.rowcount


def _row_to_entry(row: tuple) -> HistoryEntry:
    id_, kind, created_at, summary, model, request_json, response_json = row
    return HistoryEntry(
        id=id_,
        kind=kind,
        created_at=created_at,
        summary=summary,
        model=model,
        request=json.loads(request_json),
        response=json.loads(response_json),
    )
