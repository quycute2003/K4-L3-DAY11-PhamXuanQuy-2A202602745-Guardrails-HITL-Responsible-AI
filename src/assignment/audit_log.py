"""
Assignment 11 — Audit Log starter (TODO).

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[str, dict] = {}
        self._open_by_user: dict[str, deque[str]] = defaultdict(deque)
        self._sequence = 0

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None):
        """Store an input and its start time, returning the effective request ID."""
        user_id = str(user_id or "anonymous")
        self._sequence += 1
        effective_id = request_id or f"{user_id}-{self._sequence}"
        entry = {
            "request_id": effective_id,
            "user_id": user_id,
            "input": str(text or ""),
            "started_at": utc_now_iso(),
            "_started_monotonic": time.perf_counter(),
        }
        self._open[effective_id] = entry
        self._open_by_user[user_id].append(effective_id)
        return effective_id

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ):
        """Finish an audit record with the decision and elapsed time."""
        user_id = str(user_id or "anonymous")
        effective_id = request_id
        if effective_id is None:
            pending = self._open_by_user.get(user_id)
            effective_id = pending.popleft() if pending else None
        else:
            pending = self._open_by_user.get(user_id)
            if pending:
                try:
                    pending.remove(effective_id)
                except ValueError:
                    pass

        started = self._open.pop(effective_id, None) if effective_id else None
        now = time.perf_counter()
        started_monotonic = (
            started.pop("_started_monotonic") if started else now
        )
        log_entry = started or {
            "request_id": effective_id or f"{user_id}-orphan-output",
            "user_id": user_id,
            "input": "",
            "started_at": utc_now_iso(),
        }
        log_entry.update(
            {
                "output": str(text or ""),
                "blocked": bool(blocked),
                "layer": layer,
                "completed_at": utc_now_iso(),
                "latency_ms": round(max(0.0, now - started_monotonic) * 1000, 3),
            }
        )
        self.logs.append(log_entry)
        return log_entry

    def export_json(self, filepath: str | None = None):
        """Write logs to disk (JSON array) under repo-root ``outputs/`` by default."""
        path = Path(filepath or default_audit_log_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.logs, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return str(path)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
