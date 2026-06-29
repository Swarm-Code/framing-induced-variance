"""Daemon scheduler — recurring jobs (cron/loop) + idle-session compaction.

Runs inside the daemon's asyncio event loop. Two kinds of recurring work:

* **loop jobs**  — a prompt/command re-run on a fixed interval (the ``/loop`` command),
  with a 3-day auto-expiry (matching the deconstruct/Claude-Code behaviour).
* **compaction** — periodically compact sessions whose history has grown, so long-lived
  sessions stay within context bounds without manual intervention.

Jobs are persisted to ~/.multivac/jobs.json so they survive a daemon restart.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from pathlib import Path
from typing import Awaitable, Callable

from pydantic import BaseModel, Field

from .protocol import daemon_home

_MAX_LOOP_SECONDS = 3 * 24 * 3600  # 3 days


class LoopJob(BaseModel):
    id: str
    session_id: str
    prompt: str
    interval_seconds: int
    created_at: float
    next_run: float
    runs: int = 0
    expires_at: float


class JobStore(BaseModel):
    loops: dict[str, LoopJob] = Field(default_factory=dict)


# A coroutine the scheduler calls to actually run a session turn.
RunTurn = Callable[[str, str], Awaitable[str]]
# A coroutine the scheduler calls to compact a session; returns True if compacted.
CompactFn = Callable[[str], Awaitable[bool]]


class Scheduler:
    def __init__(
        self,
        run_turn: RunTurn,
        compact_fn: CompactFn,
        *,
        home: Path | None = None,
        compaction_interval: float = 300.0,
    ) -> None:
        self.run_turn = run_turn
        self.compact_fn = compact_fn
        self.home = home or daemon_home()
        self.jobs_path = self.home / "jobs.json"
        self.store = self._load()
        self.compaction_interval = compaction_interval
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()
        self._session_ids: Callable[[], list[str]] = lambda: []

    # --------------------------------------------------------------- persistence
    def _load(self) -> JobStore:
        if self.jobs_path.exists():
            return JobStore.model_validate_json(self.jobs_path.read_text())
        return JobStore()

    def save(self) -> None:
        self.jobs_path.write_text(self.store.model_dump_json(indent=2))

    # --------------------------------------------------------------------- loops
    def add_loop(self, session_id: str, prompt: str, interval_seconds: int) -> LoopJob:
        interval_seconds = max(1, min(interval_seconds, _MAX_LOOP_SECONDS))
        now = time.time()
        job = LoopJob(
            id=uuid.uuid4().hex[:8],
            session_id=session_id,
            prompt=prompt,
            interval_seconds=interval_seconds,
            created_at=now,
            next_run=now + interval_seconds,
            expires_at=now + _MAX_LOOP_SECONDS,
        )
        self.store.loops[job.id] = job
        self.save()
        return job

    def remove_loop(self, job_id: str) -> bool:
        existed = self.store.loops.pop(job_id, None) is not None
        if existed:
            self.save()
        return existed

    def list_loops(self) -> list[LoopJob]:
        return list(self.store.loops.values())

    def bind_sessions(self, fn: Callable[[], list[str]]) -> None:
        """Provide a callable that returns current session ids (for compaction sweeps)."""
        self._session_ids = fn

    # ----------------------------------------------------------------- run loop
    def start(self) -> None:
        self._stop.clear()
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            await asyncio.gather(self._task, return_exceptions=True)

    async def _run(self) -> None:
        last_compaction = 0.0
        while not self._stop.is_set():
            now = time.time()

            # Due loop jobs.
            for job in list(self.store.loops.values()):
                if now >= job.expires_at:
                    self.store.loops.pop(job.id, None)
                    self.save()
                    continue
                if now >= job.next_run:
                    try:
                        await self.run_turn(job.session_id, job.prompt)
                    except Exception:  # noqa: BLE001 - a failing job must not kill the loop
                        pass
                    job.runs += 1
                    job.next_run = now + job.interval_seconds
                    self.save()

            # Periodic compaction sweep.
            if now - last_compaction >= self.compaction_interval:
                last_compaction = now
                for sid in self._session_ids():
                    try:
                        await self.compact_fn(sid)
                    except Exception:  # noqa: BLE001
                        pass

            try:
                await asyncio.wait_for(self._stop.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                pass
