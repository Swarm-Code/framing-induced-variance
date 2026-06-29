"""Event bus — fan-out of session activity to attached clients (TUI/CLI/GUI).

The daemon publishes events (turn start, assistant output, tool pre/post, sub-agent
start/end, compaction) onto an `EventBus`. A client calls `session.attach` and the
server streams matching events down that connection as protocol `Event` frames.

Harness subsystems run in worker threads (harness.chat is offloaded via to_thread), so
publishing is thread-safe: each subscriber holds an asyncio.Queue and we push with
`loop.call_soon_threadsafe`.
"""

from __future__ import annotations

import asyncio
from typing import Any


class Subscriber:
    def __init__(self, session_id: str | None) -> None:
        self.session_id = session_id  # None = all sessions
        self.queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=1000)

    def wants(self, session_id: str | None) -> bool:
        return self.session_id is None or self.session_id == session_id


class EventBus:
    def __init__(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        self._loop = loop
        self._subscribers: list[Subscriber] = []

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def subscribe(self, session_id: str | None) -> Subscriber:
        sub = Subscriber(session_id)
        self._subscribers.append(sub)
        return sub

    def unsubscribe(self, sub: Subscriber) -> None:
        if sub in self._subscribers:
            self._subscribers.remove(sub)

    def publish(self, event: str, session_id: str | None = None, **data: Any) -> None:
        """Thread-safe publish. Safe to call from worker threads."""
        payload = {"event": event, "data": {"session": session_id, **data}}
        if self._loop is None:
            return
        self._loop.call_soon_threadsafe(self._deliver, session_id, payload)

    def _deliver(self, session_id: str | None, payload: dict) -> None:
        for sub in list(self._subscribers):
            if sub.wants(session_id):
                try:
                    sub.queue.put_nowait(payload)
                except asyncio.QueueFull:
                    pass  # slow client: drop rather than block the daemon
