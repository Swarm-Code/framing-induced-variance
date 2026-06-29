"""Daemon wire protocol — newline-delimited JSON over a Unix domain socket.

One long-lived **daemon** process owns all state (providers, profiles, workspaces,
sessions, scheduler). The CLI and any future GUI are *thin clients* that speak this
protocol, so they are always views over the same execution engine.

Frames (one JSON object per line):

* Request : {"id": int, "method": str, "params": {...}}
* Response: {"id": int, "ok": bool, "result": {...}}  | {"id": int, "ok": false, "error": str}
* Event   : {"event": str, "data": {...}}            # server-pushed, no id

Keeping it dependency-free (stdlib `asyncio` + `json`) means the daemon runs anywhere
Python does, and the same socket can later back a GUI or a tmux-driven TUI.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Request(BaseModel):
    id: int
    method: str
    params: dict[str, Any] = Field(default_factory=dict)


class Response(BaseModel):
    id: int
    ok: bool = True
    result: Any = None
    error: str | None = None

    @classmethod
    def success(cls, id: int, result: Any = None) -> "Response":
        return cls(id=id, ok=True, result=result)

    @classmethod
    def failure(cls, id: int, error: str) -> "Response":
        return cls(id=id, ok=False, error=error)


class Event(BaseModel):
    event: str
    data: dict[str, Any] = Field(default_factory=dict)


def default_socket_path() -> str:
    """Default daemon socket location (honours MULTIVAC_HOME / XDG-ish)."""
    import os
    from pathlib import Path

    home = os.environ.get("MULTIVAC_HOME")
    base = Path(home) if home else Path.home() / ".multivac"
    base.mkdir(parents=True, exist_ok=True)
    return str(base / "daemon.sock")


def daemon_home() -> "Path":  # type: ignore[name-defined]
    import os
    from pathlib import Path

    home = os.environ.get("MULTIVAC_HOME")
    base = Path(home) if home else Path.home() / ".multivac"
    base.mkdir(parents=True, exist_ok=True)
    return base
