"""tmux helper — lets the daemon/CLI run sessions inside tmux panes.

The daemon is the single client; tmux is just a way to give each session a visible,
attachable terminal. These are thin wrappers over the `tmux` CLI (no library dep).
If tmux isn't installed, callers get a clear error and can fall back to inline mode.
"""

from __future__ import annotations

import shutil
import subprocess

_PREFIX = "multivac"


def tmux_available() -> bool:
    return shutil.which("tmux") is not None


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["tmux", *args], capture_output=True, text=True, check=False
    )


def session_name(session_id: str) -> str:
    return f"{_PREFIX}-{session_id}"


def has_session(session_id: str) -> bool:
    if not tmux_available():
        return False
    return _run("has-session", "-t", session_name(session_id)).returncode == 0


def new_session(session_id: str, command: str) -> str:
    """Create a detached tmux session running `command`. Returns the tmux session name."""
    if not tmux_available():
        raise RuntimeError("tmux is not installed")
    name = session_name(session_id)
    if not has_session(session_id):
        _run("new-session", "-d", "-s", name, command)
    return name


def list_sessions() -> list[str]:
    if not tmux_available():
        return []
    out = _run("list-sessions", "-F", "#{session_name}")
    if out.returncode != 0:
        return []
    return [
        line[len(_PREFIX) + 1 :]
        for line in out.stdout.splitlines()
        if line.startswith(_PREFIX + "-")
    ]


def kill_session(session_id: str) -> bool:
    if not tmux_available():
        return False
    return _run("kill-session", "-t", session_name(session_id)).returncode == 0


def attach_command(session_id: str) -> list[str]:
    """The argv a CLI should exec to attach the user to a session's pane."""
    return ["tmux", "attach-session", "-t", session_name(session_id)]
