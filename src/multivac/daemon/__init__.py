"""Multi-Vac daemon — one process owns all state; CLI/GUI are thin clients.

* `protocol`  — newline-delimited JSON frames over a Unix socket.
* `state`     — providers, profiles, workspaces, sessions (persisted to ~/.multivac).
* `scheduler` — cron jobs, /loop recurring commands, idle-session compaction.
* `server`    — the asyncio daemon: dispatches RPC methods, runs the scheduler.
* `client`    — a small synchronous client used by the CLI.
* `tmux`      — helper so the daemon/CLI can run sessions in tmux panes.
"""

from .client import DaemonClient
from .protocol import Event, Request, Response, daemon_home, default_socket_path

__all__ = [
    "DaemonClient",
    "Request",
    "Response",
    "Event",
    "default_socket_path",
    "daemon_home",
]
