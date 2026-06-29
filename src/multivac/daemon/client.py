"""Synchronous daemon client — blocking JSON-line RPC over the Unix socket.

Used by the CLI (and usable by any tool). Each call opens-or-reuses a connection,
sends one Request line, and reads until it gets the matching Response.
"""

from __future__ import annotations

import json
import socket
from typing import Any

from .protocol import default_socket_path


class DaemonError(RuntimeError):
    pass


class DaemonClient:
    def __init__(self, socket_path: str | None = None, *, timeout: float = 30.0) -> None:
        self.socket_path = socket_path or default_socket_path()
        self.timeout = timeout
        self._sock: socket.socket | None = None
        self._buf = b""
        self._next_id = 1

    # ---------------------------------------------------------------- connection
    def connect(self) -> None:
        if self._sock is not None:
            return
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(self.timeout)
        try:
            s.connect(self.socket_path)
        except (FileNotFoundError, ConnectionRefusedError) as e:
            raise DaemonError(
                f"daemon not running at {self.socket_path} "
                f"(start it with: multivac daemon start)"
            ) from e
        self._sock = s

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close()
            self._sock = None
            self._buf = b""

    def is_running(self) -> bool:
        try:
            self.connect()
            return True
        except DaemonError:
            return False

    # ----------------------------------------------------------------------- rpc
    def call(self, method: str, **params: Any) -> Any:
        self.connect()
        assert self._sock is not None
        req_id = self._next_id
        self._next_id += 1
        line = json.dumps({"id": req_id, "method": method, "params": params}) + "\n"
        self._sock.sendall(line.encode())

        # Read until we get the response with our id (skip any interleaved events).
        while True:
            obj = self._read_frame()
            if obj is None:
                raise DaemonError("daemon closed the connection")
            if obj.get("event"):
                continue  # ignore server events in the synchronous path
            if obj.get("id") == req_id:
                if obj.get("ok"):
                    return obj.get("result")
                raise DaemonError(obj.get("error") or "unknown daemon error")

    def _read_frame(self) -> dict | None:
        assert self._sock is not None
        while b"\n" not in self._buf:
            chunk = self._sock.recv(65536)
            if not chunk:
                return None
            self._buf += chunk
        line, _, rest = self._buf.partition(b"\n")
        self._buf = rest
        if not line.strip():
            return {}
        return json.loads(line.decode())

    def attach(self, session: str | None = None):
        """Yield events from a session (or all sessions) until disconnected.

        Opens a dedicated connection (don't reuse for regular calls). First yields the
        attach ack, then a stream of event dicts: {"event": str, "data": {...}}.
        """
        self.connect()
        assert self._sock is not None
        req_id = self._next_id
        self._next_id += 1
        line = json.dumps(
            {"id": req_id, "method": "session.attach", "params": {"session": session}}
        ) + "\n"
        self._sock.sendall(line.encode())
        while True:
            obj = self._read_frame()
            if obj is None:
                return
            yield obj

    def __enter__(self) -> "DaemonClient":
        self.connect()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
