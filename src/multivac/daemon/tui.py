"""Multi-Vac TUI — attach to a daemon session and watch it live (stdlib curses).

Layout:
  ┌────────────────────────── Multi-Vac ──────────────────────────┐
  │ sessions (left)        │  event/conversation stream (right)   │
  │  > sess-abc  proj      │  you > ...                           │
  │    sess-def  other     │  multivac > ...                      │
  │                        │  ⚙ tool_pre  word_count {...}        │
  │                        │  ⚙ tool_post word_count -> 4         │
  │                        │  ↳ subagent summarizer: ...          │
  ├────────────────────────┴──────────────────────────────────────┤
  │ > _type a message or /command_                                 │
  └───────────────────────────────────────────────────────────────┘

Keys: Tab = switch focus, Up/Down = pick session, Enter = send, Ctrl-C = quit.

The TUI is a thin client: it sends `session.chat` via a request connection and reads
`session.attach` events via a second streaming connection — exactly what a GUI would do.
"""

from __future__ import annotations

import curses
import os
import threading
import time
from collections import deque

from .client import DaemonClient, DaemonError
from .protocol import default_socket_path

_EVENT_GLYPH = {
    "turn_start": "» you",
    "assistant": "« multivac",
    "tool_pre": "⚙ tool→",
    "tool_post": "⚙ tool←",
    "subagent_start": "↳ subagent→",
    "subagent_end": "↳ subagent←",
    "compacted": "↺ compacted",
}


class TUI:
    def __init__(self, socket_path: str | None = None) -> None:
        self.socket_path = socket_path or default_socket_path()
        self.rpc = DaemonClient(self.socket_path)
        self.stream = DaemonClient(self.socket_path)
        self.sessions: list[dict] = []
        self.selected = 0
        self.session_id: str | None = None
        self.lines: deque[str] = deque(maxlen=1000)
        self.input = ""
        self.focus = "input"  # "sessions" | "input"
        self._stream_thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()

    # ----------------------------------------------------------------- data
    def refresh_sessions(self) -> None:
        try:
            self.sessions = self.rpc.call("session.list")
        except DaemonError:
            self.sessions = []

    def attach(self, session_id: str) -> None:
        # Stop any prior stream.
        self._stop.set()
        if self._stream_thread:
            self._stream_thread.join(timeout=1.0)
        self._stop = threading.Event()
        self.session_id = session_id
        with self._lock:
            self.lines.clear()
            self.lines.append(f"-- attached to {session_id} --")
        self._stream_thread = threading.Thread(target=self._consume, daemon=True)
        self._stream_thread.start()

    def _consume(self) -> None:
        stream = DaemonClient(self.socket_path)
        try:
            for frame in stream.attach(self.session_id):
                if self._stop.is_set():
                    break
                ev = frame.get("event")
                if not ev:
                    continue  # attach ack
                self._render_event(ev, frame.get("data", {}))
        except DaemonError:
            pass
        finally:
            stream.close()

    def _render_event(self, ev: str, data: dict) -> None:
        glyph = _EVENT_GLYPH.get(ev, ev)
        if ev == "turn_start":
            text = data.get("message", "")
        elif ev == "assistant":
            text = data.get("output", "")
        elif ev in ("tool_pre",):
            text = f"{data.get('tool')} {data.get('args', {})}"
        elif ev == "tool_post":
            text = f"{data.get('tool')} -> {data.get('result', '')}"
        elif ev == "subagent_start":
            text = f"{data.get('name')}: {data.get('task', '')}"
        elif ev == "subagent_end":
            text = f"{data.get('name')} -> {data.get('output', '')}"
        else:
            text = str(data)
        with self._lock:
            for sub in f"{glyph}  {text}".splitlines() or [glyph]:
                self.lines.append(sub)

    def send(self, message: str) -> None:
        if not self.session_id or not message.strip():
            return
        try:
            self.rpc.call("session.chat", id=self.session_id, message=message)
        except DaemonError as e:
            with self._lock:
                self.lines.append(f"[error] {e}")

    # ------------------------------------------------------------------ render
    def run(self, stdscr) -> None:
        curses.curs_set(1)
        stdscr.nodelay(True)
        stdscr.timeout(120)
        curses.start_color()
        curses.use_default_colors()
        self.refresh_sessions()
        if self.sessions:
            self.attach(self.sessions[0]["id"])

        while True:
            self._draw(stdscr)
            try:
                ch = stdscr.getch()
            except KeyboardInterrupt:
                break
            if ch == -1:
                continue
            if ch in (3,):  # Ctrl-C
                break
            if ch == 9:  # Tab
                self.focus = "sessions" if self.focus == "input" else "input"
            elif self.focus == "sessions" and ch in (curses.KEY_UP, curses.KEY_DOWN):
                if self.sessions:
                    self.selected = (
                        self.selected + (1 if ch == curses.KEY_DOWN else -1)
                    ) % len(self.sessions)
            elif self.focus == "sessions" and ch in (curses.KEY_ENTER, 10, 13):
                if self.sessions:
                    self.attach(self.sessions[self.selected]["id"])
                    self.focus = "input"
            elif self.focus == "input":
                if ch in (curses.KEY_ENTER, 10, 13):
                    msg = self.input
                    self.input = ""
                    if msg in ("/quit", "/exit"):
                        break
                    threading.Thread(target=self.send, args=(msg,), daemon=True).start()
                elif ch in (curses.KEY_BACKSPACE, 127, 8):
                    self.input = self.input[:-1]
                elif 32 <= ch < 127:
                    self.input += chr(ch)

        self._stop.set()

    def _draw(self, stdscr) -> None:
        stdscr.erase()
        h, w = stdscr.getmaxyx()
        left = max(20, w // 4)

        stdscr.addnstr(0, 0, " Multi-Vac TUI — Tab switches pane, Enter sends, /quit ".ljust(w), w, curses.A_REVERSE)

        # session list
        for i, s in enumerate(self.sessions):
            y = 2 + i
            if y >= h - 3:
                break
            mark = ">" if i == self.selected else " "
            label = f"{mark} {s['id'][:8]} {s.get('title','')[:left-12]}"
            attr = curses.A_BOLD if s["id"] == self.session_id else curses.A_NORMAL
            if self.focus == "sessions" and i == self.selected:
                attr |= curses.A_REVERSE
            stdscr.addnstr(y, 0, label.ljust(left - 1), left - 1, attr)

        for y in range(1, h - 2):
            stdscr.addch(y, left - 1, curses.ACS_VLINE)

        # event/conversation stream (right)
        with self._lock:
            visible = list(self.lines)[-(h - 4):]
        for i, line in enumerate(visible):
            stdscr.addnstr(2 + i, left + 1, line, w - left - 2)

        # input bar
        prompt = f"[{self.session_id[:8] if self.session_id else 'no session'}] > "
        bar = (prompt + self.input)[-(w - 1):]
        stdscr.addnstr(h - 1, 0, bar.ljust(w - 1), w - 1,
                       curses.A_REVERSE if self.focus == "input" else curses.A_NORMAL)
        stdscr.refresh()


def _ensure_terminfo() -> str | None:
    """Make curses initialisable in this terminal.

    Some environments (notably tmux with TERM=tmux-256color, or a Python whose bundled
    ncurses lacks that entry) raise 'setupterm: could not find terminfo database'. We
    probe the current TERM, then fall back to widely-available entries, and make sure
    TERMINFO points at the standard system directories. Returns the TERM that works, or
    None if curses cannot be initialised at all.
    """
    import curses

    # Ensure the standard terminfo directories are searched.
    std_dirs = [
        d
        for d in ("/usr/share/terminfo", "/etc/terminfo", "/lib/terminfo")
        if os.path.isdir(d)
    ]
    if std_dirs and not os.environ.get("TERMINFO_DIRS"):
        os.environ["TERMINFO_DIRS"] = ":".join(std_dirs)

    current = os.environ.get("TERM", "")
    candidates = [
        current,
        "tmux-256color",
        "screen-256color",
        "xterm-256color",
        "xterm",
        "vt100",
        "ansi",
    ]
    for term in candidates:
        if not term:
            continue
        try:
            curses.setupterm(term)
        except Exception:  # noqa: BLE001
            continue
        os.environ["TERM"] = term
        return term
    return None


def run_tui(socket_path: str | None = None) -> int:
    import curses

    tui = TUI(socket_path)
    if not tui.rpc.is_running():
        print("daemon not running — start it with: multivac daemon start")
        return 1

    term = _ensure_terminfo()
    if term is None:
        print(
            "Could not initialise the terminal UI: no usable terminfo entry found.\n"
            f"Your TERM={os.environ.get('TERM', '')!r} isn't in this Python's terminfo "
            "database. Try one of:\n"
            "  TERM=xterm-256color multivac tui\n"
            "  TERM=xterm multivac tui\n"
            "Or use the plain REPL instead: multivac chat",
            file=__import__("sys").stderr,
        )
        return 1
    try:
        curses.wrapper(tui.run)
    except KeyboardInterrupt:
        pass
    except curses.error as e:
        print(
            f"Terminal UI error: {e}\n"
            "Fallback: use the plain REPL — multivac chat",
            file=__import__("sys").stderr,
        )
        return 1
    return 0
