"""Tests for cwd workspace inference, bundle auto-load, and event streaming."""

from __future__ import annotations

import asyncio
import os
import threading
import time
from pathlib import Path

import pytest

from multivac.daemon.state import DaemonState

EXAMPLE = Path(__file__).resolve().parent.parent / "configs" / "example"


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / "mvhome"
    h.mkdir()
    monkeypatch.setenv("MULTIVAC_HOME", str(h))
    monkeypatch.setenv("MULTIVAC_OFFLINE", "1")
    return h


# ------------------------------------------------------ cwd inference + bundle


def test_discover_bundle_finds_example():
    found = DaemonState.discover_bundle(str(EXAMPLE))
    assert found is not None
    assert found.endswith("multivac.yaml")


def test_infer_workspace_from_cwd(home):
    st = DaemonState(home)
    ws = st.infer_workspace(str(EXAMPLE))
    assert ws.path == str(EXAMPLE.resolve())
    assert ws.bundle and ws.bundle.endswith("multivac.yaml")
    # Idempotent: same cwd -> same workspace.
    ws2 = st.infer_workspace(str(EXAMPLE))
    assert ws2.name == ws.name


def test_inferred_session_loads_bundle_context(home):
    """A session created from the example cwd must load the bundle's prompt/tools/etc."""
    st = DaemonState(home)
    ws = st.infer_workspace(str(EXAMPLE))
    meta = st.create_session(workspace=ws.name)
    h = st.harness(meta.id)
    # System prompt from prompts/main.md (not the hardcoded default).
    assert "Example-Agent" in h.system_prompt
    # Bundle tools present alongside builtins.
    names = {t.__name__ for t in h._tools}
    assert {"word_count", "reverse_text"} <= names
    # Inline skill seeded + sub-agent + hooks from the bundle.
    assert h.skills.exists("greeting-style")
    assert "summarizer" in h.subagents.names()
    from multivac.types import HookEvent

    assert h.hooks.hooks_for(HookEvent.POST_RUN)  # tag_output hook
    assert h.hooks.hooks_for(HookEvent.PRE_TOOL)  # block_secret_tools hook


# ----------------------------------------------------------- event streaming


def test_attach_streams_tool_hook_events(home):
    from multivac.daemon.client import DaemonClient
    from multivac.daemon.server import DaemonServer

    sock = home / "daemon.sock"
    loop = asyncio.new_event_loop()
    server = DaemonServer(socket_path=str(sock), home=home)

    def run():
        asyncio.set_event_loop(loop)
        loop.run_until_complete(server.serve())

    threading.Thread(target=run, daemon=True).start()
    for _ in range(50):
        if sock.exists():
            break
        time.sleep(0.1)

    try:
        c = DaemonClient(str(sock))
        sid = c.call("session.create")["id"]

        events: list[dict] = []

        def listen():
            ac = DaemonClient(str(sock))
            for frame in ac.attach(sid):
                events.append(frame)
                if sum(1 for e in events if e.get("event") == "tool_post") >= 1:
                    break

        threading.Thread(target=listen, daemon=True).start()
        time.sleep(0.3)

        # Trigger a real tool call through the instrumented harness.
        h = server._harness(sid)  # noqa: SLF001
        tool = next(t for t in h._tools if t.__name__ == "skill_create")
        tool(name="demo", description="d", body="b")
        time.sleep(0.5)

        kinds = [e.get("event") for e in events if e.get("event")]
        assert "tool_pre" in kinds
        assert "tool_post" in kinds
        pre = next(e["data"] for e in events if e.get("event") == "tool_pre")
        assert pre["tool"] == "skill_create"
        assert pre["args"]["name"] == "demo"
    finally:
        loop.call_soon_threadsafe(server._stopping.set)  # noqa: SLF001
