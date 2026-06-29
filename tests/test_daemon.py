"""End-to-end daemon + client tests (offline; no network, isolated MULTIVAC_HOME)."""

from __future__ import annotations

import asyncio
import os
import time

import pytest

from multivac.daemon.commands import default_registry, parse_interval
from multivac.daemon.scheduler import Scheduler
from multivac.daemon.server import DaemonServer
from multivac.daemon.state import DaemonState, Profile, Provider, Workspace


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / "mvhome"
    h.mkdir()
    monkeypatch.setenv("MULTIVAC_HOME", str(h))
    monkeypatch.setenv("MULTIVAC_OFFLINE", "1")  # force offline harness
    return h


# ----------------------------------------------------------------- pure units


def test_parse_interval():
    assert parse_interval("5m do it") == (300, "do it")
    assert parse_interval("2h sweep") == (7200, "sweep")
    assert parse_interval("1d nightly") == (86400, "nightly")
    assert parse_interval("no interval here") == (600, "no interval here")


def test_command_registry_goal():
    reg = default_registry()
    goal = reg.get("goal")
    assert goal is not None
    prompt = goal.get_prompt("ship the harness")
    assert "ship the harness" in prompt
    assert "Goal-Oriented" in prompt


def test_state_seeds_defaults(home):
    st = DaemonState(home)
    assert "cerebras" in st.registry.providers
    assert "openai" in st.registry.providers
    assert "default" in st.registry.profiles


def test_state_session_lifecycle_offline(home):
    st = DaemonState(home)
    st.add_workspace(Workspace(name="proj", path=str(home), profile="default"))
    meta = st.create_session(workspace="proj")
    h = st.harness(meta.id)
    assert h.mode == "offline"
    r = h.chat("hello")
    assert r.output
    st.close_session(meta.id)
    assert meta.id not in st.registry.sessions


# ----------------------------------------------------------- full daemon I/O


def test_daemon_roundtrip(home):
    """Start the server in a thread; talk to it with the sync client."""
    import threading

    from multivac.daemon.client import DaemonClient

    sock = home / "daemon.sock"

    loop = asyncio.new_event_loop()
    server = DaemonServer(socket_path=str(sock), home=home)

    def run():
        asyncio.set_event_loop(loop)
        loop.run_until_complete(server.serve())

    t = threading.Thread(target=run, daemon=True)
    t.start()
    # Wait for the socket to appear.
    for _ in range(50):
        if sock.exists():
            break
        time.sleep(0.1)

    try:
        c = DaemonClient(str(sock))
        assert c.call("ping")["pong"] is True

        # providers / profiles / workspaces
        c.call("provider.add", name="cer2", base_url="x", model="m", api_key_env="K")
        assert any(p["name"] == "cer2" for p in c.call("provider.list"))
        c.call("workspace.add", name="ws", path=str(home), profile="default")
        meta = c.call("session.create", workspace="ws", title="t")
        sid = meta["id"]

        # chat turn (offline)
        out = c.call("session.chat", id=sid, message="hi")
        assert out["output"]

        # /help command path
        helped = c.call("session.chat", id=sid, message="/help")
        assert "/goal" in helped["output"]

        # /loop schedules a job
        looped = c.call("session.chat", id=sid, message="/loop 5m check status")
        assert "scheduled" in looped["output"]
        assert len(c.call("loop.list")) == 1

        # /goal injects + runs
        goaled = c.call("session.chat", id=sid, message="/goal ship it")
        assert goaled["output"]

        # skills CRUD via daemon
        c.call("skill.create", session=sid, name="note", description="d", body="b")
        assert any(s["name"] == "note" for s in c.call("skill.list", session=sid))
        c.close()
    finally:
        # Stop the server.
        loop.call_soon_threadsafe(server._stopping.set)  # noqa: SLF001
        t.join(timeout=5)
