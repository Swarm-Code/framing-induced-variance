"""Offline tests for the Multi-Vac harness (no network required)."""

from __future__ import annotations

import pytest

from multivac import (
    HookContext,
    HookDecision,
    HookEvent,
    HookResult,
    MCPServerConfig,
    Multivac,
    Settings,
    SkillStore,
    SubAgentSpec,
)
from multivac.mcp import build_mcp_toolsets


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        offline=True,
        skills_dir=str(tmp_path / "skills"),
        sessions_dir=str(tmp_path / "sessions"),
        compact_after_messages=4,
        compact_keep_recent=2,
    )


@pytest.fixture
def harness(settings) -> Multivac:
    return Multivac(settings)


# ----------------------------------------------------------------------- basic


def test_harness_offline_mode(harness):
    assert harness.mode == "offline"


def test_chat_accumulates_history(harness):
    r1 = harness.chat("hello")
    assert r1.output and not r1.blocked
    n = len(harness.history)
    assert n >= 2
    harness.chat("again")
    assert len(harness.history) > n


# ----------------------------------------------------------------------- skills


def test_skill_store_crud(settings):
    store = SkillStore(settings.skills_dir)
    assert store.list() == []
    s = store.create("deploy-app", "How to deploy", "1. build\n2. ship")
    assert s.version == 1
    assert store.exists("deploy-app")
    s2 = store.patch("deploy-app", body="1. build\n2. test\n3. ship")
    assert s2.version == 2
    assert "test" in store.view("deploy-app").body
    store.archive("deploy-app")
    assert not store.exists("deploy-app")


def test_skill_invalid_name(settings):
    store = SkillStore(settings.skills_dir)
    with pytest.raises(ValueError):
        store.create("Bad Name", "x", "y")


def test_skill_roundtrip_markdown(settings):
    store = SkillStore(settings.skills_dir)
    store.create("note", "desc", "body text", tags=["a", "b"])
    loaded = store.view("note")
    assert loaded.tags == ["a", "b"]
    assert loaded.description == "desc"


def test_harness_exposes_skill_tools(harness):
    tool_names = {t.__name__ for t in harness._tools}
    assert {"skill_list", "skill_view", "skill_create", "skill_patch"} <= tool_names


# ------------------------------------------------------------------------ hooks


def test_hook_block(harness):
    def deny(ctx: HookContext) -> HookResult:
        return HookResult(decision=HookDecision.BLOCK, reason="nope")

    harness.on([HookEvent.PRE_RUN], deny, name="deny")
    result = harness.chat("hello")
    assert result.blocked
    assert "nope" in result.block_reason


def test_hook_modify_user_message(harness):
    seen = {}

    def rewrite(ctx: HookContext) -> HookResult:
        return HookResult(decision=HookDecision.MODIFY, payload="REWRITTEN")

    def capture(ctx: HookContext):
        seen["msg"] = ctx.user_message
        return None

    harness.on([HookEvent.PRE_RUN], rewrite, name="rewrite", priority=10)
    harness.on([HookEvent.PRE_RUN], capture, name="capture", priority=20)
    harness.chat("original")
    assert seen["msg"] == "REWRITTEN"


def test_tool_hook_blocks_tool(harness):
    def block_create(ctx: HookContext):
        if ctx.tool_name == "skill_create":
            return HookResult(decision=HookDecision.BLOCK, reason="read-only mode")
        return None

    harness.on([HookEvent.PRE_TOOL], block_create, name="ro")
    # Find the wrapped skill_create tool and call it directly.
    tool = next(t for t in harness._tools if t.__name__ == "skill_create")
    out = tool(name="x", description="d", body="b")
    assert "blocked by hook" in out


# -------------------------------------------------------------------- subagents


def test_subagent_register_and_run(harness):
    harness.add_subagent(
        SubAgentSpec(name="echoer", system_prompt="Echo the task.", description="echoes")
    )
    assert "echoer" in harness.subagents.names()
    out = harness.subagents.run("echoer", "hello")
    assert isinstance(out, str) and out


def test_subagent_depth_guard(harness):
    harness.add_subagent(SubAgentSpec(name="a", system_prompt="x"))
    out = harness.subagents.run("a", "task", depth=harness.settings.max_subagent_depth)
    assert "depth limit" in out


def test_unknown_subagent(harness):
    out = harness.subagents.run.__self__  # registry
    res = harness.subagents.describe()
    assert "No sub-agents" in res


# -------------------------------------------------------------------- compaction


def test_compaction_triggers_and_keeps_tail(harness):
    # after_messages=4, keep_recent=2 in fixture.
    for i in range(5):
        harness.chat(f"message {i}")
    # History should have been compacted at least once and stay bounded.
    assert harness.last_compaction_summary != ""
    assert len(harness.history) <= harness.settings.compact_after_messages + 2


def test_manual_compact_noop_when_short(harness):
    harness.chat("hi")
    # Only a couple of messages -> below keep_recent boundary may noop.
    did = harness.compact()
    assert isinstance(did, bool)


# -------------------------------------------------------------------------- mcp


def test_mcp_empty_configs():
    assert build_mcp_toolsets([]) == []


def test_mcp_config_model():
    cfg = MCPServerConfig(
        name="fs", transport="stdio", command="uvx", args=["mcp-server-fs"]
    )
    assert cfg.transport == "stdio"
    # mcp package is installed in this env -> toolsets should build without error.
    toolsets = build_mcp_toolsets([cfg])
    assert len(toolsets) == 1


def test_mcp_http_requires_url():
    cfg = MCPServerConfig(name="bad", transport="http")
    with pytest.raises(ValueError):
        build_mcp_toolsets([cfg])


# ------------------------------------------------- Hermes lessons: hardening


def test_tool_error_returned_as_feedback(harness):
    """A throwing tool must NOT crash the run — its error comes back as feedback."""

    def boom() -> str:
        raise RuntimeError("kaboom")

    wrapped = harness._wrap_tool(boom)
    out = wrapped()
    assert "tool error in boom" in out
    assert "kaboom" in out


def test_tool_error_fires_post_tool_hook(harness):
    seen = {}

    def boom() -> str:
        raise ValueError("nope")

    def watch(ctx):
        if ctx.metadata.get("error"):
            seen["err"] = ctx.tool_result
        return None

    harness.on([HookEvent.POST_TOOL], watch, name="watch")
    harness._wrap_tool(boom)()
    assert "nope" in seen.get("err", "")


def test_circuit_breaker_settings_present(harness):
    assert harness.settings.tool_calls_limit > 0
    assert harness.settings.request_limit > 0


def test_generic_openai_key_detected(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-generic")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("OPENAI_MODEL", "some-model")
    s = Settings.load(dotenv=None)
    assert s.api_key == "sk-generic"
    assert s.base_url == "https://example.test/v1"
    assert s.model == "some-model"
    assert s.is_live
