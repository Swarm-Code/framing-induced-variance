"""Tests for the YAML-bundle config system (offline)."""

from __future__ import annotations

from pathlib import Path

import pytest

from multivac import HookEvent, Multivac, Settings
from multivac.loader import load_config, resolve_ref

BUNDLE = str(Path(__file__).resolve().parent.parent / "configs" / "example")


def _offline_settings() -> Settings:
    return Settings(offline=True)


def test_load_example_bundle():
    cfg, bundle_dir = load_config(BUNDLE)
    assert cfg.name == "example-agent"
    assert cfg.provider.model == "gemma-4-31b"
    assert {t.ref for t in cfg.agent.tools} == {"tools:word_count", "tools:reverse_text"}
    assert any(h.name == "block-secrets" for h in cfg.hooks)
    assert any(s.name == "summarizer" for s in cfg.subagents)


def test_resolve_ref_imports_bundle_code():
    _, bundle_dir = load_config(BUNDLE)
    fn = resolve_ref("tools:word_count", bundle_dir)
    assert fn("a b c") == 3


def test_build_harness_from_config_offline(tmp_path):
    s = _offline_settings().model_copy(update={"skills_dir": str(tmp_path / "skills")})
    mv = Multivac.from_config(BUNDLE, settings=s)
    assert mv.mode == "offline"
    # System prompt came from prompts/main.md (not the hardcoded default).
    assert "Example-Agent" in mv.system_prompt
    # Bundle tools registered (plus builtin skill/subagent tools).
    names = {t.__name__ for t in mv._tools}
    assert {"word_count", "reverse_text"} <= names
    # Sub-agent from config present.
    assert "summarizer" in mv.subagents.names()
    # Inline skill seeded.
    assert mv.skills.exists("greeting-style")
    # Limits applied from YAML.
    assert mv.settings.tool_calls_limit == 15
    assert mv.settings.compact_after_messages == 30


def test_config_hooks_are_active(tmp_path):
    s = _offline_settings().model_copy(update={"skills_dir": str(tmp_path / "skills")})
    mv = Multivac.from_config(BUNDLE, settings=s)
    # post_run tag hook should be registered.
    assert mv.hooks.hooks_for(HookEvent.POST_RUN)
    # pre_tool secret-block hook should be registered.
    assert mv.hooks.hooks_for(HookEvent.PRE_TOOL)


def test_config_provider_override_changes_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("CEREBRAS_API_KEY", "")  # ensure offline regardless of env
    s = _offline_settings().model_copy(update={"skills_dir": str(tmp_path / "skills")})
    mv = Multivac.from_config(BUNDLE, settings=s)
    assert mv.settings.base_url == "https://api.cerebras.ai/v1"
    assert mv.settings.model == "gemma-4-31b"


def test_missing_bundle_raises():
    with pytest.raises(FileNotFoundError):
        load_config(str(Path(__file__).parent / "does-not-exist"))
