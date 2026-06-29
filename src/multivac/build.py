"""Bridge: a validated `HarnessConfig` bundle -> a live `Multivac` instance.

Kept separate from `harness.py` so the config machinery (YAML, imports, file refs) does
not weigh on the programmatic API, and to avoid an import cycle.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

from .config import Settings
from .configschema import HarnessConfig
from .loader import load_config, read_prompt, resolve_ref
from .mcp import MCPServerConfig
from .subagents import SubAgentSpec
from .types import HookEvent


def settings_from_config(cfg: HarnessConfig, base: Settings | None = None) -> Settings:
    """Produce Settings from a bundle, layering provider + limits + compaction over env."""
    s = base or Settings.load()
    p = cfg.provider

    api_key = s.api_key
    if p.api_key_env:
        api_key = os.environ.get(p.api_key_env) or api_key
    if p.api_key:
        api_key = p.api_key

    return s.model_copy(
        update={
            "api_key": api_key,
            "base_url": p.base_url or s.base_url,
            "model": p.model or s.model,
            "offline": p.offline if p.offline is not None else s.offline,
            "skills_dir": cfg.skills_dir or s.skills_dir,
            "compact_after_messages": cfg.compaction.after_messages,
            "compact_keep_recent": cfg.compaction.keep_recent,
            "max_subagent_depth": cfg.limits.max_subagent_depth,
            "tool_calls_limit": cfg.limits.tool_calls_limit,
            "request_limit": cfg.limits.request_limit,
        }
    )


def _resolve_tools(tool_refs, bundle_dir: Path) -> list[Callable]:
    tools: list[Callable] = []
    for t in tool_refs:
        fn = resolve_ref(t.ref, bundle_dir)
        if t.name:
            fn.__name__ = t.name  # surface the configured tool name to the model
        tools.append(fn)
    return tools


def build_from_config(cls, path: str, *, settings: Settings | None = None):
    """Construct a `Multivac` (passed as `cls`) entirely from a YAML bundle."""
    cfg, bundle_dir = load_config(path)
    settings = settings_from_config(cfg, settings)

    # Prompts (inline or file).
    system_prompt = read_prompt(
        bundle_dir, cfg.agent.system_prompt, cfg.agent.system_prompt_file
    )
    summarizer_prompt = read_prompt(
        bundle_dir, cfg.compaction.summarizer_prompt, cfg.compaction.summarizer_prompt_file
    )

    # Top-level agent tools.
    extra_tools = _resolve_tools(cfg.agent.tools, bundle_dir)

    # Sub-agents (inherit provider; override prompt/tools).
    subagent_specs: list[SubAgentSpec] = []
    for sa in cfg.subagents:
        sa_prompt = read_prompt(bundle_dir, sa.system_prompt, sa.system_prompt_file) or ""
        subagent_specs.append(
            SubAgentSpec(
                name=sa.name,
                description=sa.description,
                system_prompt=sa_prompt,
                tools=_resolve_tools(sa.tools, bundle_dir),
            )
        )

    # MCP servers.
    mcp_servers = [
        MCPServerConfig(
            name=m.name,
            transport=m.transport,
            command=m.command,
            args=m.args,
            env=m.env,
            cwd=m.cwd,
            url=m.url,
            headers=m.headers,
            tool_prefix=m.tool_prefix,
        )
        for m in cfg.mcp_servers
    ]

    harness = cls(
        settings,
        system_prompt=system_prompt,
        summarizer_prompt=summarizer_prompt,
        mcp_servers=mcp_servers,
        extra_tools=extra_tools,
        subagents=subagent_specs,
    )

    # Seed skills declared inline in the bundle.
    for sk in cfg.skills:
        body = sk.body
        if sk.body_file:
            body = (bundle_dir / sk.body_file).read_text()
        if not harness.skills.exists(sk.name):
            harness.skills.create(sk.name, sk.description, body, tags=sk.tags)

    # Register hooks (imported from the bundle's own code).
    for h in cfg.hooks:
        fn = resolve_ref(h.ref, bundle_dir)
        events = [HookEvent(e) for e in h.events]
        harness.on(events, fn, name=h.name, priority=h.priority)

    return harness
