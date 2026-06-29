"""Declarative harness configuration (the YAML bundle schema).

A *bundle* fully describes a harness instance so it can be dropped into a folder and
run with zero edits to the base harness (Pi-mono style). Every previously hardcoded
value — provider, system prompts, tools, hooks, skills, sub-agents, MCP servers,
compaction, limits — is expressed here.

Code references (hooks, tools) use a dotted **import spec**: ``module.path:attribute``
(e.g. ``mybench.hooks:deny_shell``). The loader imports and binds them at build time,
so custom behaviour lives in the bundle's own package — never in the base harness.

Layout of a bundle folder::

    my_bundle/
      multivac.yaml            # the bundle (or several *.yaml merged)
      prompts/
        main.md                # referenced via system_prompt_file: prompts/main.md
      skills/                  # preloaded skill .md files (optional)
      hooks.py / tools.py      # your code, referenced by dotted path

YAML keys map 1:1 to the models below.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ProviderConfig(BaseModel):
    """Which model to talk to. OpenAI-compatible; offline falls back to a stub."""

    # Environment variable names to read the key/url/model from (resolved at load).
    api_key_env: str | None = None  # e.g. "CEREBRAS_API_KEY" or "OPENAI_API_KEY"
    api_key: str | None = None  # discouraged; prefer api_key_env
    base_url: str | None = None
    model: str | None = None
    offline: bool | None = None  # force the deterministic stub model


class ToolRef(BaseModel):
    """A tool implemented in the bundle's own code, referenced by dotted path."""

    ref: str  # "module.path:callable"
    name: str | None = None  # optional override of the tool name


class HookConfig(BaseModel):
    """A lifecycle hook implemented in the bundle's own code."""

    name: str
    ref: str  # "module.path:callable" — fn(HookContext) -> HookResult | None
    events: list[str]  # HookEvent values: pre_run/post_run/pre_tool/post_tool/...
    priority: int = 100


class SkillSeed(BaseModel):
    """A skill defined inline in the bundle (written to the skill store on build)."""

    name: str
    description: str = ""
    body: str = ""
    body_file: str | None = None  # path (relative to bundle) to a markdown body
    tags: list[str] = Field(default_factory=list)


class SubAgentConfig(BaseModel):
    """A named sub-agent. Inherits the parent provider; overrides prompt/tools only."""

    name: str
    description: str = ""
    system_prompt: str | None = None
    system_prompt_file: str | None = None
    tools: list[ToolRef] = Field(default_factory=list)


class MCPConfig(BaseModel):
    """An MCP server attached as a toolset."""

    name: str
    transport: str = "stdio"  # stdio | http | sse
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    cwd: str | None = None
    url: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)
    tool_prefix: str | None = None


class CompactionConfig(BaseModel):
    after_messages: int = 40
    keep_recent: int = 8
    summarizer_prompt: str | None = None
    summarizer_prompt_file: str | None = None


class LimitsConfig(BaseModel):
    tool_calls_limit: int = 20
    request_limit: int = 25
    max_subagent_depth: int = 3


class AgentConfig(BaseModel):
    """The top-level agent voice."""

    system_prompt: str | None = None
    system_prompt_file: str | None = None
    tools: list[ToolRef] = Field(default_factory=list)


class HarnessConfig(BaseModel):
    """The whole bundle. This is what `Multivac.from_config` consumes."""

    name: str = "multivac"
    provider: ProviderConfig = Field(default_factory=ProviderConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    skills_dir: str | None = None
    skills: list[SkillSeed] = Field(default_factory=list)
    hooks: list[HookConfig] = Field(default_factory=list)
    subagents: list[SubAgentConfig] = Field(default_factory=list)
    mcp_servers: list[MCPConfig] = Field(default_factory=list)
    compaction: CompactionConfig = Field(default_factory=CompactionConfig)
    limits: LimitsConfig = Field(default_factory=LimitsConfig)
