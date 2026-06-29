"""Multi-Vac — a solid, generic agent harness built on Pydantic AI.

Subsystems:
* `config.Settings`     — typed settings (live Cerebras Gemma 4 / offline stub).
* `provider`            — the single model choke-point.
* `skills`              — file-backed read/write skill store, exposed as agent tools.
* `hooks`               — pre/post run, pre/post tool, pre/post compact lifecycle hooks.
* `mcp`                 — MCP servers (stdio/http) attached as Pydantic AI toolsets.
* `compaction`          — automatic conversation summarization.
* `subagents`           — named specialised workers the model can spawn (depth-guarded).
* `harness.Multivac`    — assembles all of the above into one conversational agent.
"""

from .compaction import Compactor
from .config import Settings
from .configschema import HarnessConfig
from .harness import Multivac
from .hooks import Hook, HookManager
from .loader import load_config, resolve_ref
from .mcp import MCPServerConfig
from .provider import ModelProvider
from .skills import Skill, SkillStore
from .subagents import SubAgentRegistry, SubAgentSpec
from .types import (
    HookContext,
    HookDecision,
    HookEvent,
    HookResult,
    TurnResult,
)

__all__ = [
    "Settings",
    "Multivac",
    "HarnessConfig",
    "load_config",
    "resolve_ref",
    "ModelProvider",
    "SkillStore",
    "Skill",
    "Hook",
    "HookManager",
    "HookEvent",
    "HookDecision",
    "HookResult",
    "HookContext",
    "MCPServerConfig",
    "Compactor",
    "SubAgentRegistry",
    "SubAgentSpec",
    "TurnResult",
]

__version__ = "0.1.0"
