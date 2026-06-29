"""Sub-agents.

A *sub-agent* is a named, specialised worker the main harness can spawn for a bounded
task. Each sub-agent has its own system prompt and optional tools, runs to a single
final string result, and does not see the parent conversation unless that context is
passed in the task text. A depth guard prevents unbounded recursion.

The main harness exposes a `spawn_subagent(name, task)` tool so the model can delegate.
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from pydantic_ai import Agent, Tool
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.usage import UsageLimits

from .provider import ModelProvider


class SubAgentSpec(BaseModel):
    """Declarative definition of a sub-agent."""

    name: str
    system_prompt: str
    description: str = ""
    tools: list = Field(default_factory=list)

    model_config = {"arbitrary_types_allowed": True}


class SubAgentRegistry:
    """Holds sub-agent specs and runs them with a depth guard."""

    def __init__(
        self,
        provider: ModelProvider,
        *,
        max_depth: int = 3,
        tool_calls_limit: int = 20,
        request_limit: int = 25,
    ) -> None:
        self.provider = provider
        self.max_depth = max_depth
        self.tool_calls_limit = tool_calls_limit
        self.request_limit = request_limit
        self._specs: dict[str, SubAgentSpec] = {}
        self._agents: dict[str, Agent] = {}

    def register(self, spec: SubAgentSpec) -> None:
        self._specs[spec.name] = spec
        self._agents.pop(spec.name, None)  # rebuild lazily

    def names(self) -> list[str]:
        return sorted(self._specs)

    def describe(self) -> str:
        if not self._specs:
            return "No sub-agents registered."
        return "\n".join(
            f"- {s.name}: {s.description or s.system_prompt[:60]}"
            for s in self._specs.values()
        )

    def _agent_for(self, name: str) -> Agent:
        if name not in self._specs:
            raise KeyError(f"unknown sub-agent: {name}")
        if name not in self._agents:
            spec = self._specs[name]
            # Uniform strict=False so providers like Cerebras don't reject mixed tools.
            tools = [Tool(fn, strict=False) for fn in spec.tools]
            self._agents[name] = Agent(
                self.provider.model(),
                system_prompt=spec.system_prompt,
                tools=tools,
                name=f"subagent:{name}",
            )
        return self._agents[name]

    def run(self, name: str, task: str, *, depth: int = 0) -> str:
        if depth >= self.max_depth:
            return f"[sub-agent depth limit {self.max_depth} reached; refusing to recurse]"
        agent = self._agent_for(name)
        limits = UsageLimits(
            tool_calls_limit=self.tool_calls_limit, request_limit=self.request_limit
        )
        try:
            result = agent.run_sync(task, usage_limits=limits)
        except UsageLimitExceeded as e:
            return f"[sub-agent {name} stopped: tool/usage limit reached — {e}]"
        return str(result.output)
