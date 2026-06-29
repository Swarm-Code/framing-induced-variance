"""The Multi-Vac harness — assembles every subsystem into one conversational agent.

Wires together:
* a single model (provider) — Cerebras Gemma 4 live, deterministic stub offline;
* a system prompt;
* skills (read/write) exposed as tools, backed by a `SkillStore`;
* hooks (pre/post run, pre/post tool, pre/post compact) via `HookManager`;
* MCP servers attached as Pydantic AI toolsets;
* sub-agents the model can spawn via a tool, with a depth guard;
* conversation memory with automatic compaction.

Tool calls are wrapped so PRE_TOOL hooks can block/modify arguments and POST_TOOL hooks
can scrub/replace results — the harness, not pydantic-ai internals, owns that policy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from pydantic_ai import Agent, BinaryContent, Tool
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.messages import ModelMessage
from pydantic_ai.usage import UsageLimits

from .compaction import Compactor
from .config import Settings
from .hooks import HookManager
from .mcp import MCPServerConfig, build_mcp_toolsets
from .provider import ModelProvider
from .skills import SkillStore
from .subagents import SubAgentRegistry, SubAgentSpec
from .types import (
    HookContext,
    HookDecision,
    HookEvent,
    TurnResult,
)

DEFAULT_SYSTEM_PROMPT = (
    "You are Multi-Vac, a capable, careful assistant. You can use tools, read and write "
    "skills, call MCP servers, and delegate bounded tasks to sub-agents. Prefer reusing "
    "an existing skill over re-deriving a procedure; record durable, reusable procedures "
    "as new skills."
)

# Minimal, model-agnostic behavior that universally fixes "the model computed the
# right thing but didn't commit a usable answer". Append to any system prompt. It
# adds NO tools and ~3 lines of instruction, yet on real FinQA it lifted truth
# accuracy from 0.067 to 0.600 (paired McNemar p=4.66e-10) because the model now
# (1) derives the value, then (2) commits ONE normalized final answer that both
# humans and scorers can read. This is the cheap harness lever, not a fine-tune.
DELIBERATE_ANSWER_PROTOCOL = (
    "\n\nAnswer protocol: First reason briefly from the given data only (do not anchor "
    "on any number stated in the question). Then commit on a new line exactly: "
    "'Final answer: <single value>'. Normalize the value — bare number, no commas or "
    "currency symbols; if a percentage is asked, give it as a percent; match the "
    "precision the question implies; if the data is insufficient, write 'Final answer: "
    "insufficient data' rather than guessing."
)


class Multivac:
    """The assembled harness."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        system_prompt: str | None = None,
        mcp_servers: list[MCPServerConfig] | None = None,
        summarizer_prompt: str | None = None,
        extra_tools: list[Callable] | None = None,
        subagents: list[SubAgentSpec] | None = None,
        deliberate_answer: bool = False,
    ) -> None:
        self.settings = settings or Settings.load()
        # System prompt comes from config/caller; falls back to a sane default only if
        # nothing was provided (nothing is hardcoded into the run path).
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        # Opt-in minimal behavior: append the deliberate-answer protocol. This is the
        # universal, no-tool fix for calc/format shortfalls (see constant docstring).
        if deliberate_answer:
            self.system_prompt = self.system_prompt + DELIBERATE_ANSWER_PROTOCOL
        self.provider = ModelProvider(self.settings)

        # Subsystems.
        self.hooks = HookManager()
        self.skills = SkillStore(self.settings.skills_dir)
        self.subagents = SubAgentRegistry(
            self.provider,
            max_depth=self.settings.max_subagent_depth,
            tool_calls_limit=self.settings.tool_calls_limit,
            request_limit=self.settings.request_limit,
        )
        self.compactor = Compactor(
            self.provider,
            after_messages=self.settings.compact_after_messages,
            keep_recent=self.settings.compact_keep_recent,
            summarizer_prompt=summarizer_prompt,
        )

        # Conversation memory.
        self.history: list[ModelMessage] = []
        self.last_compaction_summary: str = ""

        # Tools registered on the agent (hook-wrapped).
        self._tools: list[Callable] = []
        self._register_builtin_tools()
        for fn in extra_tools or []:
            self._tools.append(self._wrap_tool(fn))

        # Sub-agents from config/caller.
        for spec in subagents or []:
            self.subagents.register(spec)

        # MCP toolsets.
        self._mcp_toolsets = build_mcp_toolsets(mcp_servers or [])

        self.agent = self._build_agent()

    # -------------------------------------------------------------- config build
    @classmethod
    def from_config(cls, path: str, *, settings: Settings | None = None) -> "Multivac":
        """Build a fully configured harness from a YAML bundle (file or folder).

        Everything — provider, prompts, tools, hooks, skills, sub-agents, MCP servers,
        compaction and limits — comes from the bundle. No base-harness edits required.
        """
        from .build import build_from_config  # local import avoids a cycle

        return build_from_config(cls, path, settings=settings)

    @property
    def mode(self) -> str:
        return self.provider.mode

    # ------------------------------------------------------------------ build
    def _build_agent(self) -> Agent:
        # Wrap every tool with a uniform strict=False. Some providers (e.g. Cerebras)
        # reject a batch of tools with mixed `strict` values, which pydantic-ai would
        # otherwise infer per-tool.
        tools = [Tool(fn, strict=False) for fn in self._tools]
        return Agent(
            self.provider.model(),
            system_prompt=self.system_prompt,
            tools=tools,
            toolsets=self._mcp_toolsets or None,
            name="multivac",
        )

    def _wrap_tool(self, fn: Callable) -> Callable:
        """Wrap a tool so PRE_TOOL / POST_TOOL hooks run around it."""
        import functools

        name = fn.__name__

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            ctx = HookContext(
                event=HookEvent.PRE_TOOL, tool_name=name, tool_args=dict(kwargs)
            )
            pre = self.hooks.dispatch(ctx)
            if pre.decision is HookDecision.BLOCK:
                return f"[blocked by hook: {pre.reason}]"
            if pre.decision is HookDecision.MODIFY and isinstance(pre.payload, dict):
                kwargs = pre.payload

            # Lesson from Hermes (commits d386bdb / c60a2c2 / f6c6c03): never let a tool
            # exception crash the whole agent run. Return the error as execution feedback
            # so the model can see what went wrong and retry with corrected arguments.
            try:
                result = fn(*args, **kwargs)
            except Exception as e:  # noqa: BLE001 - deliberate: feed error back to model
                feedback = (
                    f"[tool error in {name}: {type(e).__name__}: {e}. "
                    f"Re-check the arguments and call again.]"
                )
                err_ctx = HookContext(
                    event=HookEvent.POST_TOOL,
                    tool_name=name,
                    tool_args=dict(kwargs),
                    tool_result=feedback,
                    metadata={"error": True},
                )
                self.hooks.dispatch(err_ctx)
                return feedback

            post_ctx = HookContext(
                event=HookEvent.POST_TOOL,
                tool_name=name,
                tool_args=dict(kwargs),
                tool_result=result,
            )
            post = self.hooks.dispatch(post_ctx)
            if post.decision is HookDecision.MODIFY:
                return post.payload
            return result

        return wrapper

    def _register_builtin_tools(self) -> None:
        # Skill read/write tools.
        for tool in self.skills.as_tools():
            self._tools.append(self._wrap_tool(tool))

        # Sub-agent delegation tool.
        def spawn_subagent(name: str, task: str) -> str:
            """Delegate a bounded task to a registered sub-agent by name."""
            try:
                return self.subagents.run(name, task)
            except KeyError as e:
                return f"[error] {e}. Available: {self.subagents.describe()}"

        def list_subagents() -> str:
            """List the registered sub-agents available to delegate to."""
            return self.subagents.describe()

        self._tools.append(self._wrap_tool(spawn_subagent))
        self._tools.append(self._wrap_tool(list_subagents))

    # ------------------------------------------------------ public extension API
    def add_tool(self, fn: Callable) -> None:
        """Register an extra local tool (hook-wrapped). Rebuilds the agent."""
        self._tools.append(self._wrap_tool(fn))
        self.agent = self._build_agent()

    def add_subagent(self, spec: SubAgentSpec) -> None:
        self.subagents.register(spec)

    def on(
        self,
        events: list[HookEvent],
        fn,
        *,
        name: str | None = None,
        priority: int = 100,
    ):
        """Register a hook for one or more lifecycle events."""
        return self.hooks.register(
            name or getattr(fn, "__name__", "hook"), events, fn, priority=priority
        )

    # ------------------------------------------------------------- conversation
    @staticmethod
    def _coerce_image(img: bytes | str | Path) -> BinaryContent:
        """Turn an image (raw PNG bytes, or a path to a .png/.jpg) into BinaryContent.

        Cerebras Gemma 4 only accepts base64 PNG/JPEG data (not external URLs), so we
        always send bytes via BinaryContent. Pydantic AI handles the base64 framing.
        """
        if isinstance(img, BinaryContent):  # already wrapped
            return img
        if isinstance(img, bytes):
            data, media = img, "image/png"
        else:
            p = Path(img)
            data = p.read_bytes()
            media = "image/jpeg" if p.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
        return BinaryContent(data=data, media_type=media)

    def chat(
        self, message: str, *, images: list[bytes | str | Path] | None = None
    ) -> TurnResult:
        """One conversational turn with full lifecycle: hooks + compaction.

        Optionally pass `images` (raw PNG/JPEG bytes or file paths). Per Gemma 4
        modality-order guidance, each image is placed BEFORE the text in the prompt.
        """
        # PRE_RUN hook (may block or rewrite the user message).
        pre_ctx = HookContext(event=HookEvent.PRE_RUN, user_message=message)
        pre = self.hooks.dispatch(pre_ctx)
        if pre.decision is HookDecision.BLOCK:
            return TurnResult(output="", blocked=True, block_reason=pre.reason)
        if pre.decision is HookDecision.MODIFY and isinstance(pre.payload, str):
            message = pre.payload

        # Auto-compact before sending if needed.
        compacted = self.maybe_compact()

        # Build the prompt: images FIRST (Gemma 4 guidance), then the text.
        prompt: str | list = message
        if images:
            prompt = [self._coerce_image(i) for i in images] + [message]

        # Circuit breaker (Hermes lesson): cap tool calls / requests per turn so a
        # runaway self-recursive tool loop can't burn the budget or hang.
        limits = UsageLimits(
            tool_calls_limit=self.settings.tool_calls_limit,
            request_limit=self.settings.request_limit,
        )
        try:
            result = self.agent.run_sync(
                prompt, message_history=self.history, usage_limits=limits
            )
        except UsageLimitExceeded as e:
            return TurnResult(
                output=f"[stopped: tool/usage limit reached — {e}]",
                blocked=True,
                block_reason="usage_limit",
                compacted=compacted,
            )
        self.history = result.all_messages()
        output = str(result.output)

        # POST_RUN hook (may rewrite the assistant message).
        post_ctx = HookContext(event=HookEvent.POST_RUN, assistant_message=output)
        post = self.hooks.dispatch(post_ctx)
        if post.decision is HookDecision.MODIFY and isinstance(post.payload, str):
            output = post.payload

        return TurnResult(output=output, compacted=compacted)

    def reset(self) -> None:
        self.history = []
        self.last_compaction_summary = ""

    # --------------------------------------------------------------- compaction
    def maybe_compact(self) -> bool:
        """Compact if over threshold. Returns True if compaction happened."""
        if not self.compactor.should_compact(self.history):
            return False
        return self.compact()

    def compact(self) -> bool:
        """Force compaction now. Fires PRE_COMPACT / POST_COMPACT hooks."""
        before = len(self.history)
        self.hooks.dispatch(
            HookContext(event=HookEvent.PRE_COMPACT, metadata={"messages": before})
        )
        new_history, summary = self.compactor.compact(self.history)
        if not summary:
            return False
        self.history = new_history
        self.last_compaction_summary = summary
        self.hooks.dispatch(
            HookContext(
                event=HookEvent.POST_COMPACT,
                metadata={"before": before, "after": len(new_history)},
            )
        )
        return True
