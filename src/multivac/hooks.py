"""Hooks subsystem.

A *hook* observes — and can influence — the harness lifecycle. Hooks register for one
or more `HookEvent`s and return a `HookResult` that can CONTINUE, MODIFY (replace the
payload), or BLOCK (abort the action). Hooks are plain callables, so they are trivial
to write and test; the `HookManager` dispatches events to the matching hooks in order.

Typical uses:
* PRE_TOOL  — deny dangerous tool calls (a guardrail), redact arguments.
* POST_TOOL — scrub/limit tool output.
* PRE_RUN   — inject context / enforce policy on a user turn.
* POST_RUN  — log, score, or post-process the assistant's answer.
* PRE_COMPACT / POST_COMPACT — observe the compaction boundary.
"""

from __future__ import annotations

from typing import Callable

from .types import HookContext, HookDecision, HookEvent, HookResult

HookFn = Callable[[HookContext], HookResult | None]


class Hook:
    """A named hook bound to one or more events."""

    def __init__(
        self,
        name: str,
        events: list[HookEvent],
        fn: HookFn,
        *,
        priority: int = 100,
    ) -> None:
        self.name = name
        self.events = set(events)
        self.fn = fn
        self.priority = priority  # lower runs earlier

    def matches(self, event: HookEvent) -> bool:
        return event in self.events


class HookManager:
    """Registers hooks and dispatches lifecycle events to them."""

    def __init__(self) -> None:
        self._hooks: list[Hook] = []

    def register(
        self,
        name: str,
        events: list[HookEvent],
        fn: HookFn,
        *,
        priority: int = 100,
    ) -> Hook:
        hook = Hook(name, events, fn, priority=priority)
        self._hooks.append(hook)
        self._hooks.sort(key=lambda h: h.priority)
        return hook

    def unregister(self, name: str) -> None:
        self._hooks = [h for h in self._hooks if h.name != name]

    def hooks_for(self, event: HookEvent) -> list[Hook]:
        return [h for h in self._hooks if h.matches(event)]

    def dispatch(self, ctx: HookContext) -> HookResult:
        """Run every hook for ctx.event in priority order.

        Resolution: the first BLOCK wins (and stops the chain). MODIFY results chain —
        each MODIFY updates the relevant field of the context so later hooks see the
        replacement. If no hook blocks, the aggregated result is returned (MODIFY if any
        hook modified, else CONTINUE).
        """
        final = HookResult(decision=HookDecision.CONTINUE)
        modified = False
        for hook in self.hooks_for(ctx.event):
            result = hook.fn(ctx)
            if result is None:
                continue
            if result.decision is HookDecision.BLOCK:
                result.reason = result.reason or f"blocked by hook {hook.name}"
                return result
            if result.decision is HookDecision.MODIFY:
                modified = True
                final = result
                # Feed the replacement forward so chained hooks see it.
                if ctx.event in (HookEvent.PRE_TOOL,) and isinstance(result.payload, dict):
                    ctx.tool_args = result.payload
                elif ctx.event in (HookEvent.POST_TOOL,):
                    ctx.tool_result = result.payload
                elif ctx.event in (HookEvent.PRE_RUN,) and isinstance(result.payload, str):
                    ctx.user_message = result.payload
                elif ctx.event in (HookEvent.POST_RUN,) and isinstance(result.payload, str):
                    ctx.assistant_message = result.payload
        if modified:
            return final
        return HookResult(decision=HookDecision.CONTINUE)
