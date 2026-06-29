"""Bundle-local hooks for the example agent.

Each hook is ``fn(HookContext) -> HookResult | None``. Referenced from multivac.yaml.
"""

from __future__ import annotations

from multivac.types import HookContext, HookDecision, HookResult


def block_secret_tools(ctx: HookContext) -> HookResult | None:
    """Deny any tool whose name contains 'secret'."""
    if ctx.tool_name and "secret" in ctx.tool_name.lower():
        return HookResult(decision=HookDecision.BLOCK, reason="secret tools disabled")
    return None


def tag_output(ctx: HookContext) -> HookResult | None:
    """Append a provenance tag to the assistant's answer."""
    msg = ctx.assistant_message or ""
    return HookResult(decision=HookDecision.MODIFY, payload=f"{msg}\n\n— example-agent")
