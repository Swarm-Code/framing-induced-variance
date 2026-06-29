"""Core shared types for the harness. Pure data — no logic, no I/O."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class HookEvent(str, Enum):
    """Lifecycle points where hooks fire."""

    PRE_RUN = "pre_run"  # before a user turn is sent to the model
    POST_RUN = "post_run"  # after the model produced a final response
    PRE_TOOL = "pre_tool"  # before a tool/function call executes
    POST_TOOL = "post_tool"  # after a tool/function call returns
    PRE_COMPACT = "pre_compact"  # before conversation compaction runs
    POST_COMPACT = "post_compact"  # after conversation compaction runs


class HookDecision(str, Enum):
    CONTINUE = "continue"  # proceed normally
    MODIFY = "modify"  # proceed using HookResult.payload as replacement
    BLOCK = "block"  # abort this action (tool call / run)


class HookResult(BaseModel):
    """Returned by a hook to influence the harness."""

    decision: HookDecision = HookDecision.CONTINUE
    payload: Any = None  # replacement value when decision == MODIFY
    reason: str = ""  # human-readable explanation (esp. for BLOCK)


class HookContext(BaseModel):
    """Everything a hook can see about the current event."""

    model_config = {"arbitrary_types_allowed": True}

    event: HookEvent
    tool_name: str | None = None
    tool_args: dict[str, Any] = Field(default_factory=dict)
    tool_result: Any = None
    user_message: str | None = None
    assistant_message: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TurnResult(BaseModel):
    """The outcome of one conversational turn."""

    output: str
    blocked: bool = False
    block_reason: str = ""
    compacted: bool = False
    tool_calls: list[str] = Field(default_factory=list)
