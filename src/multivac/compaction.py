"""Conversation compaction.

When a conversation grows past a threshold, compaction summarizes the *older* messages
into a single compact note and keeps the most recent messages verbatim. This bounds
context growth while preserving continuity.

* Trigger: `should_compact()` (count-based by default; pluggable).
* Strategy: summarize the head of history with the model, keep the tail intact.
* Hook: the harness fires PRE_COMPACT / POST_COMPACT around this.

In OFFLINE mode the summary is produced deterministically (no model call) so the
mechanism is testable without a network.
"""

from __future__ import annotations

from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    UserPromptPart,
)

from .provider import ModelProvider

SUMMARIZER_PROMPT = (
    "You compress conversation history. Given a transcript, produce a concise, faithful "
    "summary that preserves decisions, facts, open questions, user preferences and any "
    "state the assistant must remember. Output plain prose, no preamble."
)


def message_text(msg: ModelMessage) -> str:
    """Best-effort readable text for a single message."""
    role = "user" if isinstance(msg, ModelRequest) else "assistant"
    chunks: list[str] = []
    for part in msg.parts:
        kind = getattr(part, "part_kind", "")
        content = getattr(part, "content", None)
        if content is None:
            continue
        if kind == "system-prompt":
            chunks.append(f"[system] {content}")
        elif isinstance(content, str):
            chunks.append(content)
        else:
            chunks.append(str(content))
    return f"{role}: " + " ".join(chunks) if chunks else ""


def transcript(messages: list[ModelMessage]) -> str:
    lines = [message_text(m) for m in messages]
    return "\n".join(line for line in lines if line)


class Compactor:
    """Summarize the head of a conversation, keep the tail."""

    def __init__(
        self,
        provider: ModelProvider,
        *,
        after_messages: int = 40,
        keep_recent: int = 8,
        summarizer_prompt: str | None = None,
    ) -> None:
        self.provider = provider
        self.after_messages = after_messages
        self.keep_recent = keep_recent
        self._summarizer = Agent(
            provider.model(),
            system_prompt=summarizer_prompt or SUMMARIZER_PROMPT,
            name="compactor",
        )

    def should_compact(self, messages: list[ModelMessage]) -> bool:
        return len(messages) > self.after_messages

    def _summarize(self, messages: list[ModelMessage]) -> str:
        text = transcript(messages)
        if not self.provider.is_live:
            # Deterministic offline summary: bounded digest, no model call.
            preview = text[:500]
            return (
                f"[compacted summary of {len(messages)} earlier messages]\n{preview}"
                + ("..." if len(text) > 500 else "")
            )
        result = self._summarizer.run_sync(
            f"Summarize this conversation so it can be safely dropped:\n\n{text}"
        )
        return f"[summary of {len(messages)} earlier messages]\n{result.output}"

    def compact(self, messages: list[ModelMessage]) -> tuple[list[ModelMessage], str]:
        """Return (new_history, summary_text).

        new_history = one synthetic request carrying the summary + the kept tail.
        If there is nothing to compact, returns the input unchanged with "".
        """
        if len(messages) <= self.keep_recent:
            return messages, ""
        head = messages[: -self.keep_recent]
        tail = messages[-self.keep_recent :]
        summary = self._summarize(head)
        summary_msg = ModelRequest(
            parts=[SystemPromptPart(content=f"Earlier conversation summary:\n{summary}")]
        )
        return [summary_msg, *tail], summary
