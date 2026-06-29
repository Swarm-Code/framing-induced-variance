"""Bundle-local tools for the example agent.

Referenced from multivac.yaml as ``tools:word_count`` etc. These live in the bundle, not
in the base harness — that's the whole point: benchmark a new agent without editing core.
"""

from __future__ import annotations


def word_count(text: str) -> int:
    """Count the words in a piece of text."""
    return len(text.split())


def reverse_text(text: str) -> str:
    """Reverse a string."""
    return text[::-1]
