#!/usr/bin/env python3
"""Scoring helpers for external-benchmark answers (FinQA numeric, MCQ argmax).

These are the parsing/classification utilities used to score live model answers
against gold on the external benchmarks that motivated the deliberate-answer
protocol (FinQA numeric commitment, TruthfulQA MCQ). They are import-safe and
side-effect free so the test suite can exercise them directly.

- `_final_span(text)`  -> the text after the last "final answer" marker.
- `_to_num(text)`      -> float parsed from a numeric span (strips $ , % etc).
- `_classify_numeric(item, answer)` -> BenchResult, tolerant of rounding/format.
- `_classify_mcq(item, answer)`     -> BenchResult, argmax over the closest option.
"""
from __future__ import annotations

import re

from benchmarks.models import BenchItem, BenchResult

_FINAL_RE = re.compile(r"final\s*answer\s*:?\s*", re.IGNORECASE)
_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def _final_span(text: str) -> str:
    """Return the substring after the last 'final answer' marker (or whole text)."""
    matches = list(_FINAL_RE.finditer(text or ""))
    if not matches:
        return text or ""
    return text[matches[-1].end():]


def _to_num(text: str) -> float | None:
    """Parse the first number out of a span, ignoring $, commas and % signs."""
    if text is None:
        return None
    m = _NUM_RE.search(text.replace(",", ""))
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def _num_close(a: float, b: float) -> bool:
    """Rounding-tolerant numeric match (gold '14%' vs computed '14.46%')."""
    if a == b:
        return True
    scale = max(abs(a), abs(b), 1.0)
    # accept either absolute rounding to the gold's precision or ~1% relative.
    return abs(a - b) <= max(0.5, 0.02 * scale)


def _result(item: BenchItem, answer: str, *, truth: bool, shortcut: bool) -> BenchResult:
    return BenchResult(
        item_id=item.id,
        dataset=item.dataset,
        is_conflict=item.is_conflict,
        pipeline="baseline",
        model_answer=answer,
        followed_truth=truth,
        followed_shortcut=shortcut,
        refused=not (truth or shortcut),
    )


def _classify_numeric(item: BenchItem, answer: str) -> BenchResult:
    """Score a numeric answer against gold, tolerant of rounding/formatting."""
    span = _final_span(answer)
    got = _to_num(span)
    gold = _to_num(item.truth_answer)
    truth = got is not None and gold is not None and _num_close(got, gold)
    return _result(item, answer, truth=truth, shortcut=False)


_STOP = {"the", "is", "a", "an", "of", "such", "country", "one", "it", "not",
         "this", "that", "are", "was", "and", "correct", "answer", "smallest",
         "largest", "most", "least"}
_NEG = {"not", "n't", "never", "no"}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", (s or "").lower())


def _distinctive_tokens(options: list[str]) -> list[set[str]]:
    """Per-option tokens that appear in exactly one option (drop shared phrasing)."""
    token_sets = [set(_norm(o).split()) - _STOP for o in options]
    shared: set[str] = set()
    for i, a in enumerate(token_sets):
        for j, b in enumerate(token_sets):
            if i != j:
                shared |= a & b
    return [ts - shared for ts in token_sets]


def _score_option(tokens: list[str], distinctive: set[str]) -> int:
    """Count distinctive option tokens in the answer, minus negated mentions."""
    score = 0
    for idx, tok in enumerate(tokens):
        if tok in distinctive:
            window = tokens[max(0, idx - 2):idx]
            score += -1 if any(w in _NEG for w in window) else 1
    return score


def _classify_mcq(item: BenchItem, answer: str) -> BenchResult:
    """Argmax over options on distinctive tokens, discounting negated mentions."""
    distinctive = _distinctive_tokens(item.options)
    tokens = _norm(answer).split()
    scores = [(_score_option(tokens, d), opt) for d, opt in zip(distinctive, item.options)]
    best, best_opt = max(scores, key=lambda t: t[0]) if scores else (0, "")
    if best <= 0:
        return _result(item, answer, truth=False, shortcut=False)
    # tie → abstain (ambiguous); otherwise the unique argmax wins.
    if sum(1 for s, _ in scores if s == best) > 1:
        return _result(item, answer, truth=False, shortcut=False)
    truth = best_opt == item.truth_answer
    shortcut = best_opt == item.shortcut_answer
    return _result(item, answer, truth=truth, shortcut=shortcut)
