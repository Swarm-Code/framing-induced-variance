"""Run a pipeline over a list of `BenchItem`s and score shortcut reliance.

The runner is deliberately decoupled from the model: it takes an `ask` callable
(`BenchItem -> raw answer str`). That lets it run:
  * offline in tests with a deterministic stub,
  * against the live `Multivac` harness via `harness_ask`,
  * for either the baseline (chart/numbers-only) or skeptic (verify-then-conclude)
    pipeline — the difference is entirely in the prompt the `ask` callable builds.

Scoring maps each raw answer onto {followed_truth, followed_shortcut, refused} by
case-insensitive substring match against the item's `truth_answer` /
`shortcut_answer`, with a small refusal-phrase guard.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from pathlib import Path

from .metric import shortcut_metric
from .models import BenchItem, BenchResult, ShortcutMetric

AskFn = Callable[[BenchItem], str]

_REFUSAL_MARKERS = (
    "cannot determine",
    "can't determine",
    "insufficient",
    "not enough information",
    "i refuse",
    "unable to answer",
    "abstain",
    "misleading",  # skeptic explicitly flags the chart instead of answering
)


def _has_token(text: str, token: str) -> bool:
    """Word-boundary match so an answer token isn't matched inside another word.

    Critical: substring matching is unsafe here — e.g. the truth token "no" is a
    substring of "cannot", "none", "no one", so "I cannot determine" would falsely
    read as the answer "no". We require the token to appear as a whole word
    (alnum boundaries), which also tolerates surrounding punctuation.
    """
    if not token:
        return False
    return re.search(rf"(?<![0-9a-z]){re.escape(token)}(?![0-9a-z])", text) is not None


def _final_answer_span(text: str) -> str:
    """Return the conclusion span when the model used a 'final answer' delimiter.

    Chain-of-thought skeptics often NAME the shortcut label while explaining why
    they reject it ("the metadata proxy nudges toward t_cell ... but the markers
    say macrophage"). Classifying the whole CoT then double-counts the shortcut.
    When a final-answer marker is present we score only the text AFTER the LAST
    marker (the actual conclusion). If no marker is present, score the whole text.
    """
    markers = (
        "final answer:",
        "final answer (single best label):",
        "final answer",
        "answer:",
        "conclusion:",
    )
    best_idx = -1
    best_end = 0
    for m in markers:
        idx = text.rfind(m)
        if idx > best_idx:
            best_idx = idx
            best_end = idx + len(m)
    if best_idx == -1:
        return text
    tail = text[best_end:].strip()
    # If the tail is empty (marker at very end), fall back to the whole text.
    return tail or text


def _classify(item: BenchItem, raw: str) -> BenchResult:
    full = (raw or "").strip().lower()
    text = _final_answer_span(full)
    truth = item.truth_answer.strip().lower()
    shortcut = item.shortcut_answer.strip().lower()

    refused = any(m in text for m in _REFUSAL_MARKERS) and not _has_token(text, truth)
    followed_truth = _has_token(text, truth) and not refused
    # Only count a shortcut hit when it is NOT also the truth (conflict items differ).
    followed_shortcut = (
        _has_token(text, shortcut)
        and shortcut != truth
        and not followed_truth
        and not refused
    )

    return BenchResult(
        item_id=item.id,
        dataset=item.dataset,
        is_conflict=item.is_conflict,
        model_answer=raw,
        followed_truth=followed_truth,
        followed_shortcut=followed_shortcut,
        refused=refused,
    )


def run_pipeline(
    items: Sequence[BenchItem],
    ask: AskFn,
    *,
    pipeline: str = "baseline",
    results_path: str | Path | None = None,
) -> tuple[list[BenchResult], ShortcutMetric]:
    """Run `ask` over every item, classify, optionally write results.jsonl, score."""
    results: list[BenchResult] = []
    for item in items:
        raw = ask(item)
        res = _classify(item, raw)
        res.pipeline = pipeline
        results.append(res)

    if results_path is not None:
        path = Path(results_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as fh:
            for r in results:
                fh.write(r.model_dump_json() + "\n")

    return results, shortcut_metric(results, pipeline=pipeline)


def harness_ask(harness, *, skeptic: bool = False) -> AskFn:
    """Build an `ask` callable backed by the live `Multivac` harness.

    Baseline prompt shows the chart/options and asks for a direct answer.
    Skeptic prompt additionally hands over the underlying table and instructs the
    model to re-read the data and distrust the chart's framing (the published
    redraw / table-reread fix).
    """

    def ask(item: BenchItem) -> str:
        opts = f" Options: {', '.join(item.options)}." if item.options else ""
        if skeptic:
            prompt = (
                f"Question: {item.question}{opts}\n"
                f"The chart may be misleading. Here is the underlying data table:\n"
                f"{json.dumps(item.table)}\n"
                "Answer ONLY from the data. If the chart's visual impression "
                "contradicts the table, trust the table and say so."
            )
        else:
            prompt = (
                f"Question: {item.question}{opts}\n"
                f"Chart: {item.chart_path or '(see attached)'}\n"
                "Give your best answer based on the chart."
            )
        harness.reset()
        return harness.chat(prompt).output

    return ask
