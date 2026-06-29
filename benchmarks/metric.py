"""Shortcut-reliance metric — the Truth-Shortcut Margin (TSM).

The non-biotech analogue of SC-ShortcutBench's headline number, computed over the
CONFLICT subset only (items where the chart misleader and the underlying data
disagree). On conflict items every answer is either truth-aligned, shortcut-aligned,
a refusal, or other.

    TSM = truth_accuracy - shortcut_rate

Interpretation:
  TSM ~ +1  : the pipeline almost always follows the data (ideal).
  TSM ~  0  : coin-flip between data and misleader.
  TSM ~ -1  : the pipeline almost always follows the misleading chart (failure).

Published MLLM baselines on misleading charts collapse to ~random, i.e. strongly
negative TSM; the project goal is to move TSM from negative toward ~0+ via the
verify-before-conclude (redraw / table-reread) loop.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from .models import BenchResult, ShortcutMetric


class _HasModalFraction(Protocol):
    modal_fraction: float


def self_consistency(cells: Sequence[_HasModalFraction]) -> float:
    """Stochastic floor: mean modal-stance fraction across (item, framing) cells.

    Each cell ran one framing `n_executions` times; `modal_fraction` is how often
    the dominant stance won within that cell. self_consistency = mean over all
    cells. 1.0 means the model was perfectly stable run-to-run for every framing
    (so any cross-framing variance is genuinely framing-induced, not noise).

    The interpretation gate (EMNLP P/¬P logic): trust an FIV signal only when
    FIV > (1 - self_consistency).
    """
    cells = list(cells)
    if not cells:
        return 0.0
    return sum(c.modal_fraction for c in cells) / len(cells)


def truth_shortcut_margin(results: Sequence[BenchResult]) -> float:
    """TSM over the conflict subset: truth_accuracy - shortcut_rate.

    Returns 0.0 when there are no conflict items (no signal either way).
    """
    conflict = [r for r in results if r.is_conflict]
    if not conflict:
        return 0.0
    n = len(conflict)
    truth_acc = sum(1 for r in conflict if r.followed_truth) / n
    shortcut_rate = sum(1 for r in conflict if r.followed_shortcut) / n
    return truth_acc - shortcut_rate


def shortcut_metric(
    results: Sequence[BenchResult], *, pipeline: str = "baseline"
) -> ShortcutMetric:
    """Full scorecard for a set of results from one pipeline."""
    conflict = [r for r in results if r.is_conflict]
    n_conflict = len(conflict)

    if n_conflict == 0:
        return ShortcutMetric(
            n_total=len(results),
            n_conflict=0,
            truth_accuracy=0.0,
            shortcut_rate=0.0,
            refusal_rate=0.0,
            truth_shortcut_margin=0.0,
            pipeline=pipeline,
        )

    truth_acc = sum(1 for r in conflict if r.followed_truth) / n_conflict
    shortcut_rate = sum(1 for r in conflict if r.followed_shortcut) / n_conflict
    refusal_rate = sum(1 for r in conflict if r.refused) / n_conflict

    return ShortcutMetric(
        n_total=len(results),
        n_conflict=n_conflict,
        truth_accuracy=truth_acc,
        shortcut_rate=shortcut_rate,
        refusal_rate=refusal_rate,
        truth_shortcut_margin=truth_acc - shortcut_rate,
        pipeline=pipeline,
    )
