"""Truth-Shortcut-Margin metric math."""

from __future__ import annotations

from benchmarks.metric import shortcut_metric, truth_shortcut_margin
from benchmarks.models import BenchResult


def _r(item_id, *, conflict, truth, shortcut, refused=False):
    return BenchResult(
        item_id=item_id,
        is_conflict=conflict,
        model_answer="x",
        followed_truth=truth,
        followed_shortcut=shortcut,
        refused=refused,
    )


def test_all_shortcut_gives_margin_minus_one():
    results = [_r(f"i{i}", conflict=True, truth=False, shortcut=True) for i in range(5)]
    assert truth_shortcut_margin(results) == -1.0


def test_all_truth_gives_margin_plus_one():
    results = [_r(f"i{i}", conflict=True, truth=True, shortcut=False) for i in range(5)]
    assert truth_shortcut_margin(results) == 1.0


def test_no_conflict_items_returns_zero():
    results = [_r("i0", conflict=False, truth=True, shortcut=False)]
    assert truth_shortcut_margin(results) == 0.0


def test_scorecard_counts_and_rates():
    results = [
        _r("c1", conflict=True, truth=True, shortcut=False),
        _r("c2", conflict=True, truth=False, shortcut=True),
        _r("c3", conflict=True, truth=False, shortcut=False, refused=True),
        _r("a1", conflict=False, truth=True, shortcut=False),
    ]
    m = shortcut_metric(results, pipeline="baseline")
    assert m.n_total == 4
    assert m.n_conflict == 3
    assert abs(m.truth_accuracy - 1 / 3) < 1e-9
    assert abs(m.shortcut_rate - 1 / 3) < 1e-9
    assert abs(m.refusal_rate - 1 / 3) < 1e-9
    assert abs(m.truth_shortcut_margin - 0.0) < 1e-9
    assert m.pipeline == "baseline"
