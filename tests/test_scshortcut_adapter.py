"""E2 GATE — SC-ShortcutBench adapter + TSM separation (offline, deterministic).

Proves the conflict-row schema reproduces the published direction:
  * a METADATA-shortcut stub follows the proxy -> strongly NEGATIVE TSM,
  * an EXPRESSION-grounded stub follows the markers -> POSITIVE TSM,
mirroring SC-ShortcutBench (foundation models score TSM ~-24 to -40pp; a verify-
on-expression skeptic moves TSM toward 0/positive).
"""

from __future__ import annotations

from benchmarks.adapters.scshortcut import load_scshortcut
from benchmarks.metric import shortcut_metric
from benchmarks.models import BenchResult
from benchmarks.runner import _classify


def _result(item, raw, pipeline):
    c = _classify(item, raw)
    return BenchResult(
        item_id=item.id,
        dataset=item.dataset,
        is_conflict=item.is_conflict,
        pipeline=pipeline,
        model_answer=raw,
        followed_truth=c.followed_truth,
        followed_shortcut=c.followed_shortcut,
        refused=c.refused,
    )


def test_adapter_loads_with_conflict_rows():
    items = load_scshortcut()
    assert len(items) >= 7
    conflict = [i for i in items if i.is_conflict]
    assert len(conflict) >= 6  # most rows are conflict rows
    # expression evidence is carried in the table (truth ground)
    assert all(it.table and "expression_markers" in it.table[0] for it in items)
    # no charts in this tabular benchmark
    assert all(it.chart_path is None for it in items)


def test_metadata_shortcut_stub_has_negative_tsm():
    items = load_scshortcut()
    # follows the metadata proxy -> answers the shortcut label on conflict rows
    results = [_result(it, it.shortcut_answer, "baseline") for it in items]
    m = shortcut_metric(results, pipeline="baseline")
    assert m.truth_shortcut_margin < 0  # strongly negative, like the paper
    assert m.shortcut_rate > m.truth_accuracy


def test_expression_grounded_stub_has_positive_tsm():
    items = load_scshortcut()
    results = [_result(it, it.truth_answer, "skeptic") for it in items]
    m = shortcut_metric(results, pipeline="skeptic")
    assert m.truth_shortcut_margin > 0
    assert m.truth_accuracy == 1.0


def test_skeptic_beats_baseline_tsm_delta():
    items = load_scshortcut()
    base = [_result(it, it.shortcut_answer, "baseline") for it in items]
    skep = [_result(it, it.truth_answer, "skeptic") for it in items]
    mb = shortcut_metric(base, pipeline="baseline")
    ms = shortcut_metric(skep, pipeline="skeptic")
    assert ms.truth_shortcut_margin - mb.truth_shortcut_margin >= 1.0
