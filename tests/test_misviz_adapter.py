"""Misviz adapter + runner on the committed offline slice (no network/model)."""

from __future__ import annotations

from benchmarks.adapters.misviz import load_misviz
from benchmarks.models import Misleader
from benchmarks.runner import run_pipeline


def test_slice_loads_and_has_conflict_items():
    items = load_misviz()
    assert len(items) >= 8
    conflict = [it for it in items if it.is_conflict]
    aligned = [it for it in items if not it.is_conflict]
    assert len(conflict) >= 6, "slice must exercise the conflict subset"
    assert len(aligned) >= 1, "slice should include an honest control"
    # Misleader taxonomy is populated on conflict items.
    assert all(it.misleader != Misleader.NONE for it in conflict)


def test_baseline_stub_follows_shortcut_negative_tsm():
    items = load_misviz()
    # A model that trusts the misleading chart -> answers the shortcut.
    _, metric = run_pipeline(
        items, ask=lambda it: it.shortcut_answer, pipeline="baseline"
    )
    assert metric.n_conflict >= 6
    assert metric.truth_shortcut_margin < 0, "chart-trusting model -> negative TSM"
    assert metric.shortcut_rate == 1.0


def test_skeptic_stub_follows_truth_positive_tsm():
    items = load_misviz()
    # A skeptic that re-reads the table -> answers the truth.
    _, metric = run_pipeline(items, ask=lambda it: it.truth_answer, pipeline="skeptic")
    assert metric.truth_shortcut_margin == 1.0
    assert metric.truth_accuracy == 1.0


def test_results_written_to_jsonl(tmp_path):
    items = load_misviz()
    out = tmp_path / "results.jsonl"
    results, _ = run_pipeline(
        items, ask=lambda it: it.truth_answer, results_path=out
    )
    assert out.exists()
    assert out.read_text().count("\n") == len(results)
