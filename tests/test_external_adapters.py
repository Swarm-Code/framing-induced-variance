"""Gate — external-benchmark adapters load REAL items with dataset gold preserved.

We assert structure only (not authored labels): the truth answer is the dataset's
own gold, options come from the dataset, and counts match the fetched slices.
"""

from __future__ import annotations

import pytest

from benchmarks.adapters.external import load_convfinqa, load_finqa, load_truthfulqa


def test_truthfulqa_loads_with_gold_labels():
    items = load_truthfulqa()
    assert len(items) >= 50
    for it in items[:20]:
        # truth must be one of the dataset's own choices, distinct from the shortcut
        assert it.truth_answer in it.options
        assert it.shortcut_answer in it.options
        assert it.truth_answer != it.shortcut_answer  # real distractor
        assert it.is_conflict  # MCQ truth vs labelled-false distractor


def test_finqa_loads_numeric_gold():
    items = load_finqa()
    assert len(items) >= 40
    for it in items[:10]:
        assert it.truth_answer  # dataset gold present
        assert it.table and ("pre_text" in it.table[0] or "post_text" in it.table[0])
        # numeric set: aligned (no labelled distractor)
        assert it.shortcut_answer == it.truth_answer


def test_convfinqa_loads_gold():
    items = load_convfinqa()
    assert len(items) >= 30
    for it in items[:10]:
        assert it.truth_answer
        assert it.question


def test_no_authored_labels_marker():
    """Sanity: adapters carry the dataset name, not a synthetic tag."""
    assert {i.dataset for i in load_truthfulqa()} == {"truthfulqa"}
    assert {i.dataset for i in load_finqa()} == {"finqa"}
