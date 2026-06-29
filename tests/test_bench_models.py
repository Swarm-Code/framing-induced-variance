"""BenchItem / conflict-flag behavior."""

from __future__ import annotations

from benchmarks.models import BenchItem, Misleader


def test_conflict_item_flags_divergent_answers():
    item = BenchItem(
        id="c1",
        question="Big difference?",
        truth_answer="Small",
        shortcut_answer="Large",
        misleader=Misleader.TRUNCATED_AXIS,
    )
    assert item.is_conflict is True


def test_aligned_item_is_not_conflict():
    item = BenchItem(
        id="a1",
        question="Did it grow?",
        truth_answer="Yes",
        shortcut_answer="Yes",
        misleader=Misleader.NONE,
    )
    assert item.is_conflict is False
