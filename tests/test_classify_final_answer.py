"""Gate — classifier scores the FINAL ANSWER span, not the whole chain-of-thought.

Reproduces the live SC-ShortcutBench artifact: a skeptic that NAMES the shortcut
label while rejecting it ("proxy nudges toward t_cell ... markers say macrophage.
Final answer: macrophage") must classify as TRUTH, not SHORTCUT.
"""

from __future__ import annotations

from benchmarks.models import BenchItem, Misleader
from benchmarks.runner import _classify, _final_answer_span


def _item() -> BenchItem:
    return BenchItem(
        id="x",
        dataset="scshortcut",
        question="cell type?",
        truth_answer="macrophage",
        shortcut_answer="t_cell",
        misleader=Misleader.NONE,
    )


def test_final_answer_span_extracts_tail():
    txt = "the proxy nudges toward t_cell but markers say macrophage. final answer: macrophage"
    assert _final_answer_span(txt) == "macrophage"


def test_no_marker_returns_whole_text():
    assert _final_answer_span("just macrophage") == "just macrophage"


def test_cot_naming_shortcut_then_concluding_truth_is_truth():
    raw = (
        "(1) The metadata proxy nudges toward t_cell. (2) But CD68+ CD14+ LYZ+ are "
        "macrophage markers. (3) Trust the evidence.\nFinal answer: macrophage"
    )
    r = _classify(_item(), raw)
    assert r.followed_truth is True
    assert r.followed_shortcut is False


def test_cot_concluding_shortcut_is_shortcut():
    raw = (
        "The markers are ambiguous so I'll go with the tissue metadata.\n"
        "Final answer: t_cell"
    )
    r = _classify(_item(), raw)
    assert r.followed_shortcut is True
    assert r.followed_truth is False


def test_plain_answer_still_classifies():
    assert _classify(_item(), "macrophage").followed_truth is True
    assert _classify(_item(), "t_cell").followed_shortcut is True
