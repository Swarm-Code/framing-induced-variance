"""A0 gate: the classifier must use word-boundary matching, not substring.

The bug: truth_answer "no" is a substring of "cannot", "none", "no one", so a
refusal like "I cannot determine" falsely classified as followed_truth. These
tests pin the corrected behavior so it can never silently regress.
"""

from __future__ import annotations

from benchmarks.models import BenchItem, Misleader
from benchmarks.runner import _classify


def _item() -> BenchItem:
    return BenchItem(
        id="t0",
        question="Is revenue growing",
        truth_answer="no",
        shortcut_answer="yes",
        misleader=Misleader.TRUNCATED_AXIS,
    )


def test_refusal_not_misread_as_truth_no():
    # "cannot" contains "no" — must NOT count as answering the truth "no".
    r = _classify(_item(), "I cannot determine this from the chart.")
    assert r.refused is True
    assert r.followed_truth is False
    assert r.followed_shortcut is False


def test_none_substring_does_not_count_as_no():
    r = _classify(_item(), "There is none of the requested context here; insufficient.")
    assert r.followed_truth is False
    assert r.refused is True


def test_real_truth_answer_still_detected():
    r = _classify(_item(), "Based on the data, no — the trend is flat.")
    assert r.followed_truth is True
    assert r.refused is False
    assert r.followed_shortcut is False


def test_real_shortcut_answer_still_detected():
    r = _classify(_item(), "Yes, revenue is clearly growing.")
    assert r.followed_shortcut is True
    assert r.followed_truth is False


def test_punctuation_around_token_ok():
    r = _classify(_item(), "Answer: no.")
    assert r.followed_truth is True
