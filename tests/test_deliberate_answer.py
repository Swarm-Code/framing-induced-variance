"""Gate — the deliberate-answer protocol (minimal universal fix) + numeric scorer.

Two pieces of the FinQA fix:
  1. Multivac(deliberate_answer=True) appends the answer protocol to the system
     prompt (no tools, ~3 lines) — the universal commit-a-final-answer behavior.
  2. the runner's numeric classifier reads the FINAL-ANSWER span and tolerates
     rounding/format (gold '14%' vs computed '14.46%').
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from multivac import Multivac, Settings  # noqa: E402
from multivac.harness import DELIBERATE_ANSWER_PROTOCOL  # noqa: E402
from run_external import _classify_numeric, _final_span, _to_num  # noqa: E402
from benchmarks.models import BenchItem, Misleader  # noqa: E402


def _settings(tmp_path) -> Settings:
    return Settings(offline=True, skills_dir=str(tmp_path / "s"), sessions_dir=str(tmp_path / "x"))


def test_protocol_appended_when_enabled(tmp_path):
    h = Multivac(_settings(tmp_path), system_prompt="Base.", deliberate_answer=True)
    assert "Final answer:" in h.system_prompt
    assert h.system_prompt.startswith("Base.")
    assert DELIBERATE_ANSWER_PROTOCOL.strip()[:20] in h.system_prompt


def test_protocol_absent_by_default(tmp_path):
    h = Multivac(_settings(tmp_path), system_prompt="Base.")
    assert h.system_prompt == "Base."


def _num_item(gold: str) -> BenchItem:
    return BenchItem(
        id="f", dataset="finqa", question="q", truth_answer=gold,
        shortcut_answer=gold, misleader=Misleader.NONE,
    )


def test_final_span_and_num():
    assert _final_span("blah blah Final answer: 94").strip() == "94"
    assert _to_num("14.46%") == 14.46
    assert _to_num("$1,234.5") == 1234.5


def test_numeric_tolerant_rounding():
    # gold rounded to 14%, model committed 14.46% -> should count as correct
    r = _classify_numeric(_num_item("14%"), "Work... Final answer: 14.46%")
    assert r.followed_truth


def test_numeric_exact_match():
    r = _classify_numeric(_num_item("94"), "calc... Final answer: 94")
    assert r.followed_truth


def test_numeric_wrong_is_wrong():
    r = _classify_numeric(_num_item("94"), "calc... Final answer: 250")
    assert not r.followed_truth
