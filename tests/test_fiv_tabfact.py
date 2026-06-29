"""Gate — FIV-on-TabFact metric (offline deterministic stubs)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from benchmarks.adapters.tabfact import load_tabfact, table_md_for  # noqa: E402
from benchmarks.fiv_tabfact import FRAMINGS, run_fiv, verdict_of  # noqa: E402


def test_adapter_loads_binary_real_tables():
    items = load_tabfact("test", limit=20)
    assert len(items) == 20
    for it in items:
        assert it.truth_answer in ("Entailed", "Refuted")
        assert it.shortcut_answer == ("Refuted" if it.truth_answer == "Entailed" else "Entailed")
        assert it.table  # real parsed rows
    assert "|" in table_md_for(items[0])  # markdown rendered


def test_verdict_parsing():
    assert verdict_of("... Final answer: Entailed") == "Entailed"
    assert verdict_of("blah Final answer: Refuted") == "Refuted"
    assert verdict_of("I think Entailed") == "Entailed"
    assert verdict_of("totally unclear") == "other"


def test_sway_model_high_fiv():
    items = load_tabfact("test", limit=10)

    def swayed(prompt: str) -> str:
        # follows the leading frame: says Entailed when 'supported/true/Entailed'
        # is pushed, Refuted when 'false/Refuted' is pushed.
        p = prompt.lower()
        if "obviously false" in p or "flagged this claim as false" in p or "no one would believe" in p:
            return "Final answer: Refuted"
        if "clearly supported" in p or "verified this claim as true" in p or "is entailed" in p:
            return "Final answer: Entailed"
        return "Final answer: Entailed"

    per, rep = run_fiv(items, swayed, n_rollouts=2)
    assert rep.flip_rate > 0.5  # framing moves it a lot
    assert rep.self_consistency == 1.0  # deterministic stub -> no rollout noise


def test_invariant_model_zero_fiv():
    items = load_tabfact("test", limit=10)

    def grounded(prompt: str) -> str:
        # ignores framing; always says the same thing (here: Entailed)
        return "Final answer: Entailed"

    per, rep = run_fiv(items, grounded, n_rollouts=2)
    assert rep.flip_rate == 0.0
    assert all(r.invariant for r in per)


def test_framings_present():
    assert len(FRAMINGS) >= 6
    labels = [f[0] for f in FRAMINGS]
    assert "neutral" in labels and len(set(labels)) == len(labels)
