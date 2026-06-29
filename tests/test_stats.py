"""Gate — statistical layer (bootstrap CIs, McNemar, paired bootstrap, effect size)."""

from __future__ import annotations

from benchmarks.models import BenchResult
from benchmarks.stats import paired_compare, tsm_ci, truth_acc_ci


def _r(i, pipeline, truth, shortcut=False, conflict=True):
    return BenchResult(
        item_id=i, dataset="t", is_conflict=conflict, pipeline=pipeline,
        model_answer="x", followed_truth=truth, followed_shortcut=shortcut, refused=False,
    )


def test_tsm_ci_brackets_point_and_is_ordered():
    # 8/10 truth, 2/10 shortcut -> TSM = 0.6
    res = [_r(f"i{i}", "b", True) for i in range(8)] + [_r(f"i{i}", "b", False, True) for i in range(8, 10)]
    ci = tsm_ci(res, n_boot=500, seed=1)
    assert abs(ci.point - 0.6) < 1e-9
    assert ci.lo <= ci.point <= ci.hi
    assert ci.n == 10


def test_truth_acc_ci_full_truth_is_one():
    res = [_r(f"i{i}", "b", True) for i in range(20)]
    ci = truth_acc_ci(res, n_boot=300, seed=2)
    assert ci.point == 1.0
    assert ci.lo == 1.0 and ci.hi == 1.0


def test_mcnemar_detects_clear_improvement():
    # 30 items: baseline wrong, skeptic right on 12; concordant-right on 18.
    ids = [f"i{i}" for i in range(30)]
    base = [_r(i, "baseline", True) for i in ids[:18]] + [_r(i, "baseline", False, True) for i in ids[18:]]
    skep = [_r(i, "skeptic", True) for i in ids]  # skeptic right on all
    t = paired_compare(base, skep, n_boot=500, seed=3)
    assert t.c == 12 and t.b == 0  # 12 discordant in skeptic's favor
    assert t.significant  # p < 0.05
    assert t.delta > 0
    assert t.delta_ci.lo <= t.delta <= t.delta_ci.hi


def test_no_difference_is_not_significant():
    ids = [f"i{i}" for i in range(20)]
    base = [_r(i, "baseline", True) for i in ids]
    skep = [_r(i, "skeptic", True) for i in ids]  # identical
    t = paired_compare(base, skep, n_boot=300, seed=4)
    assert t.b == 0 and t.c == 0
    assert not t.significant
    assert t.mcnemar_p == 1.0


def test_small_n_is_not_significant():
    # The n=7 critique: a 4-3 split must NOT reach significance.
    ids = [f"i{i}" for i in range(7)]
    base = [_r(i, "baseline", False, True) for i in ids]
    skep = [_r(i, "skeptic", True) for i in ids[:4]] + [_r(i, "skeptic", False, True) for i in ids[4:]]
    t = paired_compare(base, skep, n_boot=300, seed=5)
    assert t.c == 4 and t.b == 0
    # exact binomial two-sided p for 4/4 = 2*0.5^4 = 0.125 -> NOT significant
    assert not t.significant
