"""Statistical rigor for baseline-vs-skeptic comparisons — G1.

Point estimates over n=7 are anecdotes. This module turns per-item outcomes into
distributions and significance tests so claims survive review:

  * bootstrap confidence intervals (percentile) for any rate / margin,
  * paired comparison of two pipelines on the SAME items:
      - McNemar exact test (binomial) on discordant pairs,
      - paired bootstrap CI on the metric delta,
  * effect size (Cohen's h for two proportions).

Pure stdlib (random + math) so it is deterministic under a seed and adds no heavy
dependency to the test path. Operates on lists of BenchResult.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from .models import BenchResult


@dataclass
class CI:
    point: float
    lo: float
    hi: float
    n: int

    def __str__(self) -> str:
        return f"{self.point:+.3f} [{self.lo:+.3f}, {self.hi:+.3f}] (n={self.n})"


def _tsm_of(results: Sequence[BenchResult]) -> float:
    conflict = [r for r in results if r.is_conflict]
    if not conflict:
        return 0.0
    n = len(conflict)
    t = sum(1 for r in conflict if r.followed_truth) / n
    s = sum(1 for r in conflict if r.followed_shortcut) / n
    return t - s


def _truth_acc_of(results: Sequence[BenchResult]) -> float:
    if not results:
        return 0.0
    return sum(1 for r in results if r.followed_truth) / len(results)


def bootstrap_ci(
    results: Sequence[BenchResult],
    stat: Callable[[Sequence[BenchResult]], float] = _tsm_of,
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> CI:
    """Percentile bootstrap CI for `stat` by resampling items with replacement."""
    items = list(results)
    n = len(items)
    if n == 0:
        return CI(0.0, 0.0, 0.0, 0)
    rng = random.Random(seed)
    point = stat(items)
    boots = []
    for _ in range(n_boot):
        sample = [items[rng.randrange(n)] for _ in range(n)]
        boots.append(stat(sample))
    boots.sort()
    lo = boots[int((alpha / 2) * n_boot)]
    hi = boots[int((1 - alpha / 2) * n_boot) - 1]
    return CI(point, lo, hi, n)


def tsm_ci(results, **kw) -> CI:
    return bootstrap_ci(results, _tsm_of, **kw)


def truth_acc_ci(results, **kw) -> CI:
    return bootstrap_ci(results, _truth_acc_of, **kw)


@dataclass
class PairedTest:
    delta: float  # skeptic_metric - baseline_metric
    delta_ci: CI  # paired bootstrap CI on the delta
    n_pairs: int
    b: int  # baseline-right, skeptic-wrong (discordant)
    c: int  # skeptic-right, baseline-wrong (discordant)
    mcnemar_p: float  # exact two-sided binomial p on discordant pairs
    cohens_h: float

    @property
    def significant(self) -> bool:
        return self.mcnemar_p < 0.05


def _binom_two_sided_p(b: int, c: int) -> float:
    """Exact McNemar: two-sided binomial p with k=min(b,c), n=b+c, p=0.5."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # P(X<=k) under Binom(n,0.5), doubled, capped at 1.
    cum = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * cum)


def _cohens_h(p1: float, p2: float) -> float:
    def phi(p: float) -> float:
        p = min(max(p, 0.0), 1.0)
        return 2 * math.asin(math.sqrt(p))

    return phi(p1) - phi(p2)


def paired_compare(
    baseline: Sequence[BenchResult],
    skeptic: Sequence[BenchResult],
    *,
    correct: Callable[[BenchResult], bool] = lambda r: r.followed_truth,
    metric: Callable[[Sequence[BenchResult]], float] = _tsm_of,
    n_boot: int = 2000,
    seed: int = 0,
) -> PairedTest:
    """Compare two pipelines run on the SAME items (matched by item_id).

    McNemar on per-item correctness + paired bootstrap CI on the metric delta.
    """
    bmap = {r.item_id: r for r in baseline}
    smap = {r.item_id: r for r in skeptic}
    ids = [i for i in bmap if i in smap]
    bpairs = [bmap[i] for i in ids]
    spairs = [smap[i] for i in ids]

    b = sum(1 for i in ids if correct(bmap[i]) and not correct(smap[i]))
    c = sum(1 for i in ids if correct(smap[i]) and not correct(bmap[i]))
    p = _binom_two_sided_p(b, c)

    delta = metric(spairs) - metric(bpairs)
    rng = random.Random(seed)
    n = len(ids)
    boots = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)] if n else []
        boots.append(metric([spairs[j] for j in idx]) - metric([bpairs[j] for j in idx]))
    boots.sort()
    lo = boots[int(0.025 * n_boot)] if boots else 0.0
    hi = boots[int(0.975 * n_boot) - 1] if boots else 0.0

    h = _cohens_h(
        sum(1 for r in spairs if correct(r)) / n if n else 0.0,
        sum(1 for r in bpairs if correct(r)) / n if n else 0.0,
    )
    return PairedTest(
        delta=delta,
        delta_ci=CI(delta, lo, hi, n),
        n_pairs=n,
        b=b,
        c=c,
        mcnemar_p=p,
        cohens_h=h,
    )
