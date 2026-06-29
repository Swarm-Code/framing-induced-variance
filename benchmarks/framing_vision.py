"""Stochastic FIV grid — B2.

Extends single-shot FIV (`run_fiv`) with *multiple stochastic executions* per
framing, so we can separate two different sources of answer variation:

  * FIV (framing-induced variance): the answer changes because the *question
    framing* changed — the thing we care about (steering by language, not data).
  * stochastic noise: the answer changes run-to-run for the *same* framing — the
    model is just non-deterministic.

The EMNLP P/¬P argument: cross-framing variance is only meaningful if it exceeds
the within-framing stochastic floor. We quantify that floor as `self_consistency`
(see metric.py): the mean, over (item, framing), of the modal-stance fraction
across executions. FIV must exceed (1 - self_consistency) to prove framing — not
noise — moves the answer.

Model-decoupled exactly like `run_fiv`: takes `ask: BenchItem -> str`.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path

from pydantic import BaseModel, Field

from .framing import FivMetric, Framing, fiv_metric, frame_variants
from .framing import FivResult
from .models import BenchItem
from .runner import _classify

AskFn = Callable[[BenchItem], str]


class GridCell(BaseModel):
    """All executions for one (item, framing) pair."""

    item_id: str
    framing: str
    raw_answers: list[str] = Field(default_factory=list)
    stances: list[str] = Field(default_factory=list)
    modal_stance: str = "other"
    modal_fraction: float = 0.0  # how dominant the modal stance was (stochastic floor)


def _stance(item: BenchItem, raw: str) -> str:
    r = _classify(item, raw)
    if r.refused:
        return "refused"
    if r.followed_truth:
        return "truth"
    if r.followed_shortcut:
        return "shortcut"
    return "other"


def run_fiv_grid(
    items: Sequence[BenchItem],
    ask: AskFn,
    *,
    n_framings: int = 12,
    n_executions: int = 10,
    pipeline: str = "baseline",
    results_path: str | Path | None = None,
) -> tuple[list[FivResult], FivMetric, list[GridCell]]:
    """Run `ask` over `n_framings` framings × `n_executions` reps per item.

    Returns:
      * per-item `FivResult` (using the MODAL stance per framing — robust to noise),
      * the aggregate `FivMetric`,
      * the flat list of `GridCell`s (one per item×framing) for self_consistency.
    """
    results: list[FivResult] = []
    cells: list[GridCell] = []

    for item in items:
        framings: list[Framing] = frame_variants(item, n=n_framings)
        answers: dict[str, str] = {}
        stances: dict[str, str] = {}

        for fr in framings:
            variant = item.model_copy(update={"question": fr.question})
            raws = [ask(variant) for _ in range(max(1, n_executions))]
            st = [_stance(item, raw) for raw in raws]
            counts = Counter(st)
            modal, modal_n = counts.most_common(1)[0]
            cells.append(
                GridCell(
                    item_id=item.id,
                    framing=fr.label,
                    raw_answers=raws,
                    stances=st,
                    modal_stance=modal,
                    modal_fraction=modal_n / len(st),
                )
            )
            # The framing's representative answer/stance is the modal one.
            answers[fr.label] = raws[st.index(modal)]
            stances[fr.label] = modal

        invariant = len(set(stances.values())) == 1
        truth_invariant = all(s == "truth" for s in stances.values())
        results.append(
            FivResult(
                item_id=item.id,
                dataset=item.dataset,
                pipeline=pipeline,
                answers=answers,
                stances=stances,
                invariant=invariant,
                truth_invariant=truth_invariant,
            )
        )

    if results_path is not None:
        path = Path(results_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as fh:
            for r in results:
                fh.write(r.model_dump_json() + "\n")

    return results, fiv_metric(results, pipeline=pipeline), cells
