"""Data contract for the eval harness. Pure Pydantic — no logic, no I/O.

Every external benchmark is normalized by an adapter into a list of `BenchItem`,
so the runner and the metric never depend on a specific dataset's raw shape.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Misleader(str, Enum):
    """Taxonomy of chart 'misleaders' (the shortcut cue) — mirrors the Misviz /
    misleading-visualization literature. A conflict item carries exactly one
    dominant misleader whose visual implication contradicts the table."""

    NONE = "none"  # honest chart (control / aligned item)
    TRUNCATED_AXIS = "truncated_axis"  # y-axis doesn't start at 0 -> exaggerated
    INVERTED_AXIS = "inverted_axis"  # axis reversed -> trend looks flipped
    DUAL_AXIS = "dual_axis"  # two y-scales -> false correlation
    CHERRY_PICKED_RANGE = "cherry_picked_range"  # cropped x-range -> false trend
    THREE_D_DISTORTION = "three_d_distortion"  # 3D perspective skews magnitudes
    INCONSISTENT_BINNING = "inconsistent_binning"  # uneven bins -> misread density
    MISLEADING_AGGREGATION = "misleading_aggregation"  # wrong stat hides signal


class BenchItem(BaseModel):
    """One normalized benchmark example.

    A *conflict* item is constructed so the chart's visual cue (the shortcut)
    implies `shortcut_answer` while the underlying data implies `truth_answer`,
    and the two differ. An *aligned* item has them equal (honest chart / control).
    """

    id: str
    dataset: str = "misviz"  # provenance tag
    question: str  # the QA prompt posed to the model

    # The chart the model "looks" at, and the raw table behind it.
    chart_path: str | None = None  # rendered PNG (relative to repo root)
    table: list[dict] = Field(default_factory=list)  # ground-truth data rows

    # Multiple-choice options (Misviz/CALVI are MCQ); free-text allowed if empty.
    options: list[str] = Field(default_factory=list)

    # The two answers that let us tell which cue the model followed.
    truth_answer: str  # correct answer per the underlying data
    shortcut_answer: str  # answer the chart misleader nudges toward

    misleader: Misleader = Misleader.NONE

    @property
    def is_conflict(self) -> bool:
        """True when the shortcut and truth point to different answers."""
        return self.truth_answer != self.shortcut_answer


class BenchResult(BaseModel):
    """The outcome of running one pipeline against one `BenchItem`."""

    item_id: str
    dataset: str = "misviz"
    is_conflict: bool
    pipeline: str = "baseline"  # "baseline" | "skeptic"

    model_answer: str  # raw answer the pipeline produced
    followed_truth: bool  # answer matched truth_answer
    followed_shortcut: bool  # answer matched shortcut_answer
    refused: bool = False  # pipeline abstained (allowed, counts as not-shortcut)


class ShortcutMetric(BaseModel):
    """Aggregate shortcut-reliance scorecard over a set of results."""

    n_total: int
    n_conflict: int
    truth_accuracy: float  # frac of conflict items answered with truth
    shortcut_rate: float  # frac of conflict items answered with shortcut
    refusal_rate: float  # frac of conflict items refused
    truth_shortcut_margin: float  # truth_accuracy - shortcut_rate  (TSM)
    pipeline: str = "baseline"
