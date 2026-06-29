"""Adapters for EXISTING external benchmarks — we run the baseline-vs-skeptic
pipeline against them, we do NOT author labels.

Sources (real, fetched into data/external/ via the HF datasets-server):
  * TruthfulQA (truthfulqa/truthful_qa, multiple_choice) — truthfulness. The gold
    is the dataset's own mc1 label (labels[i]==1). The shortcut is a real distractor
    choice (a plausible-but-false answer) — exactly the "imitative falsehood" the
    benchmark is built around. Truth-vs-shortcut is the dataset's design, not ours.
  * FinQA (dreamerdeo/finqa) — numeric financial QA over filings. Gold = `answer`.
  * ConvFinQA (TheFinAI/flare-convfinqa) — conversational financial QA. Gold = `answer`.

Each row becomes a `BenchItem` whose `truth_answer` is the dataset's gold. For the
MCQ truthfulness set we also carry a `shortcut_answer` (a labelled-false choice) so
the Truth-Shortcut Margin applies directly. For the numeric finance sets there is
no labelled distractor, so `shortcut_answer == truth_answer` (aligned) and we score
exact-match truth accuracy + framing-induced variance instead.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..models import BenchItem, Misleader

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXT = _REPO_ROOT / "data" / "external"


def _load(name: str) -> list[dict]:
    p = _EXT / f"{name}.json"
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. Fetch real items via the datasets-server first "
            "(see loop_tracker G2')."
        )
    return json.loads(p.read_text())


def load_truthfulqa(path: str | Path | None = None) -> list[BenchItem]:
    """TruthfulQA mc1 → BenchItem. truth = labelled-true choice; shortcut = the
    first labelled-false choice (a real imitative falsehood). Options preserved."""
    rows = json.loads(Path(path).read_text()) if path else _load("truthfulqa")
    items: list[BenchItem] = []
    for i, r in enumerate(rows):
        mc1 = r.get("mc1_targets", {})
        choices = mc1.get("choices", [])
        labels = mc1.get("labels", [])
        if not choices or 1 not in labels:
            continue
        truth = choices[labels.index(1)]
        wrong = [c for c, lab in zip(choices, labels) if lab == 0]
        shortcut = wrong[0] if wrong else truth
        items.append(
            BenchItem(
                id=f"tqa-{i:03d}",
                dataset="truthfulqa",
                question=r["question"],
                options=choices,
                truth_answer=truth,
                shortcut_answer=shortcut,
                misleader=Misleader.NONE,
            )
        )
    return items


def _finqa_context(r: dict, *, max_chars: int = 2000) -> list[dict]:
    pre = " ".join(r.get("pre_text", []) or [])
    post = " ".join(r.get("post_text", []) or [])
    table = r.get("table")
    ctx = {"pre_text": pre[:max_chars], "post_text": post[:max_chars]}
    if table:
        ctx["table"] = table
    return [ctx]


def load_finqa(path: str | Path | None = None) -> list[BenchItem]:
    """FinQA → BenchItem. Numeric gold = `answer`; context (filing text + table)
    carried in `table`. No labelled distractor → aligned (truth==shortcut); scored
    by exact-match truth accuracy and framing-induced variance."""
    rows = json.loads(Path(path).read_text()) if path else _load("finqa")
    items: list[BenchItem] = []
    for i, r in enumerate(rows):
        ans = str(r.get("answer", "")).strip()
        if not ans:
            continue
        items.append(
            BenchItem(
                id=f"finqa-{i:03d}",
                dataset="finqa",
                question=r["question"],
                table=_finqa_context(r),
                truth_answer=ans,
                shortcut_answer=ans,  # numeric set: no labelled distractor
                misleader=Misleader.NONE,
            )
        )
    return items


def load_convfinqa(path: str | Path | None = None) -> list[BenchItem]:
    """ConvFinQA → BenchItem. Gold = `answer`; the `query` already embeds the
    pre/post/table context for the final turn."""
    rows = json.loads(Path(path).read_text()) if path else _load("convfinqa")
    items: list[BenchItem] = []
    for i, r in enumerate(rows):
        ans = str(r.get("answer", "")).strip()
        if not ans:
            continue
        items.append(
            BenchItem(
                id=f"convfinqa-{i:03d}",
                dataset="convfinqa",
                question=r["query"],
                truth_answer=ans,
                shortcut_answer=ans,
                misleader=Misleader.NONE,
            )
        )
    return items


LOADERS = {
    "truthfulqa": load_truthfulqa,
    "finqa": load_finqa,
    "convfinqa": load_convfinqa,
}
