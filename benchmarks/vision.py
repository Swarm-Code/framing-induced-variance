"""Vision ask-builders — C1.

Turns a `Multivac` harness into an `AskFn` (`BenchItem -> str`) for the FIV /
shortcut runners, in one of two pipelines:

  * baseline — the model answers from the CHART IMAGE only (numbers/visual), the
    naive "look at the dashboard and tell me the trend" agent.
  * skeptic  — verify-before-conclude: the model gets the chart AND the underlying
    data table, is told the chart MAY be misleading, and is instructed to re-read
    the table and, if the visual impression contradicts the data, trust the data
    and name which misleader (gorilla) it sees from the Misleader taxonomy.

The ONLY difference between the two pipelines is the prompt text — identical
model, identical image, identical tools — so any measured gap is attributable to
the harness/prompt, not the model. Mirrors the model-decoupled runner contract.
"""

from __future__ import annotations

import json
from collections.abc import Callable

from .models import BenchItem, Misleader

AskFn = Callable[[BenchItem], str]

_MISLEADERS = ", ".join(m.value for m in Misleader if m is not Misleader.NONE)

BASELINE_PROMPT_CHART = (
    "You are a data analyst. Here is a chart. Answer the question based on the "
    "chart.\n\nQuestion: {question}\n{options}"
    "Answer concisely."
)

BASELINE_PROMPT_TABLE = (
    "You are a data analyst. Answer the question based on the information given.\n\n"
    "Question: {question}\n{options}"
    "Available information:\n{table}\n\n"
    "Answer concisely with the single best label."
)

SKEPTIC_PROMPT_CHART = (
    "You are a SKEPTICAL data analyst. You are shown a chart, but charts can be "
    "misleading. Do NOT trust the chart's visual impression on its own.\n\n"
    "Question: {question}\n{options}"
    "Here is the underlying data table (ground truth):\n{table}\n\n"
    "Procedure: (1) state what the chart visually appears to say; (2) re-read the "
    "data table and compute the real answer from the numbers; (3) if the visual "
    "impression contradicts the data, TRUST THE DATA and name which misleader you "
    f"see from this taxonomy: [{_MISLEADERS}]. If the data is insufficient or the "
    "chart is misleading, say so explicitly rather than guessing.\n"
    "Final answer:"
)

SKEPTIC_PROMPT_TABLE = (
    "You are a SKEPTICAL analyst. A shortcut cue (e.g. a metadata proxy such as "
    "tissue / disease / study) is given alongside the real evidence, but the cue "
    "can be MISLEADING. Do NOT answer from the metadata proxy on its own.\n\n"
    "Question: {question}\n{options}"
    "Available information (proxy cue AND the real evidence):\n{table}\n\n"
    "Procedure: (1) note what the metadata proxy nudges toward; (2) read the real "
    "evidence (e.g. expression markers) and derive the answer from it; (3) if the "
    "proxy contradicts the evidence, TRUST THE EVIDENCE, not the proxy. If the "
    "evidence is insufficient, say so rather than guessing.\n"
    "Final answer (single best label):"
)


def _fmt_options(item: BenchItem) -> str:
    if not item.options:
        return ""
    opts = "; ".join(item.options)
    return f"Options: {opts}\n"


def _fmt_table(item: BenchItem, *, max_rows: int = 50) -> str:
    if not item.table:
        return "(no table provided)"
    rows = item.table[:max_rows]
    return json.dumps(rows, separators=(",", ":"))


def build_prompt(item: BenchItem, *, skeptic: bool) -> str:
    """Build the pipeline-specific prompt text for one item (pure, testable).

    Chart-aware: if the item has a `chart_path` we use the chart-centric templates
    (the model will receive the image); otherwise (tabular benchmarks like
    SC-ShortcutBench) we use table-centric templates that never reference a chart,
    so the model doesn't refuse with "please provide the chart".
    """
    has_chart = bool(item.chart_path)
    if skeptic:
        tmpl = SKEPTIC_PROMPT_CHART if has_chart else SKEPTIC_PROMPT_TABLE
    else:
        tmpl = BASELINE_PROMPT_CHART if has_chart else BASELINE_PROMPT_TABLE
    return tmpl.format(
        question=item.question,
        options=_fmt_options(item),
        table=_fmt_table(item),
    )


def vision_ask(harness, *, skeptic: bool) -> AskFn:
    """Return an `AskFn` that runs one `BenchItem` through the harness with vision.

    Each call resets conversation history (items are independent), builds the
    pipeline prompt, and passes the rendered chart PNG via `images=` when present.
    Works offline (deterministic provider) and live (Cerebras Gemma 4) unchanged.
    """

    def ask(item: BenchItem) -> str:
        harness.reset()
        prompt = build_prompt(item, skeptic=skeptic)
        images = [item.chart_path] if item.chart_path else None
        return harness.chat(prompt, images=images).output

    return ask
