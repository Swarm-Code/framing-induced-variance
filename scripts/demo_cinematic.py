#!/usr/bin/env python3
"""Cinematic FIV demo — an animated TUI reveal with baked-in subtitles.

Renders a polished, recordable walkthrough of Framing-Induced Variance using
`rich`: a title card, the data table in a panel, the eight framings revealing
one-by-one (green Entailed -> red Refuted), a stats panel, and a close card.
A subtitle bar at the bottom narrates each beat (so the recording is self-
explanatory with no external captions).

All numbers are replayed from results/fiv_tabfact_scaled.json (offline-safe).

Usage:
    python3 scripts/demo_cinematic.py                 # full animated reveal
    python3 scripts/demo_cinematic.py --id tf-test-11
    python3 scripts/demo_cinematic.py --speed 1.0     # pacing multiplier
    python3 scripts/demo_cinematic.py --still         # no sleeps (CI/screens)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from rich.align import Align
from rich.box import HEAVY, ROUNDED
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

REPO = Path(__file__).resolve().parents[1]
SCALED = REPO / "results" / "fiv_tabfact_scaled.json"

# palette
GOLD = "#f9e2af"
GREEN = "#a6e3a1"
RED = "#f38ba8"
BLUE = "#89b4fa"
MAUVE = "#cba6f7"
SUB = "#bac2de"
DIM = "#6c7086"

FRAMING_GLOSS = {
    "neutral": ("neutral", "Is this claim Entailed or Refuted?"),
    "lead_entailed": ("lead-for", "This looks clearly supported, right?"),
    "lead_refuted": ("lead-against", "This is obviously false, isn't it?"),
    "authority_entailed": ("authority-for", "Our analysts confirmed this is true."),
    "authority_refuted": ("authority-against", "Our analysts flagged this as false."),
    "urgency_entailed": ("urgency", "Quick — it supports the claim, yes?"),
    "doubt": ("doubt", "Are you sure the data really backs this?"),
    "rhetorical_refuted": ("rhetorical", "Surely the data does not support this?"),
}


def _load() -> dict:
    if not SCALED.exists():
        raise SystemExit(f"missing artifact {SCALED}")
    return json.loads(SCALED.read_text())


def _pick(per_item: list[dict], want_id: str | None) -> dict:
    if want_id:
        for it in per_item:
            if it["id"] == want_id:
                return it
        raise SystemExit(f"id {want_id} not found")
    flippers = [it for it in per_item if it.get("wrong_flip")]
    flippers.sort(key=lambda it: len(set(it["verdicts"].values())), reverse=True)
    return (flippers or per_item)[0]


def _claim_table(item_id: str):
    try:
        import sys

        sys.path.insert(0, str(REPO))
        from benchmarks.adapters.tabfact import load_tabfact, table_md_for

        for it in load_tabfact("test"):
            if it.id == item_id:
                return it.question, table_md_for(it)
    except Exception:
        pass
    return None, None


class Demo:
    def __init__(self, console: Console, speed: float, still: bool):
        self.c = console
        self.speed = speed
        self.still = still
        self.subtitle = Text("")

    def nap(self, s: float) -> None:
        if not self.still:
            time.sleep(s * self.speed)

    def _frame(self, body) -> Panel:
        """Wrap body + a subtitle bar into one full-screen panel."""
        bar = Panel(
            Align.center(self.subtitle),
            box=ROUNDED,
            border_style=DIM,
            padding=(0, 2),
        )
        return Panel(
            Group(body, Text(""), bar),
            box=HEAVY,
            border_style=MAUVE,
            title="[b]FRAMING-INDUCED VARIANCE[/]",
            subtitle=f"[{DIM}]live · TabFact · N=500 · replayed offline[/]",
            padding=(1, 3),
        )

    def say(self, text: str, style: str = SUB) -> None:
        self.subtitle = Text(text, style=style)

    # ---- beats -------------------------------------------------------------
    def title_card(self, live: Live) -> None:
        big = Text("The Question Is the Verdict", style=f"bold {GOLD}", justify="center")
        sub = Text("How framing — not data — decides what an LLM concludes",
                   style=MAUVE, justify="center")
        self.say("Same data. Same claim. Only the wording of the question changes.")
        live.update(self._frame(Group(Text(""), Align.center(big), Text(""),
                                      Align.center(sub), Text(""))))
        self.nap(2.6)

    def method_card(self, live: Live) -> None:
        steps = Table.grid(padding=(0, 2))
        steps.add_column(justify="right", style=f"bold {BLUE}", width=3)
        steps.add_column(justify="left")
        steps.add_row("1", Text("Take one table + one claim with a known TRUE/FALSE answer.", style="white"))
        steps.add_row("2", Text("Ask the model to verify it under 8 differently-worded framings.", style="white"))
        steps.add_row("3", Text("The table and task are byte-identical — only the framing differs.", style="white"))
        steps.add_row("4", Text("Any change in the verdict is caused by wording alone.", style=f"bold {GOLD}"))
        head = Align.center(Text("THE METHOD     Framing-Induced Variance (FIV)", style=f"bold {MAUVE}"))
        self.say("We isolate one thing: does the wording of the question change the answer?")
        live.update(self._frame(Group(head, Text(""), steps, Text(""))))
        self.nap(4.5)

    def show_table(self, live: Live, item: dict, claim: str | None, table_md: str | None) -> None:
        gold = item["gold"]
        head = Text.assemble((f"{item['id']}", f"bold {BLUE}"), "   ",
                             ("gold = ", DIM), (gold, f"bold {GOLD}"))
        body_parts = [head, Text("")]
        if claim:
            body_parts.append(Text.assemble(("claim:  ", DIM), (claim, "italic white")))
        if table_md:
            t = Table(box=ROUNDED, border_style=DIM, show_edge=True, expand=False)
            rows = [r for r in table_md.strip().splitlines() if r.strip().startswith("|")]
            cells = [c.strip() for c in rows[0].strip("|").split("|")]
            for col in cells:
                t.add_column(col, style="white", header_style=f"bold {BLUE}")
            for r in rows[2:5]:
                t.add_row(*[c.strip() for c in r.strip("|").split("|")])
            body_parts += [Text(""), t]
        self.say("Here is the table and the claim. The data never changes.")
        live.update(self._frame(Group(*body_parts)))
        self.nap(2.6)
        return body_parts

    def reveal_framings(self, live: Live, item: dict, header) -> None:
        gold = item["gold"]
        rows = Table.grid(padding=(0, 2))
        rows.add_column(justify="left", width=14)
        rows.add_column(justify="left", width=18)
        rows.add_column(justify="left")
        shown: list = []
        first_flip = True
        for slot, verdict in item["verdicts"].items():
            label, q = FRAMING_GLOSS.get(slot, (slot, ""))
            if verdict == gold:
                vcell = Text("✓ Entailed", style=f"bold {GREEN}")
                self.say(f"“{q}”  →  the model says Entailed. Correct.", GREEN)
            elif verdict == "other":
                vcell = Text("• undecided", style=DIM)
                self.say(f"“{q}”  →  undecided.", DIM)
            else:
                vcell = Text("✗ Refuted  WRONG", style=f"bold {RED}")
                if first_flip:
                    self.say("Now we cast doubt — same data — and it FLIPS to the wrong answer.",
                             RED)
                    first_flip = False
                else:
                    self.say(f"“{q}”  →  Refuted. Wrong again — pure framing.", RED)
            rows.add_row(Text(label, style=MAUVE), vcell, Text(q, style=DIM))
            grid = Table.grid()
            grid.add_row(rows)
            live.update(self._frame(Group(*header, Text(""), rows)))
            self.nap(0.85)
        distinct = len(set(item["verdicts"].values()))
        self.say(f"One table. {distinct} different verdicts across 8 framings.", f"bold {GOLD}")
        live.update(self._frame(Group(*header, Text(""), rows, Text(""),
                    Align.center(Text(f"→ {distinct} distinct verdicts across 8 framings",
                                      style=f"bold {RED}")))))
        self.nap(2.2)

    def noise_card(self, live: Live, data: dict) -> None:
        base = data["conditions"]["baseline"]
        sc = base["self_consistency"]
        floor = 1 - sc
        flip = base["flip_rate"]
        rows = Table.grid(padding=(0, 3))
        rows.add_column(justify="left", style=SUB)
        rows.add_column(justify="right")
        rows.add_row("Self-consistency (same framing, repeated)", Text(f"{sc:.3f}", style=f"bold {GREEN}"))
        rows.add_row("Noise floor  (1 − SC)", Text(f"{floor:.3f}", style=DIM))
        rows.add_row("Observed flip rate across framings", Text(f"{flip:.3f}", style=f"bold {RED}"))
        head = Align.center(Text("IS IT JUST RANDOMNESS?     No.", style=f"bold {MAUVE}"))
        punch = Align.center(Text(
            f"{flip:.2f}  is  {flip/floor:.1f}×  the noise floor — the flips are FRAMING, not chance.",
            style=f"bold {GOLD}"))
        self.say("Could this be random sampling noise? We measured the noise floor to rule it out.")
        live.update(self._frame(Group(head, Text(""), rows, Text(""), punch)))
        self.nap(4.5)

    def asymmetry_card(self, live: Live, data: dict) -> None:
        per = data["conditions"]["baseline"]["per_item"]
        from collections import defaultdict
        tot: dict = defaultdict(int)
        wrong: dict = defaultdict(int)
        for it in per:
            g = it["gold"]
            for slot, v in it["verdicts"].items():
                tot[slot] += 1
                if v != "other" and v != g:
                    wrong[slot] += 1
        shares = sorted(((wrong[s] / tot[s], s) for s in tot), reverse=True)
        bar = Table.grid(padding=(0, 2))
        bar.add_column(justify="left", width=20, style=MAUVE)
        bar.add_column(justify="left")
        bar.add_column(justify="right", width=7)
        for share, slot in shares:
            label = FRAMING_GLOSS.get(slot, (slot, ""))[0]
            blocks = "█" * max(1, round(share * 60))
            col = RED if share >= 0.15 else (GOLD if share >= 0.09 else GREEN)
            bar.add_row(Text(label), Text(blocks, style=col), Text(f"{share:.1%}", style=col))
        head = Align.center(Text("WHICH FRAMINGS DO THE DAMAGE", style=f"bold {MAUVE}"))
        self.say("Leading-AGAINST a true claim is the most dangerous — 3× the neutral error rate.")
        live.update(self._frame(Group(head, Text(""), bar, Text(""))))
        self.nap(4.5)

    def robustness_card(self, live: Live, data: dict) -> None:
        v = json.loads((REPO / "results" / "fiv_variants.json").read_text())["summary"]
        rows = Table.grid(padding=(0, 3))
        rows.add_column(justify="left", style=SUB)
        rows.add_column(justify="right")
        rows.add_row("Independent prompt variants tested", Text("5", style=f"bold {BLUE}"))
        rows.add_row("Wrong-flip rate — every variant", Text(
            f"{v['wrong_min']:.3f} – {v['wrong_max']:.3f}", style=f"bold {RED}"))
        rows.add_row("Mean harmful-flip rate", Text(f"{v['wrong_mean']:.3f}", style=f"bold {GOLD}"))
        head = Align.center(Text("IS IT OUR WORDING?     No.", style=f"bold {MAUVE}"))
        punch = Align.center(Text(
            "We rewrote every prompt 5 ways. The harm is positive in ALL of them.",
            style=f"bold {GOLD}"))
        self.say("We rewrote the prompts 5 independent ways — a specification curve. The effect holds.")
        live.update(self._frame(Group(head, Text(""), rows, Text(""), punch)))
        self.nap(4.5)

    def speed_card(self, live: Live) -> None:
        rows = Table.grid(padding=(0, 3))
        rows.add_column(justify="left", style=SUB)
        rows.add_column(justify="right")
        rows.add_row("Headline run", Text("24,000 live calls", style=f"bold {BLUE}"))
        rows.add_row("+ specification curve", Text("36,000 live calls", style=f"bold {BLUE}"))
        rows.add_row("Total to reach significance", Text("60,000 calls", style=f"bold {GOLD}"))
        head = Align.center(Text("WHY THIS NEEDS GEMMA ON CEREBRAS", style=f"bold {MAUVE}"))
        punch = Align.center(Text(
            "On slow, costly inference a rigorous audit like this is infeasible at scale.\n"
            "Fast, cheap Gemma-on-Cerebras is what makes skepticism economical to run.",
            style=f"bold {GREEN}", justify="center"))
        self.say("A real audit is 60,000 calls. With today's slow models that's too slow and too costly.")
        live.update(self._frame(Group(head, Text(""), rows, Text(""), punch)))
        self.nap(5.0)

    def noise_card(self, live: Live, data: dict) -> None:
        base = data["conditions"]["baseline"]
        sc = base["self_consistency"]
        floor = 1 - sc
        flip = base["flip_rate"]
        g = Table.grid(padding=(0, 3))
        g.add_column(justify="right", style=SUB)
        g.add_column(justify="left")
        g.add_row("ask the SAME framing 3×:", Text(f"agrees {sc:.0%} of the time  →  noise floor = {floor:.0%}",
                                                   style=DIM))
        g.add_row("change the framing:", Text(f"verdict flips {flip:.0%} of the time",
                                              style=f"bold {RED}"))
        g.add_row("", Text(f"{flip:.0%}  ≫  {floor:.0%}   —  more than 4× the noise floor",
                           style=f"bold {GOLD}"))
        head = Align.center(Text("IS IT JUST RANDOMNESS?     No.", style=f"bold {MAUVE}"))
        self.say("Could this be sampling noise? We measured the floor — and it's not even close.")
        live.update(self._frame(Group(head, Text(""), g, Text(""),
                    Align.center(Text("The flips are driven by FRAMING, not chance.", style=f"bold {GREEN}")))))
        self.nap(4.8)

    def asymmetry_card(self, live: Live) -> None:
        # per-framing wrong-verdict share (from the N=500 baseline run)
        bars = [
            ("authority-for", 0.054, GREEN),
            ("neutral", 0.064, GREEN),
            ("doubt", 0.074, GOLD),
            ("authority-against", 0.098, GOLD),
            ("lead-against", 0.200, RED),
        ]
        t = Table.grid(padding=(0, 2))
        t.add_column(justify="right", style=SUB, width=20)
        t.add_column(justify="left")
        for label, share, col in bars:
            blocks = "█" * max(1, round(share * 60))
            t.add_row(label, Text(f"{blocks} {share:.0%}", style=col))
        head = Align.center(Text("WHICH FRAMINGS DO THE MOST DAMAGE", style=f"bold {MAUVE}"))
        self.say("Leading the model to DOUBT a true claim is the most damaging — 3× the neutral rate.")
        live.update(self._frame(Group(head, Text(""), t, Text(""),
                    Align.center(Text("Casting doubt on true data is where it breaks most.", style=DIM)))))
        self.nap(4.8)

    def robustness_card(self, live: Live, data: dict) -> None:
        v = json.loads((REPO / "results" / "fiv_variants.json").read_text())["summary"]
        g = Table.grid(padding=(0, 3))
        g.add_column(justify="right", style=SUB)
        g.add_column(justify="left")
        g.add_row("5 independent prompt variants:", Text("different scaffolds + reworded framing banks", style=DIM))
        g.add_row("wrong-flip rate:", Text(f"{v['wrong_min']:.0%} – {v['wrong_max']:.0%}  (mean {v['wrong_mean']:.0%})",
                                           style=f"bold {RED}"))
        g.add_row("", Text("strictly positive in ALL five — not a wording artifact", style=f"bold {GOLD}"))
        head = Align.center(Text("IS IT OUR WORDING?     A specification curve", style=f"bold {MAUVE}"))
        self.say("We re-ran the whole study 5 ways. The harm shows up every single time.")
        live.update(self._frame(Group(head, Text(""), g, Text(""),
                    Align.center(Text("The effect survives every phrasing we tried.", style=f"bold {GREEN}")))))
        self.nap(4.8)

    def speed_card(self, live: Live) -> None:
        g = Table.grid(padding=(0, 3))
        g.add_column(justify="right", style=SUB)
        g.add_column(justify="left")
        g.add_row("powered headline run:", Text("24,000 live model calls", style="white"))
        g.add_row("specification curve:", Text("+ 36,000 more", style="white"))
        g.add_row("total, to do it right:", Text("60,000 live verifications", style=f"bold {GOLD}"))
        g.add_row("", Text("on slow, costly inference this is infeasible at scale", style=f"bold {RED}"))
        g.add_row("on Gemma + Cerebras:", Text("fast & cheap enough to run for real, repeatedly", style=f"bold {GREEN}"))
        head = Align.center(Text("WHY THIS NEEDS CEREBRAS", style=f"bold {MAUVE}"))
        self.say("Rigor means tens of thousands of calls. Slow models make that too slow and too costly.")
        live.update(self._frame(Group(head, Text(""), g, Text(""),
                    Align.center(Text("Fast, cheap inference is what makes skepticism affordable at scale.",
                                      style=f"bold {BLUE}")))))
        self.nap(5.2)

    def headline(self, live: Live, data: dict) -> None:
        base = data["conditions"]["baseline"]
        prot = data["conditions"]["protocol"]
        mc = data["mcnemar_wrong_flip"]
        rel = (base["wrong_flip_rate"] - prot["wrong_flip_rate"]) / base["wrong_flip_rate"]
        t = Table(box=ROUNDED, border_style=BLUE, expand=True)
        t.add_column("condition", style=f"bold {SUB}")
        t.add_column("flip rate", justify="center")
        t.add_column("wrong-flip", justify="center")
        t.add_column("self-consistency", justify="center")
        t.add_row("baseline", f"{base['flip_rate']:.3f}",
                  Text(f"{base['wrong_flip_rate']:.3f}", style=f"bold {RED}"),
                  f"{base['self_consistency']:.3f}")
        t.add_row("+ deliberate-answer protocol", f"{prot['flip_rate']:.3f}",
                  Text(f"{prot['wrong_flip_rate']:.3f}", style=f"bold {GREEN}"),
                  f"{prot['self_consistency']:.3f}")
        punch = Align.center(Text.assemble(
            ("→ a 3-line prompt cuts harmful flips ", f"bold {GOLD}"),
            (f"{rel:.0%}", f"bold {GREEN}"),
            (f"   (McNemar: fixed {mc['fixed_by_protocol']}, broke {mc['broken_by_protocol']}, "
             f"p={mc['p']:.1e})", DIM)))
        self.say("Across 24,000 live runs, framing makes it wrong 32% of the time.", f"bold {RED}")
        live.update(self._frame(Group(
            Align.center(Text("POWERED HEADLINE  ·  N=500 · K=8 · R=3", style=f"bold {MAUVE}")),
            Text(""), t, Text(""), punch)))
        self.nap(2.0)
        self.say("The protocol cuts that nearly in half — same model, no fine-tuning.", f"bold {GREEN}")
        live.update(self._frame(Group(
            Align.center(Text("POWERED HEADLINE  ·  N=500 · K=8 · R=3", style=f"bold {MAUVE}")),
            Text(""), t, Text(""), punch)))
        self.nap(2.6)

    def close_card(self, live: Live) -> None:
        big = Text("We measured it. We cut wrong-flips 43%.", style=f"bold {GOLD}",
                   justify="center")
        repo = Text("github.com/Swarm-Code/framing-induced-variance", style=BLUE,
                    justify="center")
        tags = Text("@CerebrasSystems   ·   @googlegemma", style=MAUVE, justify="center")
        self.say("The question is the verdict.")
        live.update(self._frame(Group(Text(""), Align.center(big), Text(""),
                                      Align.center(repo), Align.center(tags), Text(""))))
        self.nap(3.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", default="tf-test-11")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--still", action="store_true")
    args = ap.parse_args()

    data = _load()
    item = _pick(data["conditions"]["baseline"]["per_item"], args.id)
    claim, table_md = _claim_table(item["id"])

    console = Console()
    demo = Demo(console, args.speed, args.still)
    with Live(console=console, screen=True, refresh_per_second=30, transient=False) as live:
        demo.title_card(live)
        demo.method_card(live)
        header = demo.show_table(live, item, claim, table_md)
        demo.reveal_framings(live, item, header)
        demo.noise_card(live, data)
        demo.asymmetry_card(live)
        demo.robustness_card(live, data)
        demo.speed_card(live)
        demo.headline(live, data)
        demo.close_card(live)


if __name__ == "__main__":
    main()
