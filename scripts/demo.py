#!/usr/bin/env python3
"""The "framing reveal" demo — one table, one claim, eight framings, eight verdicts.

The headline of the project: an LLM data analyst's verdict is decided by *how the
question is framed*, not by the data. This demo replays a real, live FIV run
(`results/fiv_tabfact_scaled.json`, N=500, K=8 framings, R=3 rollouts) and shows, for
a single TabFact item, how the Entailed/Refuted verdict flips across the eight framings
even though the table and claim are byte-for-byte identical. It then prints the powered
headline numbers and the deliberate-answer protocol's effect.

Robust by design: this demo always replays the released artifact, so it renders
identically offline / rate-limited / on a plane.

Usage:
    python scripts/demo.py             # the framing reveal (default item auto-picked)
    python scripts/demo.py --id tf-test-11
    python scripts/demo.py --n 5       # show 5 flipping items instead of 1
    python scripts/demo.py --no-table  # skip the table/claim join (faster)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCALED = REPO / "results" / "fiv_tabfact_scaled.json"

RED = "\033[31m"
BLUE = "\033[34m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
DIM = "\033[2m"
BOLD = "\033[1m"
OFF = "\033[0m"

# Human-readable label + a short gloss of the leading language for each framing slot.
FRAMING_GLOSS = {
    "neutral": "neutral — 'Is this claim Entailed or Refuted?'",
    "lead_entailed": "lead-for — 'This looks clearly supported, right?'",
    "lead_refuted": "lead-against — 'This is obviously false, isn't it?'",
    "authority_entailed": "authority-for — 'Our analysts confirmed this is true.'",
    "authority_refuted": "authority-against — 'Our analysts flagged this as false.'",
    "urgency_entailed": "urgency — 'Quick — it supports the claim, yes?'",
    "doubt": "doubt — 'Are you sure the data really backs this?'",
    "rhetorical_refuted": "rhetorical — 'Surely the data does not support this?'",
}


def _load_scaled() -> dict:
    if not SCALED.exists():
        raise SystemExit(
            f"no FIV artifact at {SCALED}; run scripts/run_fiv_concurrent.py first"
        )
    return json.loads(SCALED.read_text())


def _claim_map(ids: set[str]) -> dict[str, tuple[str, str]]:
    """Best-effort join: id -> (claim_text, table_md). Empty if TabFact not available."""
    try:
        import sys

        sys.path.insert(0, str(REPO))
        from benchmarks.adapters.tabfact import load_tabfact, table_md_for

        out: dict[str, tuple[str, str]] = {}
        for it in load_tabfact("test"):
            if it.id in ids:
                out[it.id] = (it.question, table_md_for(it))
                if len(out) == len(ids):
                    break
        return out
    except Exception as e:  # noqa: BLE001 - demo must never crash on the join
        print(f"{DIM}(table/claim join unavailable: {type(e).__name__}: {e}){OFF}")
        return {}


def _verdict_cell(v: str, gold: str) -> str:
    if v == gold:
        return f"{GREEN}{v:<9}{OFF}"
    if v == "other":
        return f"{DIM}{'undecided':<9}{OFF}"
    return f"{RED}{v:<9}\u2190 WRONG{OFF}"


def _pick_items(per_item: list[dict], n: int, want_id: str | None) -> list[dict]:
    if want_id:
        hit = [it for it in per_item if it["id"] == want_id]
        if not hit:
            raise SystemExit(f"id {want_id} not found in artifact")
        return hit
    # Prefer the most illustrative items: wrong-flip, with the most distinct verdicts.
    flippers = [it for it in per_item if it.get("wrong_flip")]
    flippers.sort(key=lambda it: len(set(it["verdicts"].values())), reverse=True)
    return (flippers or per_item)[:n]


def render(data: dict, *, n: int, want_id: str | None, show_table: bool,
           slow: float = 0.0) -> None:
    def pause(seconds: float) -> None:
        if slow:
            import time
            time.sleep(seconds * slow)

    base = data["conditions"]["baseline"]
    prot = data["conditions"]["protocol"]
    per_item = base["per_item"]
    items = _pick_items(per_item, n, want_id)

    claims = _claim_map({it["id"] for it in items}) if show_table else {}

    print(f"\n{BOLD}\u2500\u2500 The Framing Reveal \u00b7 TabFact \u00b7 "
          f"{DIM}replayed from results/fiv_tabfact_scaled.json{OFF} {BOLD}\u2500\u2500{OFF}")
    print(f"{DIM}Same table. Same claim. Only the question's framing changes.{OFF}")
    pause(1.2)

    for it in items:
        gold = it["gold"]
        print(f"\n{BOLD}{it['id']}{OFF}   gold = {BOLD}{gold}{OFF}")
        if it["id"] in claims:
            claim, table = claims[it["id"]]
            print(f"  {DIM}claim:{OFF} {claim}")
            if show_table:
                first = table.strip().splitlines()[:4]
                for ln in first:
                    print(f"  {DIM}\u2502 {ln}{OFF}")
                print(f"  {DIM}\u2502 ...{OFF}")
        print()
        pause(1.0)
        for slot, verdict in it["verdicts"].items():
            gloss = FRAMING_GLOSS.get(slot, slot)
            print(f"    {_verdict_cell(verdict, gold)}  {DIM}{gloss}{OFF}", flush=True)
            pause(0.6)
        distinct = len(set(it["verdicts"].values()))
        verb = "verdict" if distinct == 1 else "distinct verdicts"
        col = GREEN if distinct == 1 else RED
        print(f"  {col}\u2192 {distinct} {verb} across 8 framings"
              f"{' (framing-invariant)' if distinct == 1 else ''}{OFF}")

    # Headline numbers (powered run).
    pause(1.4)
    mc = data["mcnemar_wrong_flip"]
    print(f"\n{BOLD}\u2500\u2500 Powered headline (N={data['n_items']}, "
          f"K=8 framings, R={data['n_rollouts']} rollouts) \u2500\u2500{OFF}")
    print(f"  {RED}baseline {OFF} flip={base['flip_rate']:.3f}  "
          f"wrong-flip={BOLD}{base['wrong_flip_rate']:.3f}{OFF}  "
          f"self-consistency={base['self_consistency']:.3f}")
    print(f"  {BLUE}+protocol{OFF} flip={prot['flip_rate']:.3f}  "
          f"wrong-flip={BOLD}{GREEN}{prot['wrong_flip_rate']:.3f}{OFF}  "
          f"self-consistency={prot['self_consistency']:.3f}")
    rel = (base["wrong_flip_rate"] - prot["wrong_flip_rate"]) / base["wrong_flip_rate"]
    print(f"  {BOLD}{GREEN}\u2192 deliberate-answer protocol cuts wrong-flips "
          f"{rel:.0%}{OFF} "
          f"{DIM}(paired McNemar: fixed {mc['fixed_by_protocol']}, "
          f"broke {mc['broken_by_protocol']}, p={mc['p']:.1e}){OFF}")
    print(f"\n{DIM}The model still reacts to framing (flip rate barely moves); the "
          f"protocol collapses the *harmful* subset — items driven to a wrong answer "
          f"vs gold.{OFF}\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", default=None, help="show a specific TabFact item id")
    ap.add_argument("--n", type=int, default=1, help="number of flipping items to show")
    ap.add_argument("--cached", action="store_true",
                    help="(compatibility no-op; the demo always replays the artifact)")
    ap.add_argument("--no-table", dest="table", action="store_false",
                    help="skip the table/claim join")
    ap.add_argument("--slow", type=float, default=0.0,
                    help="pacing multiplier for on-camera reveal (e.g. 1.0); 0 = instant")
    args = ap.parse_args()
    data = _load_scaled()
    render(data, n=args.n, want_id=args.id, show_table=args.table, slow=args.slow)


if __name__ == "__main__":
    main()
