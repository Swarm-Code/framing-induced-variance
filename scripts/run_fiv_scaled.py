#!/usr/bin/env python3
"""Scaled live FIV on TabFact: baseline vs +deliberate protocol, with statistics.

Runs FIV (same table+claim, K leading framings, R rollouts) over N items for two
system-prompt conditions and reports flip_rate / wrong_flip_rate with bootstrap CIs
plus a paired McNemar test on per-item wrong-flip. Designed to run thousands of
live executions; writes results/fiv_tabfact_scaled.json.

Usage: python scripts/run_fiv_scaled.py --n 200 --rollouts 3
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from benchmarks.adapters.tabfact import load_tabfact  # noqa: E402
from benchmarks.fiv_tabfact import run_fiv  # noqa: E402
from benchmarks.stats import _binom_two_sided_p  # noqa: E402


def _ci(values: list[int], *, n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0
    point = sum(values) / n
    rng = random.Random(seed)
    boots = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return point, boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1]


def _balanced_sample(n: int):
    items = load_tabfact("test")
    ent = [i for i in items if i.truth_answer == "Entailed"]
    ref = [i for i in items if i.truth_answer == "Refuted"]
    half = n // 2
    return ent[:half] + ref[: n - half]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--rollouts", type=int, default=3)
    args = ap.parse_args()

    from multivac import Multivac, Settings

    s = Settings.load()
    if not s.is_live:
        raise SystemExit("ABORT: not live. This experiment runs LIVE only — no "
                         "offline/stub fallback, to guarantee no fabricated data.")
    s = s.model_copy(update={"model": s.vision_model})
    items = _balanced_sample(args.n)
    print(f"LIVE FIV scaled: model={s.model} n={len(items)} K=8 R={args.rollouts} "
          f"(~{len(items) * 8 * args.rollouts * 2} live calls)", flush=True)

    def make_ask(deliberate: bool):
        h = Multivac(s, system_prompt="You are a meticulous data analyst.",
                     deliberate_answer=deliberate)

        def ask(prompt: str) -> str:
            h.reset()
            return h.chat(prompt).output

        return ask

    def progress(i, total, rec):
        flag = "WRONG-FLIP" if rec.flipped_to_wrong else ("flip" if not rec.invariant else "stable")
        print(f"    {i:>4}/{total}  {rec.item_id:<14} gold={rec.gold:<8} {flag}", flush=True)

    out = {"n_items": len(items), "n_rollouts": args.rollouts, "model": s.model,
           "conditions": {}}
    per_condition = {}
    for cond, delib in [("baseline", False), ("protocol", True)]:
        print(f"\n=== CONDITION: {cond} (deliberate={delib}) ===", flush=True)
        t0 = time.time()
        per, rep = run_fiv(items, make_ask(delib), n_rollouts=args.rollouts, progress=progress)
        dt = time.time() - t0
        per_condition[cond] = per
        wrong = [1 if r.flipped_to_wrong else 0 for r in per]
        flip = [0 if r.invariant else 1 for r in per]
        wf = _ci(wrong)
        fr = _ci(flip)
        out["conditions"][cond] = {
            "flip_rate": rep.flip_rate, "flip_ci": [fr[1], fr[2]],
            "wrong_flip_rate": rep.wrong_flip_rate, "wrong_flip_ci": [wf[1], wf[2]],
            "self_consistency": rep.self_consistency, "mean_accuracy": rep.mean_accuracy,
            "seconds": dt,
            "per_item": [{"id": r.item_id, "gold": r.gold, "invariant": r.invariant,
                          "wrong_flip": r.flipped_to_wrong, "verdicts": r.framing_verdicts}
                         for r in per],
        }
        print(f"  [{cond}] flip={rep.flip_rate:.3f} {fr[1:]}  wrong_flip={rep.wrong_flip_rate:.3f} "
              f"{wf[1:]}  acc={rep.mean_accuracy:.3f}  sc={rep.self_consistency:.3f}  ({dt:.0f}s)",
              flush=True)

    # paired McNemar on wrong-flip (does the protocol reduce wrong flips?)
    b = sum(1 for rb, rp in zip(per_condition["baseline"], per_condition["protocol"])
            if rb.flipped_to_wrong and not rp.flipped_to_wrong)  # fixed by protocol
    c = sum(1 for rb, rp in zip(per_condition["baseline"], per_condition["protocol"])
            if not rb.flipped_to_wrong and rp.flipped_to_wrong)  # broken by protocol
    p = _binom_two_sided_p(b, c)
    out["mcnemar_wrong_flip"] = {"fixed_by_protocol": b, "broken_by_protocol": c,
                                 "p": p, "significant": p < 0.05}
    print(f"\n  McNemar wrong-flip: fixed={b} broken={c} p={p:.4g} "
          f"{'SIGNIFICANT' if p < 0.05 else 'n.s.'}", flush=True)

    Path("results").mkdir(exist_ok=True)
    Path("results/fiv_tabfact_scaled.json").write_text(json.dumps(out, indent=2))
    print("  wrote results/fiv_tabfact_scaled.json", flush=True)


if __name__ == "__main__":
    main()
