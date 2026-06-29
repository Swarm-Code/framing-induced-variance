#!/usr/bin/env python3
"""FIV robustness sweep — run the experiment across 5 prompt/framing VARIANTS.

To rule out that the FIV effect is an artifact of our specific wording, we pair
each of 5 system/question scaffolds with a reworded framing bank (same 8 semantic
slots) and run a full FIV measurement for each variant. We report flip_rate and
wrong_flip_rate per variant with bootstrap CIs, then summarize the spread across
variants (a specification-curve / multiverse analysis). LIVE only.

Usage: python scripts/run_fiv_variants.py --n 150 --rollouts 3 --concurrency 200
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from benchmarks.adapters.tabfact import load_tabfact, table_md_for  # noqa: E402
from benchmarks.fiv_tabfact import verdict_of  # noqa: E402
from benchmarks.fiv_variants import FRAMING_BANKS, SCAFFOLDS  # noqa: E402
from benchmarks.stats import _binom_two_sided_p  # noqa: E402

UA = "Mozilla/5.0 (X11; Linux x86_64) python-requests/2.31"
PROTOCOL = ("\n\nAnswer protocol: reason briefly from the table only (do not anchor on any "
            "framing in the question), then commit on a new line exactly 'Final answer: "
            "Entailed' or 'Final answer: Refuted'.")

# Pair scaffold k with framing-bank k for variant V_k.
VARIANTS = list(zip(SCAFFOLDS.items(), FRAMING_BANKS.items()))


def _env():
    e = {}
    for line in (REPO / ".env").read_text().splitlines():
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            e[k.strip()] = v.strip().strip('"').strip("'")
    return e


def _providers(env):
    out = []
    if env.get("FIREWORKS_API_KEY"):
        fw = {"url": env.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1") + "/chat/completions",
              "key": env["FIREWORKS_API_KEY"], "model": env.get("FIREWORKS_MODEL")}
        out += [fw, fw, fw, fw]
    if env.get("CEREBRAS_API_KEY"):
        out.append({"url": env.get("CEREBRAS_BASE_URL", "https://api.cerebras.ai/v1") + "/chat/completions",
                    "key": env["CEREBRAS_API_KEY"], "model": env.get("MODEL_VISION", "gemma-4-31b")})
    if env.get("CEREBRAS_API_KEY_FALLBACK"):
        out.append({"url": env.get("CEREBRAS_BASE_URL", "https://api.cerebras.ai/v1") + "/chat/completions",
                    "key": env["CEREBRAS_API_KEY_FALLBACK"], "model": env.get("MODEL_VISION", "gemma-4-31b")})
    return out


def _balanced(n):
    items = load_tabfact("test")
    ent = [i for i in items if i.truth_answer == "Entailed"]
    ref = [i for i in items if i.truth_answer == "Refuted"]
    h = n // 2
    return ent[:h] + ref[: n - h]


def _messages(item, scaffold, framing_tmpl, deliberate):
    sysmsg = scaffold["sys"] + (PROTOCOL if deliberate else "")
    user = (scaffold["head"].format(table=table_md_for(item))
            + framing_tmpl.format(claim=item.question) + scaffold["tail"])
    return [{"role": "system", "content": sysmsg}, {"role": "user", "content": user}]


async def _call(client, provs, rr, messages, sem):
    async with sem:
        for attempt in range(6):
            prov = provs[rr[0] % len(provs)]; rr[0] += 1
            try:
                r = await client.post(prov["url"],
                    headers={"Authorization": f"Bearer {prov['key']}", "Content-Type": "application/json", "User-Agent": UA},
                    json={"model": prov["model"], "max_tokens": 4096, "messages": messages}, timeout=240.0)
                if r.status_code in (429, 500, 502, 503, 529):
                    await asyncio.sleep(min(1.0 * (attempt + 1), 5.0)); continue
                r.raise_for_status()
                m = r.json()["choices"][0]["message"]
                return m.get("content") or m.get("reasoning_content") or ""
            except Exception:
                await asyncio.sleep(min(1.0 * (attempt + 1), 5.0))
        return ""


async def _run(items, scaffold, bank, deliberate, provs, conc, rollouts, tag):
    sem = asyncio.Semaphore(conc); rr = [0]
    limits = httpx.Limits(max_connections=conc + 32, max_keepalive_connections=conc + 32, keepalive_expiry=30.0)
    timeout = httpx.Timeout(connect=15.0, read=240.0, write=30.0, pool=240.0)
    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        tasks = {}
        for it in items:
            for fl, ft in bank:
                for rep in range(rollouts):
                    tasks[(it.id, fl, rep)] = asyncio.ensure_future(
                        _call(client, provs, rr, _messages(it, scaffold, ft, deliberate), sem))
        done = 0; t0 = time.time(); total = len(tasks)
        for fut in asyncio.as_completed(list(tasks.values())):
            await fut; done += 1
            if done % 500 == 0:
                print(f"    [{tag}] {done}/{total} ({done/(time.time()-t0):.1f}/s)", flush=True)
        return {k: verdict_of(v.result()) for k, v in tasks.items()}


def _aggregate(items, bank, results, rollouts):
    per = []; consist = []
    slots = [l for l, _ in bank]
    for it in items:
        verdicts = {}
        for fl in slots:
            reps = [results[(it.id, fl, r)] for r in range(rollouts)]
            c = Counter(reps); modal, mn = c.most_common(1)[0]
            verdicts[fl] = modal; consist.append(mn / len(reps))
        inv = len(set(verdicts.values())) == 1
        wrong = any(v in ("Entailed", "Refuted") and v != it.truth_answer for v in verdicts.values())
        per.append({"id": it.id, "gold": it.truth_answer, "invariant": inv, "wrong_flip": wrong, "verdicts": verdicts})
    n = len(items)
    return {"flip_rate": sum(0 if p["invariant"] else 1 for p in per) / n,
            "wrong_flip_rate": sum(1 for p in per if p["wrong_flip"]) / n,
            "self_consistency": sum(consist) / len(consist), "per_item": per}


def _ci(vals, n_boot=2000, seed=0):
    n = len(vals)
    if not n:
        return [0.0, 0.0]
    rng = random.Random(seed)
    b = sorted(sum(vals[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return [b[int(0.025 * n_boot)], b[int(0.975 * n_boot) - 1]]


async def main_async(args):
    env = _env(); provs = _providers(env)
    if not provs:
        raise SystemExit("ABORT: no live provider key")
    items = _balanced(args.n)
    print(f"LIVE FIV VARIANT SWEEP: {len(VARIANTS)} variants x n={len(items)} x K=8 x R={args.rollouts} "
          f"x2 conditions. providers={len(provs)} conc={args.concurrency}", flush=True)
    out = {"n_items": len(items), "n_rollouts": args.rollouts, "variants": {}}
    # Resume: keep variants already completed in an existing results file.
    existing = Path("results/fiv_variants.json")
    if existing.exists():
        try:
            prev = json.loads(existing.read_text())
            if prev.get("n_items") == len(items):
                out["variants"] = prev.get("variants", {})
                if out["variants"]:
                    print(f"RESUME: {len(out['variants'])} variant(s) already done: "
                          f"{list(out['variants'])}", flush=True)
        except Exception:
            pass
    for (sname, scaffold), (fname, bank) in VARIANTS:
        vid = f"{sname}+{fname}"
        if vid in out["variants"]:
            print(f"=== SKIP {vid} (already collected) ===", flush=True)
            continue
        print(f"\n=== VARIANT {vid} ===", flush=True)
        vres = {"scaffold": sname, "framing_bank": fname, "conditions": {}}
        per_cond = {}
        for cond, delib in [("baseline", False), ("protocol", True)]:
            t0 = time.time()
            res = await _run(items, scaffold, bank, delib, provs, args.concurrency, args.rollouts, f"{vid}/{cond}")
            agg = _aggregate(items, bank, res, args.rollouts)
            agg["flip_ci"] = _ci([0 if p["invariant"] else 1 for p in agg["per_item"]])
            agg["wrong_flip_ci"] = _ci([1 if p["wrong_flip"] else 0 for p in agg["per_item"]])
            agg["seconds"] = time.time() - t0
            vres["conditions"][cond] = agg
            per_cond[cond] = agg["per_item"]
            print(f"  [{cond}] flip={agg['flip_rate']:.3f} {agg['flip_ci']}  "
                  f"wrong_flip={agg['wrong_flip_rate']:.3f} {agg['wrong_flip_ci']}  "
                  f"sc={agg['self_consistency']:.3f}", flush=True)
        b = sum(1 for rb, rp in zip(per_cond["baseline"], per_cond["protocol"]) if rb["wrong_flip"] and not rp["wrong_flip"])
        c = sum(1 for rb, rp in zip(per_cond["baseline"], per_cond["protocol"]) if not rb["wrong_flip"] and rp["wrong_flip"])
        vres["mcnemar"] = {"fixed": b, "broken": c, "p": _binom_two_sided_p(b, c), "significant": _binom_two_sided_p(b, c) < 0.05}
        out["variants"][vid] = vres
        Path("results").mkdir(exist_ok=True)
        Path("results/fiv_variants.json").write_text(json.dumps(out, indent=2))  # incremental save
        print(f"  saved (variant {vid} done)", flush=True)

    # cross-variant summary
    bl = [v["conditions"]["baseline"]["flip_rate"] for v in out["variants"].values()]
    wf = [v["conditions"]["baseline"]["wrong_flip_rate"] for v in out["variants"].values()]
    out["summary"] = {"flip_min": min(bl), "flip_max": max(bl), "flip_mean": sum(bl) / len(bl),
                      "wrong_min": min(wf), "wrong_max": max(wf), "wrong_mean": sum(wf) / len(wf)}
    Path("results/fiv_variants.json").write_text(json.dumps(out, indent=2))
    print(f"\n=== CROSS-VARIANT SUMMARY ===\n  flip_rate range [{min(bl):.3f},{max(bl):.3f}] mean {sum(bl)/len(bl):.3f}")
    print(f"  wrong_flip range [{min(wf):.3f},{max(wf):.3f}] mean {sum(wf)/len(wf):.3f}")
    print("  wrote results/fiv_variants.json", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--rollouts", type=int, default=3)
    ap.add_argument("--concurrency", type=int, default=200)
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
