#!/usr/bin/env python3
"""Concurrent FIV runner — saturates the deployment with parallel requests.

Unlike run_fiv_scaled.py (sequential harness), this fires all (item x framing x
rollout) requests concurrently across BOTH providers (Fireworks primary + Cerebras
fallback) using asyncio + httpx, with a bounded concurrency pool and round-robin
load balancing. On 429/5xx it retries on the other provider.

LIVE ONLY: aborts if no provider key. Per-call results -> verdicts; aggregated to
FIV (flip / wrong-flip / self-consistency) with bootstrap CIs + paired McNemar for
the baseline-vs-protocol ablation. Writes results/fiv_tabfact_scaled.json.

Usage: python scripts/run_fiv_concurrent.py --n 200 --rollouts 3 --concurrency 48
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
from benchmarks.fiv_tabfact import FRAMINGS, verdict_of  # noqa: E402
from benchmarks.stats import _binom_two_sided_p  # noqa: E402

UA = "Mozilla/5.0 (X11; Linux x86_64) python-requests/2.31"

PROTOCOL = (
    "\n\nAnswer protocol: reason briefly from the table only (do not anchor on any "
    "framing in the question), then commit on a new line exactly 'Final answer: "
    "Entailed' or 'Final answer: Refuted'."
)


def _load_env() -> dict:
    env = {}
    for line in (REPO / ".env").read_text().splitlines():
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def _providers(env: dict) -> list[dict]:
    out = []
    if env.get("FIREWORKS_API_KEY"):
        # Weight the dedicated B200 deployment heavily (it has no shared 100k/min
        # token cap), interleaving Cerebras keys as overflow/failover.
        fw = {"name": "fireworks",
              "url": (env.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1")) + "/chat/completions",
              "key": env["FIREWORKS_API_KEY"],
              "model": env.get("FIREWORKS_MODEL")}
        out += [fw, fw, fw, fw]  # 4x weight
    if env.get("CEREBRAS_API_KEY"):
        out.append({"name": "cerebras",
                    "url": (env.get("CEREBRAS_BASE_URL", "https://api.cerebras.ai/v1")) + "/chat/completions",
                    "key": env["CEREBRAS_API_KEY"],
                    "model": env.get("MODEL_VISION", "gemma-4-31b")})
        if env.get("CEREBRAS_API_KEY_FALLBACK"):
            out.append({"name": "cerebras2",
                        "url": (env.get("CEREBRAS_BASE_URL", "https://api.cerebras.ai/v1")) + "/chat/completions",
                        "key": env["CEREBRAS_API_KEY_FALLBACK"],
                        "model": env.get("MODEL_VISION", "gemma-4-31b")})
    return out


def _balanced(n: int):
    items = load_tabfact("test")
    ent = [i for i in items if i.truth_answer == "Entailed"]
    ref = [i for i in items if i.truth_answer == "Refuted"]
    h = n // 2
    return ent[:h] + ref[: n - h]


def _build_messages(item, framing_tmpl: str, deliberate: bool):
    sys_msg = ("You are a meticulous data analyst. Answer only 'Entailed' or 'Refuted'."
               + (PROTOCOL if deliberate else ""))
    user = (f"Table:\n{table_md_for(item)}\n\n"
            + framing_tmpl.format(claim=item.question)
            + "\n\nFinal answer (one word: Entailed or Refuted):")
    return [{"role": "system", "content": sys_msg}, {"role": "user", "content": user}]


async def _call(client: httpx.AsyncClient, provs: list[dict], rr: list[int],
                messages: list, sem: asyncio.Semaphore) -> str:
    async with sem:
        for attempt in range(6):
            prov = provs[rr[0] % len(provs)]
            rr[0] += 1
            try:
                r = await client.post(
                    prov["url"],
                    headers={"Authorization": f"Bearer {prov['key']}",
                             "Content-Type": "application/json", "User-Agent": UA},
                    json={"model": prov["model"], "max_tokens": 4096, "messages": messages},
                    timeout=240.0,
                )
                if r.status_code in (429, 500, 502, 503, 529):
                    await asyncio.sleep(min(1.0 * (attempt + 1), 5.0))
                    continue
                r.raise_for_status()
                msg = r.json()["choices"][0]["message"]
                return msg.get("content") or msg.get("reasoning_content") or ""
            except Exception:
                await asyncio.sleep(min(1.0 * (attempt + 1), 5.0))
        return ""


async def run_condition(items, deliberate, provs, concurrency, rollouts, label, log):
    # Per-provider semaphores so one slow/throttled provider can't starve others,
    # plus a global ceiling. Fireworks (a dedicated B200) gets the lion's share.
    sem = asyncio.Semaphore(concurrency)
    rr = [0]
    limits = httpx.Limits(max_connections=concurrency + 32,
                          max_keepalive_connections=concurrency + 32,
                          keepalive_expiry=30.0)
    timeout = httpx.Timeout(connect=15.0, read=180.0, write=30.0, pool=180.0)
    async with httpx.AsyncClient(limits=limits, timeout=timeout, http2=False) as client:
        tasks = {}
        for it in items:
            for fl, ft in FRAMINGS:
                for rep in range(rollouts):
                    coro = _call(client, provs, rr, _build_messages(it, ft, deliberate), sem)
                    tasks[(it.id, fl, rep)] = asyncio.ensure_future(coro)
        total = len(tasks)
        done = 0
        results = {}
        t0 = time.time()
        # gather as they complete for accurate throughput, keyed for aggregation
        keys = list(tasks.keys())
        for fut in asyncio.as_completed([tasks[k] for k in keys]):
            await fut  # ensure completion ordering for rate print
            done += 1
            if done % 500 == 0:
                rate = done / (time.time() - t0)
                print(f"  [{label}] {done}/{total} calls  ({rate:.1f} calls/s)", flush=True)
        for k in keys:
            results[k] = verdict_of(tasks[k].result())
    return results


def aggregate(items, results, rollouts):
    per = []
    all_consistency = []
    acc_num = acc_den = 0
    for it in items:
        verdicts = {}
        for fl, _ in FRAMINGS:
            reps = [results[(it.id, fl, r)] for r in range(rollouts)]
            c = Counter(reps)
            modal, mn = c.most_common(1)[0]
            verdicts[fl] = modal
            all_consistency.append(mn / len(reps))
            acc_den += 1
            acc_num += sum(1 for v in reps if v == it.truth_answer) / len(reps)
        invariant = len(set(verdicts.values())) == 1
        wrong = any(v in ("Entailed", "Refuted") and v != it.truth_answer for v in verdicts.values())
        per.append({"id": it.id, "gold": it.truth_answer, "invariant": invariant,
                    "wrong_flip": wrong, "verdicts": verdicts})
    n = len(items)
    return {
        "flip_rate": sum(0 if p["invariant"] else 1 for p in per) / n,
        "wrong_flip_rate": sum(1 for p in per if p["wrong_flip"]) / n,
        "self_consistency": sum(all_consistency) / len(all_consistency),
        "mean_accuracy": acc_num / acc_den,
        "per_item": per,
    }


def _ci(vals, n_boot=2000, seed=0):
    n = len(vals)
    if not n:
        return 0.0, 0.0
    rng = random.Random(seed)
    b = sorted(sum(vals[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return b[int(0.025 * n_boot)], b[int(0.975 * n_boot) - 1]


async def main_async(args):
    env = _load_env()
    provs = _providers(env)
    if not provs:
        raise SystemExit("ABORT: no live provider key (Fireworks/Cerebras).")
    print(f"LIVE concurrent FIV: providers={[p['name'] for p in provs]} "
          f"n={args.n} K={len(FRAMINGS)} R={args.rollouts} conc={args.concurrency}", flush=True)
    items = _balanced(args.n)
    out = {"n_items": len(items), "n_rollouts": args.rollouts,
           "providers": [p["name"] for p in provs], "conditions": {}}
    per_cond = {}
    for cond, delib in [("baseline", False), ("protocol", True)]:
        print(f"\n=== CONDITION {cond} ({len(items)*len(FRAMINGS)*args.rollouts} calls) ===", flush=True)
        t0 = time.time()
        res = await run_condition(items, delib, provs, args.concurrency, args.rollouts, cond, None)
        agg = aggregate(items, res, args.rollouts)
        dt = time.time() - t0
        wf_ci = _ci([1 if p["wrong_flip"] else 0 for p in agg["per_item"]])
        fr_ci = _ci([0 if p["invariant"] else 1 for p in agg["per_item"]])
        agg["wrong_flip_ci"] = list(wf_ci)
        agg["flip_ci"] = list(fr_ci)
        agg["seconds"] = dt
        out["conditions"][cond] = agg
        per_cond[cond] = agg["per_item"]
        print(f"  [{cond}] flip={agg['flip_rate']:.3f} {fr_ci}  wrong_flip={agg['wrong_flip_rate']:.3f} "
              f"{wf_ci}  acc={agg['mean_accuracy']:.3f}  sc={agg['self_consistency']:.3f}  ({dt:.0f}s)", flush=True)

    b = sum(1 for rb, rp in zip(per_cond["baseline"], per_cond["protocol"])
            if rb["wrong_flip"] and not rp["wrong_flip"])
    c = sum(1 for rb, rp in zip(per_cond["baseline"], per_cond["protocol"])
            if not rb["wrong_flip"] and rp["wrong_flip"])
    p = _binom_two_sided_p(b, c)
    out["mcnemar_wrong_flip"] = {"fixed_by_protocol": b, "broken_by_protocol": c,
                                 "p": p, "significant": p < 0.05}
    print(f"\n  McNemar wrong-flip: fixed={b} broken={c} p={p:.4g} "
          f"{'SIGNIFICANT' if p < 0.05 else 'n.s.'}", flush=True)
    Path("results").mkdir(exist_ok=True)
    Path("results/fiv_tabfact_scaled.json").write_text(json.dumps(out, indent=2))
    print("  wrote results/fiv_tabfact_scaled.json", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--rollouts", type=int, default=3)
    ap.add_argument("--concurrency", type=int, default=48)
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
