#!/usr/bin/env python3
"""Emit a full figure-grade CSV suite for the FIV paper from live result JSONs.
All pure stdlib so R reads plain CSV. Outputs under results/figdata/.
"""
import json
import os
from collections import defaultdict, Counter
from itertools import combinations

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(RES, "figdata")
os.makedirs(OUT, exist_ok=True)

FRAMINGS = ["neutral", "lead_entailed", "lead_refuted", "authority_entailed",
            "authority_refuted", "urgency_entailed", "doubt", "rhetorical_refuted"]
DECIDED = ("Entailed", "Refuted")


def w(name, header, rows):
    p = os.path.join(OUT, name)
    with open(p, "w") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")
    print("wrote", p, len(rows), "rows")


def main():
    scaled = json.load(open(os.path.join(RES, "fiv_tabfact_scaled.json")))
    base = scaled["conditions"]["baseline"]["per_item"]
    prot = scaled["conditions"]["protocol"]["per_item"]

    # 1) rates (baseline+protocol) with CIs --------------------------------
    rows = []
    for nm in ("baseline", "protocol"):
        c = scaled["conditions"][nm]
        flo, fhi = c["flip_ci"]; wlo, whi = c["wrong_flip_ci"]
        rows.append([nm, round(c["flip_rate"], 4), round(flo, 4), round(fhi, 4),
                     round(c["wrong_flip_rate"], 4), round(wlo, 4), round(whi, 4),
                     round(c["self_consistency"], 4)])
    w("rates.csv", ["condition", "flip", "flip_lo", "flip_hi",
                    "wrong", "wf_lo", "wf_hi", "sc"], rows)

    # 2) per-framing verdict composition (baseline): Entailed/Refuted/other,
    #    plus correct/wrong vs gold ----------------------------------------
    comp = {f: Counter() for f in FRAMINGS}
    corr = {f: Counter() for f in FRAMINGS}
    for it in base:
        g = it["gold"]
        for f, v in it["verdicts"].items():
            comp[f][v] += 1
            if v not in DECIDED:
                corr[f]["abstain"] += 1
            elif v == g:
                corr[f]["correct"] += 1
            else:
                corr[f]["wrong"] += 1
    n = len(base)
    rows = []
    for f in FRAMINGS:
        rows.append([f, round(corr[f]["correct"] / n, 4),
                     round(corr[f]["wrong"] / n, 4),
                     round(corr[f]["abstain"] / n, 4)])
    w("framing_composition.csv", ["framing", "correct", "wrong", "abstain"], rows)

    # 3) directional steering: signed net push toward Refuted per framing.
    #    For each item, +1 if framing says Refuted, -1 if Entailed, 0 other.
    #    Report mean signed verdict and net wrong-direction. -----------------
    rows = []
    for f in FRAMINGS:
        push = 0
        wrong_E = 0  # gold Entailed but framing said Refuted
        wrong_R = 0  # gold Refuted but framing said Entailed
        for it in base:
            v = it["verdicts"][f]; g = it["gold"]
            if v == "Refuted":
                push += 1
            elif v == "Entailed":
                push -= 1
            if g == "Entailed" and v == "Refuted":
                wrong_E += 1
            if g == "Refuted" and v == "Entailed":
                wrong_R += 1
        rows.append([f, round(push / n, 4),
                     round(wrong_E / n, 4), round(wrong_R / n, 4)])
    w("directional.csv", ["framing", "net_push_refuted",
                           "false_refute", "false_entail"], rows)

    # 4) per-item number of DISTINCT decided verdicts across framings ------
    #    1 = invariant; 2 = flips between E/R. Histogram for base & prot.
    rows = []
    for nm, data in (("baseline", base), ("protocol", prot)):
        hist = Counter()
        for it in data:
            verds = set(v for v in it["verdicts"].values() if v in DECIDED)
            # also count 'other' as its own bucket of disagreement
            allv = set(it["verdicts"].values())
            k = len(allv)
            hist[k] += 1
        for k in sorted(hist):
            rows.append([nm, k, hist[k], round(hist[k] / len(data), 4)])
    w("flip_hist.csv", ["condition", "n_distinct_verdicts", "count", "frac"], rows)

    # 5) 8x8 pairwise framing agreement matrix (baseline) ------------------
    rows = []
    for a in FRAMINGS:
        for b in FRAMINGS:
            agree = sum(1 for it in base if it["verdicts"][a] == it["verdicts"][b])
            rows.append([a, b, round(agree / n, 4)])
    w("agreement_matrix.csv", ["fa", "fb", "agree"], rows)

    # 6) McNemar contingency (wrong-flip status baseline vs protocol) ------
    mc = scaled["mcnemar_wrong_flip"]
    # build full 2x2: both-wrong, base-wrong-only(fixed), prot-wrong-only(broke), both-ok
    bw = {it["id"]: it["wrong_flip"] for it in base}
    pw = {it["id"]: it["wrong_flip"] for it in prot}
    both = b_only = p_only = neither = 0
    for i in bw:
        a, c = bw[i], pw.get(i, False)
        if a and c: both += 1
        elif a and not c: b_only += 1
        elif (not a) and c: p_only += 1
        else: neither += 1
    rows = [
        ["base_wrong", "prot_wrong", both],
        ["base_wrong", "prot_ok", b_only],
        ["base_ok", "prot_wrong", p_only],
        ["base_ok", "prot_ok", neither],
    ]
    w("mcnemar.csv", ["base", "prot", "count"], rows)
    print("McNemar fixed(b_only)=%d broke(p_only)=%d p=%.2e" %
          (b_only, p_only, mc["p"]))

    # 7) spec curve (variants) ---------------------------------------------
    variants = json.load(open(os.path.join(RES, "fiv_variants.json")))
    label = {"S1_meticulous+F1_original": "V1\noriginal",
             "S2_auditor+F2_colloquial": "V2\ncolloquial",
             "S3_terse+F3_formal": "V3\nformal",
             "S4_scientist+F4_inverted_order": "V4\ninverted",
             "S5_neutral_tool+F5_minimal": "V5\nminimal"}
    rows = []
    for k, v in variants["variants"].items():
        c = v["conditions"]; bb, pp = c["baseline"], c["protocol"]
        rows.append([label.get(k, k).replace("\n", " "),
                     round(bb["flip_rate"], 4), round(bb["wrong_flip_rate"], 4),
                     round(pp["flip_rate"], 4), round(pp["wrong_flip_rate"], 4),
                     round(bb["self_consistency"], 4)])
    w("spec_curve.csv", ["variant", "base_flip", "base_wrong",
                          "prot_flip", "prot_wrong", "sc"], rows)


if __name__ == "__main__":
    main()
