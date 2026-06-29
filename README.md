# The Question Is the Verdict

**Framing-Induced Variance (FIV): how the *wording* of a question — not the data — decides what an LLM data analyst concludes.** Built on Pydantic AI, evaluated live on Google **Gemma 4 31B** via **Cerebras** inference. For the Cerebras × Google Gemma 4 hackathon.

**Authors:** Luis Alejandro Rincon (SwarmCode) · Ned Dana (AMD)

---

## The one-line thesis

A data agent's biggest risk isn't bad arithmetic — it's that **the same table and the same claim produce a different verdict depending on how the question is framed.** Lead the model toward "this is obviously false, isn't it?" and it refutes; ask neutrally and it entails. We name and measure this effect — **Framing-Induced Variance (FIV)** — on a binary, grey-area, information-invariant task, prove it is framing-driven (not sampling noise), and show a minimal three-line prompt protocol that significantly reduces the *harmful* subset.

## What we measure

We hold the **table and claim byte-for-byte identical** and only change the question's framing across **K = 8** leading variants (neutral, lead-for/against, authority-for/against, urgency, doubt, rhetorical). Gold labels are binary (Entailed / Refuted) so every flip is scorable as right or wrong.

- **flip rate** — fraction of items whose verdict is not constant across the 8 framings.
- **wrong-flip rate** *(headline)* — fraction of items where *some* framing drives the model to a demonstrably wrong answer vs gold.
- **self-consistency (SC)** — within-framing agreement across rollouts; the noise floor is `1 − SC`. We only credit framing for an effect when `flip rate ≫ 1 − SC`.

## Measured results (live Gemma 4 31B on Cerebras)

Powered run: **N = 500** balanced TabFact items × **K = 8** framings × **R = 3** rollouts × 2 conditions = **24,000 live calls**.

| Condition | Flip rate [95% CI] | Wrong-flip [95% CI] | Self-consistency |
|---|---|---|---|
| baseline | 0.526 [0.484, 0.568] | **0.320** [0.280, 0.360] | 0.880 |
| **+ deliberate-answer protocol** | 0.512 [0.468, 0.560] | **0.184** [0.152, 0.218] | 0.867 |

- **Framing, not noise.** SC = 0.880 → noise floor `1 − SC = 0.120`. The observed flip rate 0.526 exceeds it by **>4×**: verdict changes are framing-driven.
- **The protocol works, with power.** Wrong-flips fall **0.320 → 0.184 (−43% relative)**. Paired McNemar exact test: fixed **105**, broke **37**, **p = 9.9 × 10⁻⁹** (significant). An earlier N = 200 pilot was underpowered (p = 0.46) — we report both to make the power dependence explicit.

**Robustness to prompt wording (specification curve).** We re-ran the full experiment as **5 independent variants**, each pairing a distinct system scaffold with a reworded framing bank (same 8 semantic slots). The headline survives every variant: **wrong-flip rate is strictly positive in all five** (range 0.113–0.313, mean 0.197). The raw flip rate is wording-sensitive (range 0.127–0.920), which is exactly the specification sensitivity the analysis is designed to surface — and why we treat the more stable **wrong-flip** rate as the headline.

Artifacts: `results/fiv_tabfact_scaled.json` (N=500), `results/fiv_variants.json` (5-variant sweep). Stats in `benchmarks/stats.py` (bootstrap CIs, McNemar exact, Cohen's h). Full write-up in `paper/fiv_report.pdf`.

## The deliberate-answer protocol

A three-line addition to the system prompt — no tools, no fine-tune:

> reason from the data; commit a single normalized `Final answer: <value>`; say `insufficient data` if genuinely uncertain.

It does not stop the model reacting to framing (the flip rate barely moves); it collapses the *harmful* subset — the items framing drives to a wrong answer vs gold.

## Why Cerebras matters

FIV is a high-throughput measurement: every reported number is the aggregate of tens of thousands of live calls (24,000 for the headline; +36,000 for the specification curve). Fast inference is what makes a rigorous, properly-powered framing audit affordable to run end-to-end rather than projected from a tiny slice.

## Quickstart

```bash
# offline (deterministic stub) — full test suite, no key needed
python3 -m pytest -q

# the demo — replays the real FIV run (offline-safe, always renders)
python3 scripts/demo.py                 # the framing reveal (auto-picks a flipping item)
python3 scripts/demo.py --id tf-test-11 # a specific item, with its table + claim
python3 scripts/demo.py --n 5           # show 5 flipping items
```

Live config (`.env`): `CEREBRAS_API_KEY`, `CEREBRAS_BASE_URL`, `MODEL_TEXT`, `MODEL_VISION` (map to Gemma 4 `gemma-4-31b`).

## Reproduce the headline run

```bash
# pulls full TabFact, runs the powered FIV grid live, writes results/fiv_tabfact_scaled.json
python3 scripts/run_fiv_concurrent.py --n 500 --rollouts 3

# the 5-variant specification curve → results/fiv_variants.json
python3 scripts/run_fiv_variants.py

# regenerate every figure in the paper from the released artifacts
cd paper && Rscript fiv_figures.R && pdflatex fiv_report.tex
```

## Repo map

```
src/multivac/        the Pydantic-AI harness (provider, harness.chat, config, deliberate_answer flag)
benchmarks/          fiv_tabfact (FIV runner), fiv_variants (5 scaffolds × 5 banks),
                     stats (bootstrap CI, McNemar, Cohen's h), adapters/{tabfact, external}
data/tabfact/        full TabFact (train/val/test = 117,854 rows, binary gold)
results/             fiv_tabfact_scaled.json (N=500), fiv_variants.json (5-variant sweep)
paper/               fiv_report.tex (arXiv-style report) + fiv_figures.R (CSV→figures)
scripts/demo.py      the framing-reveal demo (replays the real artifact)
```

## Notes

- **FIV's information invariance** is the design that rules out the obvious confound: every framing shows the *identical* table and claim, so any verdict change is attributable to wording alone.
- **TabFact** was chosen deliberately: binary gold + grey-area refuted claims that need multi-step aggregation — the regime where framing can actually push a verdict across the decision boundary.
- All reported numbers are **live**; the runner refuses to execute offline so no result can come from a stub.
