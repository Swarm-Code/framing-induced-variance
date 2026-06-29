# Submission — The Question Is the Verdict (Framing-Induced Variance)

Cerebras × Google Gemma 4 Hackathon · Gemma 4 31B on Cerebras · Pydantic AI

**Authors:** Luis Alejandro Rincon (SwarmCode) · Ned Dana (AMD)

## What we built

A live, properly-powered measurement of **Framing-Induced Variance (FIV)**: holding a table and claim **byte-for-byte identical**, we change only the *framing* of the question across 8 leading variants and show the model's Entailed/Refuted verdict flips — often to a demonstrably wrong answer vs gold. We then ship a minimal three-line **deliberate-answer protocol** that significantly reduces the harmful subset. Same model, same data — we changed the *question wording* and the *answer-commitment behavior*, nothing else.

## Measured results (live, not projected)

Powered run: **N = 500** TabFact items × **K = 8** framings × **R = 3** rollouts × 2 conditions = **24,000 live calls** on Gemma 4 31B @ Cerebras.

| Condition | Flip rate | Wrong-flip | SC |
|---|---|---|---|
| baseline | 0.526 | **0.320** | 0.880 |
| + protocol | 0.512 | **0.184** | 0.867 |

- Wrong-flips **0.320 → 0.184 (−43%)**, paired McNemar fixed 105 / broke 37, **p = 9.9 × 10⁻⁹**.
- Framing, not noise: flip rate 0.526 ≫ noise floor `1 − SC = 0.120` (>4×).
- Robust across **5 prompt-specification variants**: wrong-flip strictly positive in all five (0.113–0.313, mean 0.197).

Artifacts: `results/fiv_tabfact_scaled.json`, `results/fiv_variants.json`, `paper/fiv_report.pdf`, figures in `paper/figs/`.

## Tracks targeted

1. **Enterprise impact** — PRIMARY. Trustworthy analytics: a deployed "AI data analyst" that silently flips its verdict on the wording of a stakeholder's question is a liability. We quantify the harm and ship a free mitigation.
2. **Best use of Gemma 4 on Cerebras** — a rigorous, high-throughput evaluation (24k + 36k live calls) that only an inference platform this fast makes affordable to run end-to-end.
3. **People's Choice (social)** — the "framing reveal" clip + thread below.

## The 60-second "framing reveal" clip (shot list)

1. **0–8s — hook.** Full-screen a TabFact table + a single claim. VO: *"Same data. Same claim. Watch the AI's verdict change — just because we reword the question."*
2. **8–22s — the reveal.** Run `python scripts/demo.py --id tf-test-11`. The 8 framings scroll: neutral/lead-for/authority-for say **Entailed** (green); the moment the framing turns to doubt/urgency/authority-against the verdict flips to **Refuted ← WRONG** (red). Same table on screen the whole time.
3. **22–38s — why it matters.** VO: *"Nothing about the data changed. Only the question's tone. On 500 items, framing drives the model to a wrong answer 32% of the time."*
4. **38–52s — the fix.** Bring up the headline panel: wrong-flip **0.320 → 0.184**, McNemar **p = 9.9e-9**. VO: *"A three-line prompt — reason from the data, commit one final answer — cuts the harmful flips 43%. No fine-tuning. Same Gemma 4, same tools."*
5. **52–60s — close.** Title card: **"The question is the verdict."** repo + `@CerebrasSystems @googlegemma`.

The demo replays `results/fiv_tabfact_scaled.json`, so it renders identically offline — a live rate-limit can't break the take.

## X thread copy

> 1/ Every AI "data analyst" has a hidden bug: change *how you ask* and it changes *what it concludes* — on the same data. We measured it on 24,000 live calls and named it Framing-Induced Variance. 🧵
>
> 2/ The setup: same table, same claim, byte-for-byte. We only reword the question 8 ways (neutral → "obviously false, isn't it?"). Gold labels are binary, so every flipped verdict is scorable as right or wrong.
>
> 3/ Result on live Gemma 4 31B @Cerebras: on 53% of items the verdict isn't stable across framings, and on **32%** some framing drives the model to a demonstrably WRONG answer vs gold. It's framing, not sampling noise (>4× the noise floor).
>
> 4/ The fix: a three-line "deliberate-answer" protocol — reason from the data, commit one final answer. Harmful flips drop **32% → 18%**, paired McNemar **p = 9.9e-9**. No fine-tuning. And it holds across 5 reworded prompt variants.
>
> 5/ Why @CerebrasSystems matters: a properly-powered framing audit is 24k+ live calls. Fast inference is what makes rigor affordable instead of a tiny underpowered slice. Built on Pydantic AI + @googlegemma Gemma 4. Repo + 60s demo 👇 #Gemma4 #Cerebras

## Submission checklist

- [x] Powered live run captured: `results/fiv_tabfact_scaled.json` (N=500, 24k calls)
- [x] Robustness sweep captured: `results/fiv_variants.json` (5 variants)
- [x] arXiv-style report compiles: `paper/fiv_report.pdf` (16 pages, figures from real artifacts)
- [x] Demo replays the real artifact offline: `scripts/demo.py`
- [x] README with thesis, results table, quickstart, reproduce, full attribution
- [ ] Record 60s clip (against the demo)
- [ ] Post X thread, tag @CerebrasSystems + @googlegemma
- [ ] Submit repo link

## Reproduce

```bash
python3 -m pytest -q                      # offline test suite
python3 scripts/demo.py --id tf-test-11   # the framing reveal, offline
# live powered run (needs CEREBRAS_API_KEY in .env):
python3 scripts/run_fiv_concurrent.py --n 500 --rollouts 3
```
