#!/usr/bin/env bash
# Clean the repo of build/agent garbage before publishing.
# Idempotent: safe to run multiple times. Operates on the working tree only;
# git history is handled separately by the publish step.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== removing agent snapshots, debug artifacts, IDE + build cruft =="
rm -rf .swarm/snapshots .swarm/skills 2>/dev/null || true
rm -rf .idea .multivac .pytest_cache 2>/dev/null || true
rm -rf debug 2>/dev/null || true
rm -f swarmos_debug.log loop_tracker.txt 2>/dev/null || true

# discarded visual-benchmark artifacts (superseded by FIV)
rm -f results/live_misviz_tsm.json results/live_scshortcut_tsm.json results/live_tsm_summary.csv 2>/dev/null || true

# stray downloaded PDF in repo root
rm -f 2606.16914v1.pdf 2>/dev/null || true

# LaTeX / R build cruft (keep the .tex, .R and the final .pdf)
rm -f paper/*.aux paper/*.log paper/*.out paper/*.toc paper/Rplots.pdf 2>/dev/null || true
# the old discarded paper (superseded by fiv_report)
rm -f paper/main.tex paper/main.pdf paper/main.aux paper/main.log paper/main.out paper/figures.R 2>/dev/null || true

# python caches anywhere
find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
find . -type f -name '*.pyc' -delete 2>/dev/null || true

echo "== done =="
