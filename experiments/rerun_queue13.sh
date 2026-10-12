#!/usr/bin/env bash
# Thirteenth CPU queue: G-0 (docs/preregistration-g0-gain-law.md), after queue 12 (D-3, then F-1b).
case "$0" in
  */*) cd "${0%/*}/.." ;;
esac
export PYTHONPATH=.
LOG=results/rerun_queue13.log
say () { printf '[%(%Y-%m-%d %H:%M)T] %s\n' -1 "$*" >> "$LOG"; }

until grep -q "queue 12 done" results/rerun_queue12.log 2>/dev/null; do
  sleep 900
done
say "starting G-0 (GPT-2 layer 0)"
python experiments/e01_gate.py --neurons 50 --restarts 2 --steps 1600 --tokens 8000 --batch 32 --layer 0 \
  --independent-units --units "$(cat results/g0_units.txt)" --out results/g0_gpt2_l0.jsonl \
  >> results/g0_gpt2_l0.log 2>&1
say "G-0 exit $?"
say "queue 13 done"
