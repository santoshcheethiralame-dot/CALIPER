#!/usr/bin/env bash
# Eleventh CPU queue: D-3 (docs/preregistration-d3-token-scaling.md), after queue 10 (D-1).
# The same 150 fresh GPT-2 layer-6 units at four token budgets; the primary's budgets first.
case "$0" in
  */*) cd "${0%/*}/.." ;;
esac
export PYTHONPATH=.
LOG=results/rerun_queue11.log
say () { printf '[%(%Y-%m-%d %H:%M)T] %s\n' -1 "$*" >> "$LOG"; }

until grep -q "queue 10 done" results/rerun_queue10.log 2>/dev/null; do
  sleep 900
done
for t in 8000 16000 4000 2000; do
  say "starting D-3 at $t tokens"
  python experiments/e01_gate.py --neurons 150 --restarts 2 --steps 1600 --tokens $t --batch 32 \
    --layer 6 --independent-units --units "$(cat results/d3_units.txt)" \
    --out results/d3_gpt2_l6_t$t.jsonl >> results/d3_gpt2_l6_t$t.log 2>&1
  say "D-3 $t exit $?"
done
say "queue 11 done"
