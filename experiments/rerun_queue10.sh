#!/usr/bin/env bash
# Tenth CPU queue: D-1 (docs/preregistration-d1-fresh-units.md), after queue 9 (F-1b).
# The filed B-14 command on 200 fresh GPT-2 layer-6 units (results/d1_units.txt).
case "$0" in
  */*) cd "${0%/*}/.." ;;
esac
export PYTHONPATH=.
LOG=results/rerun_queue10.log
say () { printf '[%(%Y-%m-%d %H:%M)T] %s\n' -1 "$*" >> "$LOG"; }

until grep -q "queue 9 done" results/rerun_queue9.log 2>/dev/null; do
  sleep 900
done
say "starting D-1 (fresh GPT-2 L6 units)"
python experiments/e01_gate.py --neurons 200 --restarts 2 --steps 1600 --tokens 8000 \
  --batch 32 --layer 6 --independent-units --units "$(cat results/d1_units.txt)" \
  --out results/d1_gpt2_l6_fresh.jsonl >> results/d1_gpt2_l6_fresh.log 2>&1
say "D-1 exit $?"
say "queue 10 done"
