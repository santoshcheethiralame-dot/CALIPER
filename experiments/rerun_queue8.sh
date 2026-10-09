#!/usr/bin/env bash
# Eighth CPU queue: B-14 re-run with fitted directions saved (internal review round 2, A1), after
# queue 7 (F-1). Identical command to the original B-14 except --out; rows are checked against
# results/b14_primary_gpt2_indep.jsonl before the directions are used.
case "$0" in
  */*) cd "${0%/*}/.." ;;
esac
export PYTHONPATH=.
LOG=results/rerun_queue8.log
say () { printf '[%(%Y-%m-%d %H:%M)T] %s\n' -1 "$*" >> "$LOG"; }

until grep -q "queue 7 done" results/rerun_queue7.log 2>/dev/null; do
  sleep 900
done
say "starting B-14r (B-14 with directions saved)"
python experiments/e01_gate.py --neurons 300 --restarts 2 --steps 1600 --tokens 8000 \
  --batch 32 --layer 6 --independent-units --out results/b14r_primary_gpt2_dirs.jsonl \
  >> results/b14r_primary_gpt2_dirs.log 2>&1
say "B-14r exit $?"
say "queue 8 done"
