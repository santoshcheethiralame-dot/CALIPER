#!/usr/bin/env bash
# Third CPU queue: B-2d (docs/preregistration-b2d-precision.md), after queue 2 finishes.
# A separate script rather than an edit to rerun_queue2.sh, because bash reads a running
# script from disk as it goes, and queue 2 is already running.
case "$0" in
  */*) cd "${0%/*}/.." ;;
esac
export PYTHONPATH=.
LOG=results/rerun_queue3.log
say () { printf '[%(%Y-%m-%d %H:%M)T] %s\n' -1 "$*" >> "$LOG"; }

until grep -q "queue 2 done" results/rerun_queue2.log 2>/dev/null; do
  sleep 900
done
say "starting B-2d (pythia-160m L6, n=50, fp32)"
python experiments/e01_gate.py --model EleutherAI/pythia-160m --layer 6 --d-mlp 3072 \
  --neuron-pool 300 --neurons 50 --restarts 2 --independent-units --dtype fp32 \
  --out results/b2d_pythia160m_fp32.jsonl >> results/b2d_pythia160m_fp32.log 2>&1
say "B-2d exit $?"
