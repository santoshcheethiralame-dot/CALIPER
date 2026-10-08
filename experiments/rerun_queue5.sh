#!/usr/bin/env bash
# Fifth CPU queue: X-1 (docs/preregistration-x1-cross-sample.md), after queue 4 (N-1).
# Launched from Task Scheduler (caliper-queue5); every fit resumes from its last finished unit.
case "$0" in
  */*) cd "${0%/*}/.." ;;
esac
export PYTHONPATH=.
LOG=results/rerun_queue5.log
say () { printf '[%(%Y-%m-%d %H:%M)T] %s\n' -1 "$*" >> "$LOG"; }
run () {  # run <name> <out.jsonl> <args...>
  local name=$1 out=$2; shift 2
  say "starting $name -> $out"
  python experiments/e01_gate.py "$@" --out "$out" >> "${out%.jsonl}.log" 2>&1
  say "$name exit $?"
}

until grep -q "queue 4 done" results/rerun_queue4.log 2>/dev/null; do
  sleep 900
done
UNITS=$(tr -d '\n\r' < results/x1_units.txt)
X1="--model EleutherAI/gpt-neo-125M --layer 10 --d-mlp 3072 --units $UNITS --restarts 2 --independent-units"
run "X-1a (corpus seed 0)" results/x1a_gptneo_l10.jsonl $X1
run "X-1b (corpus seed 1, seed-0 documents excluded)" results/x1b_gptneo_l10.jsonl $X1 \
  --corpus-seed 1 --exclude-corpus-seed 0
python experiments/diagnose_units.py --rows results/x1a_gptneo_l10.jsonl \
  --model EleutherAI/gpt-neo-125M --layer 10 --corpus-seed 2 --exclude-seed 0 \
  --out results/x1a_diagnostics_fresh.json >> results/x1a_diagnostics_fresh.log 2>&1
say "X-1a fresh diagnostics exit $?"
say "queue 5 done"
