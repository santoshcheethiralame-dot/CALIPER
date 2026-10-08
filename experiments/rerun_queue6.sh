#!/usr/bin/env bash
# Sixth CPU queue: B-17b/c/d (docs/preregistration-b17b-scale-artifact.md), after queue 5 (X-1).
# Launched from Task Scheduler (caliper-queue6); every fit resumes from its last finished unit.
case "$0" in
  */*) cd "${0%/*}/.." ;;
esac
export PYTHONPATH=.
LOG=results/rerun_queue6.log
say () { printf '[%(%Y-%m-%d %H:%M)T] %s\n' -1 "$*" >> "$LOG"; }
run () {  # run <name> <out.jsonl> <args...>
  local name=$1 out=$2; shift 2
  say "starting $name -> $out"
  python experiments/e01_gate.py "$@" --out "$out" >> "${out%.jsonl}.log" 2>&1
  say "$name exit $?"
}

until grep -q "queue 5 done" results/rerun_queue5.log 2>/dev/null; do
  sleep 900
done
B17=$(python -c "import json;print(','.join(str(json.loads(l)['_key']) for l in open('results/b17_gptneo125m_l6.jsonl')))")
L3=$(tr -d '\n\r' < results/b17c_units.txt)
COMMON="--d-mlp 3072 --restarts 2 --independent-units"
run "B-17b (GPT-Neo L6, coordinate dropped)" results/b17b_gptneo125m_l6_drop.jsonl \
  --model EleutherAI/gpt-neo-125M --layer 6 --units $B17 $COMMON --drop-constant-coords 0.01
run "B-17c (GPT-2 L3, as usual)" results/b17c_gpt2_l3.jsonl --layer 3 --units $L3 $COMMON
run "B-17d (GPT-2 L3, coordinate dropped)" results/b17d_gpt2_l3_drop.jsonl \
  --layer 3 --units $L3 $COMMON --drop-constant-coords 0.01
say "queue 6 done"
