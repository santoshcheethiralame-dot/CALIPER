#!/usr/bin/env bash
# Seventh CPU queue: F-1 (docs/preregistration-f1-interventional.md), after queue 6 (B-17b/c/d).
# Launched from Task Scheduler (caliper-queue7); every fit resumes from its last finished unit.
case "$0" in
  */*) cd "${0%/*}/.." ;;
esac
export PYTHONPATH=.
LOG=results/rerun_queue7.log
say () { printf '[%(%Y-%m-%d %H:%M)T] %s\n' -1 "$*" >> "$LOG"; }
run () {  # run <name> <out.jsonl> <args...>
  local name=$1 out=$2; shift 2
  say "starting $name -> $out"
  python experiments/e01_gate.py "$@" --out "$out" >> "${out%.jsonl}.log" 2>&1
  say "$name exit $?"
}

until grep -q "queue 6 done" results/rerun_queue6.log 2>/dev/null; do
  sleep 900
done
NEO="--model EleutherAI/gpt-neo-125M --layer 10 --d-mlp 3072 --restarts 2 --independent-units \
  --units $(cat results/f1_units_b8b.txt)"
GPT2="--layer 6 --d-mlp 3072 --fit-seed 1 --restarts 5 --independent-units \
  --units $(cat results/f1_units_b15a.txt)"
for arm in targeted natural random; do
  run "F-1 GPT-2 L6 ($arm)" results/f1_b15a_$arm.jsonl $GPT2 --augment $arm
done
for arm in targeted natural random; do
  run "F-1 GPT-Neo L10 ($arm)" results/f1_b8b_$arm.jsonl $NEO --augment $arm
done
say "queue 7 done"
