#!/usr/bin/env bash
# Nineteenth CPU queue (16 Oct): F-2 (docs/preregistration-f2-active.md), after queue 18 (F-3a).
# One unit per run (the active arm selects tokens per unit); rows resume by unit.
# Units and per-group settings are F-1b's (results/f1b_units.json), populations B and C.
case "$0" in
  */*) cd "${0%/*}/.." ;;
esac
export PYTHONPATH=. HF_HUB_DISABLE_XET=1
LOG=results/rerun_queue19.log
say () { printf '[%(%Y-%m-%d %H:%M)T] %s\n' -1 "$*" >> "$LOG"; }

say "queue 19 waiting for queue 18"
until grep -q "queue 18 done" results/rerun_queue18.log 2>/dev/null; do sleep 300; done
COMMON="--d-mlp 3072 --independent-units --augment active --augment-n 2000 --augment-pool 20000"
declare -A FLAGS=(
  [neo10_c0]="--model EleutherAI/gpt-neo-125M --layer 10 --restarts 2"
  [neo10_c1]="--model EleutherAI/gpt-neo-125M --layer 10 --restarts 2 --corpus-seed 1"
  [gpt2_b15a]="--layer 6 --restarts 5 --fit-seed 1"
  [gpt2_b15b]="--layer 6 --restarts 2 --corpus-seed 1"
  [gpt2_b15c]="--layer 6 --restarts 2 --sequence-split"
  [neo6]="--model EleutherAI/gpt-neo-125M --layer 6 --restarts 2"
)
for g in neo10_c0 gpt2_b15a gpt2_b15b neo10_c1 neo6 gpt2_b15c; do
  units=$(python -c "import json; d=json.load(open('results/f1b_units.json'))['groups']['$g']['units']; print(' '.join(u for u, p in d.items() if p in 'BC'))")
  out=results/f2_${g}_active.jsonl
  say "starting F-2 $g ($units)"
  for u in $units; do
    python experiments/e01_gate.py ${FLAGS[$g]} $COMMON --units $u --out $out >> "${out%.jsonl}.log" 2>&1
    say "F-2 $g unit $u exit $?"
  done
done
say "queue 19 done"
