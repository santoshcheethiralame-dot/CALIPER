#!/usr/bin/env bash
# Ninth CPU queue: F-1b (docs/preregistration-f1b-identifiable.md), after queue 8 (B-14r).
# Every fit resumes from its last finished unit. Units and per-group settings: results/f1b_units.json.
case "$0" in
  */*) cd "${0%/*}/.." ;;
esac
export PYTHONPATH=.
LOG=results/rerun_queue9.log
say () { printf '[%(%Y-%m-%d %H:%M)T] %s\n' -1 "$*" >> "$LOG"; }
run () {  # run <name> <out.jsonl> <args...>
  local name=$1 out=$2; shift 2
  say "starting $name -> $out"
  python experiments/e01_gate.py "$@" --out "$out" >> "${out%.jsonl}.log" 2>&1
  say "$name exit $?"
}

until grep -q "queue 8 done" results/rerun_queue8.log 2>/dev/null; do
  sleep 900
done
COMMON="--d-mlp 3072 --independent-units"
declare -A FLAGS=(
  [neo10_c0]="--model EleutherAI/gpt-neo-125M --layer 10 --restarts 2"
  [neo10_c1]="--model EleutherAI/gpt-neo-125M --layer 10 --restarts 2 --corpus-seed 1"
  [gpt2_b15a]="--layer 6 --restarts 5 --fit-seed 1"
  [gpt2_b15b]="--layer 6 --restarts 2 --corpus-seed 1"
  [gpt2_b15c]="--layer 6 --restarts 2 --sequence-split"
  [neo6]="--model EleutherAI/gpt-neo-125M --layer 6 --restarts 2"
)
declare -A RESEED=([gpt2_b15a]=101)
for g in neo10_c0 gpt2_b15a gpt2_b15b neo10_c1 neo6 gpt2_b15c; do
  units=$(cat results/f1b_units_$g.txt)
  for arm in targeted natural random; do
    run "F-1b $g ($arm)" results/f1b_${g}_$arm.jsonl ${FLAGS[$g]} $COMMON --units $units --augment $arm
  done
  run "F-1b $g (reseed)" results/f1b_${g}_reseed.jsonl ${FLAGS[$g]} $COMMON --units $units \
    --fit-seed ${RESEED[$g]:-100}
done
say "queue 9 done"
