#!/usr/bin/env bash
# Fourteenth CPU queue (12 Oct, reordered so Paper 1's G-0 runs before the rest of F-1b): D-3's
# remaining budgets, then G-0, then F-1b. Replaces queues 12 and 13; a complete output is skipped
# by row count without being read, and a partial one resumes from its last finished unit.
case "$0" in
  */*) cd "${0%/*}/.." ;;
esac
export PYTHONPATH=.
LOG=results/rerun_queue14.log
say () { printf '[%(%Y-%m-%d %H:%M)T] %s\n' -1 "$*" >> "$LOG"; }
rows () { local n=0; [ -f "$1" ] && n=$(grep -c . "$1"); echo "${n:-0}"; }
run () {  # run <name> <out.jsonl> <expected rows> <args...>
  local name=$1 out=$2 n=$3; shift 3
  if [ "$(rows $out)" -ge "$n" ]; then say "$name complete, skipped"; return; fi
  say "starting $name -> $out"
  python experiments/e01_gate.py "$@" --out "$out" >> "${out%.jsonl}.log" 2>&1
  say "$name exit $?"
}

for t in 8000 16000 4000 2000; do
  run "D-3 at $t tokens" results/d3_gpt2_l6_t$t.jsonl 150 --neurons 150 --restarts 2 --steps 1600 --tokens $t \
    --batch 32 --layer 6 --independent-units --units "$(cat results/d3_units.txt)"
done
say "D-3 done"
run "G-0 (GPT-2 layer 0)" results/g0_gpt2_l0.jsonl 50 --neurons 50 --restarts 2 --steps 1600 --tokens 8000 \
  --batch 32 --layer 0 --independent-units --units "$(cat results/g0_units.txt)"
say "G-0 done"

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
  n=$(echo "$units" | tr ',' '\n' | grep -c .)
  for arm in targeted natural random; do
    run "F-1b $g ($arm)" results/f1b_${g}_$arm.jsonl $n ${FLAGS[$g]} $COMMON --units $units --augment $arm
  done
  run "F-1b $g (reseed)" results/f1b_${g}_reseed.jsonl $n ${FLAGS[$g]} $COMMON --units $units --fit-seed ${RESEED[$g]:-100}
done
say "queue 9 done"
say "queue 14 done"
