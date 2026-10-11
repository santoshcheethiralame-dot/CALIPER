#!/usr/bin/env bash
# Twelfth CPU queue (11 Oct, reordered so Paper 1 runs first): D-1, then D-3, then the rest of
# F-1b. Replaces the waits in queues 9-11; every fit resumes from its last finished unit, and a
# complete F-1b arm is skipped without rebuilding its stimulus.
case "$0" in
  */*) cd "${0%/*}/.." ;;
esac
export PYTHONPATH=.
LOG=results/rerun_queue12.log
say () { printf '[%(%Y-%m-%d %H:%M)T] %s\n' -1 "$*" >> "$LOG"; }
run () {  # run <name> <out.jsonl> <args...>
  local name=$1 out=$2; shift 2
  say "starting $name -> $out"
  python experiments/e01_gate.py "$@" --out "$out" >> "${out%.jsonl}.log" 2>&1
  say "$name exit $?"
}
rows () { local n=0; [ -f "$1" ] && n=$(grep -c . "$1"); echo "${n:-0}"; }

run "D-1 (fresh GPT-2 L6 units)" results/d1_gpt2_l6_fresh.jsonl --neurons 200 --restarts 2 --steps 1600 \
  --tokens 8000 --batch 32 --layer 6 --independent-units --units "$(cat results/d1_units.txt)"
say "D-1 done"
for t in 8000 16000 4000 2000; do
  run "D-3 at $t tokens" results/d3_gpt2_l6_t$t.jsonl --neurons 150 --restarts 2 --steps 1600 --tokens $t \
    --batch 32 --layer 6 --independent-units --units "$(cat results/d3_units.txt)"
done
say "D-3 done"

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
    out=results/f1b_${g}_$arm.jsonl
    [ "$(rows $out)" -ge "$n" ] && { say "F-1b $g ($arm) complete, skipped"; continue; }
    run "F-1b $g ($arm)" $out ${FLAGS[$g]} $COMMON --units $units --augment $arm
  done
  out=results/f1b_${g}_reseed.jsonl
  if [ "$(rows $out)" -ge "$n" ]; then say "F-1b $g (reseed) complete, skipped"; else
    run "F-1b $g (reseed)" $out ${FLAGS[$g]} $COMMON --units $units --fit-seed ${RESEED[$g]:-100}
  fi
done
say "queue 9 done"
say "queue 12 done"
