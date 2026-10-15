#!/usr/bin/env bash
# Eighteenth CPU queue (16 Oct): F-3a (docs/preregistration-f3a-opt-standardisation.md), OPT-125m,
# after queue 17. Fits resume by unit.
case "$0" in
  */*) cd "${0%/*}/.." ;;
esac
export PYTHONPATH=. HF_HUB_DISABLE_XET=1
LOG=results/rerun_queue18.log
say () { printf '[%(%Y-%m-%d %H:%M)T] %s\n' -1 "$*" >> "$LOG"; }
run () {  # run <name> <out.jsonl> <args...>
  local name=$1 out=$2; shift 2
  say "starting $name -> $out"
  python experiments/e01_gate.py "$@" --out "$out" >> "${out%.jsonl}.log" 2>&1
  say "$name exit $?"
}

say "queue 18 waiting for queue 17"
until grep -q "queue 17 done" results/rerun_queue17.log 2>/dev/null; do sleep 300; done
U=$(python -c "import json; print(','.join(map(str, json.load(open('results/f3a_units.json'))['units'])))")
COMMON="--model facebook/opt-125m --units $U --d-mlp 3072 --restarts 2 --independent-units"
run "F-3a-1 (L5, as usual)" results/f3a_opt125m_l5.jsonl --layer 5 $COMMON
run "F-3a-2 (L5, coordinate dropped)" results/f3a_opt125m_l5_drop.jsonl --layer 5 $COMMON --drop-constant-coords 0.01
run "F-3a-3 (L3, as usual)" results/f3a_opt125m_l3.jsonl --layer 3 $COMMON
run "F-3a-4 (L6, control)" results/f3a_opt125m_l6.jsonl --layer 6 $COMMON
say "queue 18 done"
