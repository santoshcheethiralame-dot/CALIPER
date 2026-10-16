#!/usr/bin/env bash
# Queue 18b (16 Oct): finish F-3a's three runs that stopped at a unit silent on every validation row
# (e01_gate.py now reports such units as unscoreable). Rows resume by unit; after queue 19.
case "$0" in
  */*) cd "${0%/*}/.." ;;
esac
export PYTHONPATH=. HF_HUB_DISABLE_XET=1
LOG=results/rerun_queue18.log
say () { printf '[%(%Y-%m-%d %H:%M)T] %s\n' -1 "$*" >> "$LOG"; }
run () {
  local name=$1 out=$2; shift 2
  say "resuming $name -> $out"
  python experiments/e01_gate.py "$@" --out "$out" >> "${out%.jsonl}.log" 2>&1
  say "$name exit $?"
}
until grep -q "queue 19 done" results/rerun_queue19.log 2>/dev/null; do sleep 300; done
U=$(python -c "import json; print(','.join(map(str, json.load(open('results/f3a_units.json'))['units'])))")
COMMON="--model facebook/opt-125m --units $U --d-mlp 3072 --restarts 2 --independent-units"
run "F-3a-1 (L5, as usual)" results/f3a_opt125m_l5.jsonl --layer 5 $COMMON
run "F-3a-2 (L5, coordinate dropped)" results/f3a_opt125m_l5_drop.jsonl --layer 5 $COMMON --drop-constant-coords 0.01
run "F-3a-3 (L3, as usual)" results/f3a_opt125m_l3.jsonl --layer 3 $COMMON
say "queue 18b done"
