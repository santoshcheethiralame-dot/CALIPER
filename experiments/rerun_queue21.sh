#!/usr/bin/env bash
# Twenty-first CPU queue (17 Oct): I-1, E-3c, then R-2 (scale floor) for Paper 1's revision.
case "$0" in
  */*) cd "${0%/*}/.." ;;
esac
export PYTHONPATH=. HF_HUB_DISABLE_XET=1
LOG=results/rerun_queue21.log
say () { printf '[%(%Y-%m-%d %H:%M)T] %s\n' -1 "$*" >> "$LOG"; }
say "starting I-1"; python experiments/i1_intervention.py > results/i1_intervention.log 2>&1; say "I-1 exit $?"
say "starting E-3c"; python experiments/e3c_probe_controlled.py > results/e3c_probe.log 2>&1; say "E-3c exit $?"
NEO=$(python -c "import json;print(','.join(str(json.loads(l)['_key']) for l in open('results/b17_gptneo125m_l6.jsonl')))")
G3=$(tr -d '\n\r' < results/b17c_units.txt)
export CALIPER_SCALE_FLOOR=0.01
COMMON="--d-mlp 3072 --restarts 2 --independent-units"
say "starting R-2a"; python experiments/e01_gate.py --model EleutherAI/gpt-neo-125M --layer 6 --units $NEO $COMMON --out results/r2_ls_gptneo_l6_floor.jsonl >> results/r2_ls_gptneo_l6_floor.log 2>&1; say "R-2a exit $?"
say "starting R-2b"; python experiments/e01_gate.py --layer 3 --units $G3 $COMMON --out results/r2_ls_gpt2_l3_floor.jsonl >> results/r2_ls_gpt2_l3_floor.log 2>&1; say "R-2b exit $?"
say "starting R-2c"; python experiments/e2_mid.py --model EleutherAI/gpt-neo-125M --layer 6 --units "$NEO" --out results/r2_mid_gptneo_l6_floor.jsonl >> results/r2_mid_gptneo_l6_floor.log 2>&1; say "R-2c exit $?"
say "starting R-2d"; python experiments/e2_mid.py --model gpt2 --layer 3 --units "$G3" --out results/r2_mid_gpt2_l3_floor.jsonl >> results/r2_mid_gpt2_l3_floor.log 2>&1; say "R-2d exit $?"
say "queue 21 done"
