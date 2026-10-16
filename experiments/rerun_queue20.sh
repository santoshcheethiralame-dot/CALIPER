#!/usr/bin/env bash
# Twentieth CPU queue (16 Oct): E-2b (docs/preregistration-e2b-mid-artifact.md), MID at GPT-Neo L6 and
# GPT-2 L3, after queue 18b. Fits resume by unit.
case "$0" in
  */*) cd "${0%/*}/.." ;;
esac
export PYTHONPATH=. HF_HUB_DISABLE_XET=1
LOG=results/rerun_queue20.log
say () { printf '[%(%Y-%m-%d %H:%M)T] %s\n' -1 "$*" >> "$LOG"; }
say "queue 20 waiting for queue 18b"
until grep -q "queue 18b done" results/rerun_queue18.log 2>/dev/null; do sleep 300; done
NEO=$(python -c "import json;print(','.join(str(json.loads(l)['_key']) for l in open('results/b17_gptneo125m_l6.jsonl')))")
G3=$(tr -d '\n\r' < results/b17c_units.txt)
say "starting E-2b GPT-Neo L6"
python experiments/e2_mid.py --model EleutherAI/gpt-neo-125M --layer 6 --units "$NEO" \
  --out results/e2b_mid_gptneo_l6.jsonl >> results/e2b_mid_gptneo_l6.log 2>&1
say "E-2b GPT-Neo L6 exit $?"
say "starting E-2b GPT-2 L3"
python experiments/e2_mid.py --model gpt2 --layer 3 --units "$G3" \
  --out results/e2b_mid_gpt2_l3.jsonl >> results/e2b_mid_gpt2_l3.log 2>&1
say "E-2b GPT-2 L3 exit $?"
say "queue 20 done"
