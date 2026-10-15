#!/usr/bin/env bash
# Sixteenth CPU queue (15 Oct): E-1 (docs/preregistration-e1-fair-comparison.md), GPT-2 layer 6 then
# GPT-Neo-125m layer 10. Waits for queue 15 (P2-H-1B) to finish so the two never share the CPU.
# Fits resume by unit.
case "$0" in
  */*) cd "${0%/*}/.." ;;
esac
export PYTHONPATH=. HF_HUB_DISABLE_XET=1
LOG=results/rerun_queue16.log
say () { printf '[%(%Y-%m-%d %H:%M)T] %s\n' -1 "$*" >> "$LOG"; }
units () { python -c "import json; print(','.join(map(str, json.load(open('results/e1_units.json'))['$1'])))"; }

say "queue 16 waiting for queue 15"
# Queue 15b's failed evening already wrote one "queue 15 done"; queue 15c writes the second.
until [ "$(grep -c "queue 15 done" results/rerun_queue15.log)" -ge 2 ]; do sleep 300; done
for spec in "gpt2_l6|gpt2|6" "gptneo_l10|EleutherAI/gpt-neo-125M|10"; do
  IFS='|' read -r arm model layer <<< "$spec"
  say "starting E-1 $arm"
  python experiments/e1_fit.py --model "$model" --layer "$layer" --units "$(units $arm)" \
    --out results/e1_${arm}.jsonl >> results/e1_${arm}.log 2>&1
  say "E-1 $arm exit $?"
done
say "queue 16 done"
