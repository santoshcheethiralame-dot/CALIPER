#!/usr/bin/env bash
# Seventeenth CPU queue (15 Oct): after E-1 (queue 16) finishes, test R^2 on unseen documents for the
# earlier noiseless arms with saved directions (exploratory), then E-2 (MID, docs/preregistration-e2-mid.md)
# on E-1's GPT-2 then GPT-Neo units. Fits resume by unit; the test-R^2 job re-runs whole (minutes).
case "$0" in
  */*) cd "${0%/*}/.." ;;
esac
export PYTHONPATH=. HF_HUB_DISABLE_XET=1
LOG=results/rerun_queue17.log
say () { printf '[%(%Y-%m-%d %H:%M)T] %s\n' -1 "$*" >> "$LOG"; }
units () { python -c "import json; print(','.join(map(str, json.load(open('results/e1_units.json'))['$1'])))"; }

say "queue 17 waiting for queue 16"
until grep -q "queue 16 done" results/rerun_queue16.log 2>/dev/null; do sleep 300; done
say "starting test R2 on saved arms"
python experiments/test_r2_saved_arms.py >> results/test_r2_saved_arms.log 2>&1
say "test R2 on saved arms exit $?"
for spec in "gpt2_l6|gpt2|6" "gptneo_l10|EleutherAI/gpt-neo-125M|10"; do
  IFS='|' read -r arm model layer <<< "$spec"
  say "starting E-2 $arm"
  python experiments/e2_mid.py --model "$model" --layer "$layer" --units "$(units $arm)" \
    --out results/e2_mid_${arm}.jsonl >> results/e2_mid_${arm}.log 2>&1
  say "E-2 $arm exit $?"
done
say "queue 17 done"
