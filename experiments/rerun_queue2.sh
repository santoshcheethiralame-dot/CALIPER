#!/usr/bin/env bash
# Second CPU queue, started after rerun_fixed.sh (B-2c, B-8b) finishes:
#   T-SAE-0 pilot (docs/preregistration-tsae-amendment-1.md), B-15a/b/c
#   (docs/preregistration-b15-replicates.md), B-17 (docs/preregistration-b17.md), then the main
#   T-SAE run at the operating point the pilot's rule selects.
# Launched from Task Scheduler with a hidden window, like rerun_fixed.sh. Every run resumes
# from its last finished unit, so starting this again after an interruption continues it.
case "$0" in
  */*) cd "${0%/*}/.." ;;
esac
export PYTHONPATH=.
LOG=results/rerun_queue2.log
say () { printf '[%(%Y-%m-%d %H:%M)T] %s\n' -1 "$*" >> "$LOG"; }
run () {  # run <name> <out.jsonl> <args...>
  local name=$1 out=$2; shift 2
  say "starting $name -> $out"
  python experiments/e01_gate.py "$@" --out "$out" >> "${out%.jsonl}.log" 2>&1
  say "$name exit $?"
}

while [ "$(wc -l < results/b8b_gptneo125m_indep.jsonl 2>/dev/null || echo 0)" -lt 100 ] \
      || tasklist //fi "imagename eq python.exe" 2>/dev/null | grep -q python; do
  sleep 600
done
say "B-8b complete, queue 2 starting"

PILOT=3652,20588,24478,14970,7537,2721,5559,5507,15372,24515,14404,8615,1392,19106,18647,12834
run "T-SAE-0 A (8k/1600)" results/tsae0_a.jsonl --target sae --layer 6 --units $PILOT \
  --restarts 2 --independent-units
run "T-SAE-0 B (16k/3200)" results/tsae0_b.jsonl --target sae --layer 6 --units $PILOT \
  --restarts 2 --independent-units --tokens 16000 --steps 3200

B14="--layer 6 --d-mlp 3072 --neuron-pool 300 --neurons 100 --independent-units"
run "B-15a (fit seed 1, 5 restarts)" results/b15a_fitseed1.jsonl $B14 --fit-seed 1 --restarts 5
run "B-15b (corpus seed 1)" results/b15b_corpusseed1.jsonl $B14 --corpus-seed 1 --restarts 2
run "B-15c (sequence split)" results/b15c_seqsplit.jsonl $B14 --sequence-split --restarts 2

run "B-17 (GPT-Neo L6, n=20)" results/b17_gptneo125m_l6.jsonl --model EleutherAI/gpt-neo-125M \
  --layer 6 --d-mlp 3072 --neurons 20 --restarts 2 --independent-units
python experiments/diagnose_units.py --rows results/b17_gptneo125m_l6.jsonl \
  --model EleutherAI/gpt-neo-125M --layer 6 --out results/b17_diagnostics.json \
  >> results/b17_gptneo125m_l6.log 2>&1
say "B-17 diagnostics exit $?"

# The pilot rule, applied to failure counts only (Amendment 1, with Amendment 2's
# completion of the A > 12, B < 4 case). Prints "tokens steps n min_events".
CHOICE=$(python experiments/tsae_operating_point.py results/tsae0_a.jsonl results/tsae0_b.jsonl)
say "T-SAE operating point: $CHOICE"
set -- $CHOICE
run "T-SAE main" results/tsae_main.jsonl --target sae --layer 6 --neurons "$3" \
  --sae-min-events "$4" --tokens "$1" --steps "$2" --restarts 2 --independent-units
say "queue 2 done"
