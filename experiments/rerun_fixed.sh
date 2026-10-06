#!/usr/bin/env bash
# B-2c then B-8b: the two coupled arms re-run on the fixed estimator
# (docs/preregistration-b2c-b8b-reruns.md). Launched from Task Scheduler through
# rerun_fixed.cmd with a hidden window: a visible console that gets closed kills the run
# (0xC000013A, twice on 5 Oct). Each run resumes from its last finished unit, so starting
# this script again after an interruption continues rather than restarts.
case "$0" in
  */*) cd "${0%/*}/.." ;;
esac
export PYTHONPATH=.
LOG=results/rerun_fixed.log
say () { printf '[%(%Y-%m-%d %H:%M)T] %s\n' -1 "$*" >> "$LOG"; }

say "starting B-2c (pythia-160m L6, n=300, fixed estimator)"
python experiments/e01_gate.py --model EleutherAI/pythia-160m --layer 6 --d-mlp 3072 \
  --neurons 300 --restarts 2 --independent-units \
  --out results/b2c_pythia160m_indep.jsonl >> results/b2c_pythia160m_indep.log 2>&1
say "B-2c exit $?"

say "starting B-8b (gpt-neo-125M L10, n=100, fixed estimator)"
python experiments/e01_gate.py --model EleutherAI/gpt-neo-125M --layer 10 --d-mlp 3072 \
  --neurons 100 --restarts 2 --independent-units \
  --out results/b8b_gptneo125m_indep.jsonl >> results/b8b_gptneo125m_indep.log 2>&1
say "B-8b exit $?"
