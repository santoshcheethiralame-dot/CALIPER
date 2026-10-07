#!/usr/bin/env bash
# Fourth CPU queue: N-1 (docs/preregistration-n1-response-noise.md), after queue 3 (B-2d).
# Launched like the others; every run resumes from its last finished unit.
case "$0" in
  */*) cd "${0%/*}/.." ;;
esac
export PYTHONPATH=.
LOG=results/rerun_queue4.log
say () { printf '[%(%Y-%m-%d %H:%M)T] %s\n' -1 "$*" >> "$LOG"; }
run () {  # run <name> <out.jsonl> <args...>
  local name=$1 out=$2; shift 2
  say "starting $name -> $out"
  python experiments/e01_gate.py "$@" --out "$out" >> "${out%.jsonl}.log" 2>&1
  say "$name exit $?"
}

until grep -q "B-2d exit" results/rerun_queue3.log 2>/dev/null; do
  sleep 900
done
B14="--layer 6 --d-mlp 3072 --neuron-pool 300 --neurons 100 --independent-units --restarts 2"
run "N-1a (SNR log-uniform 1-19, primary)" results/n1a_snr_mixed.jsonl $B14 --snr-range 1 19
run "N-1b (SNR 19)" results/n1b_snr19.jsonl $B14 --snr 19
run "N-1c (SNR 4)" results/n1c_snr4.jsonl $B14 --snr 4
say "queue 4 done"
