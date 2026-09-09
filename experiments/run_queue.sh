#!/usr/bin/env bash
# Serial queue for the B-series, local CPU.
#
# One run at a time on purpose. The machine has 10 threads and a single gate run already
# saturates them, so running two concurrently halves each without finishing anything
# sooner - and it doubles peak RAM against a 16 GB ceiling that is already the binding
# constraint on the scale runs.
#
# Every run is resumable (caliper.runtime.Checkpoint), so killing this script and
# restarting it continues from the last completed unit instead of starting over.
#
#   bash experiments/run_queue.sh          # runs everything not already complete
#
# Progress:  wc -l results/b1_*.jsonl
# Stop:      kill the process; rerun the same command to resume.

set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=.

# Refuse to start if the queue is already going.
#
# Two failures produced this, both worth keeping:
#
#   1. The queue was launched while B-1 was already running. The row guard reads
#      rows-on-disk, which was still zero because the first batch of 32 had not finished,
#      so it started a SECOND identical B-1 against the same output file. Rows-on-disk
#      tells you what finished, never what is running.
#   2. The obvious fix, `pgrep -f experiments/e01_gate.py`, silently matches nothing under
#      Git Bash on Windows - it cannot see a Windows process command line - so the guard
#      passed and started a THIRD run. A guard that cannot fail loudly is worse than none.
#
# A lockfile carrying the PID works on both, and `kill -0` distinguishes a live owner from
# a stale file left by a killed run.
LOCK="results/.queue.lock"
mkdir -p results
if [ -f "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
  echo "queue already running as PID $(cat "$LOCK") - refusing to start a second."
  echo "  wc -l results/b*.jsonl   # progress"
  echo "  kill $(cat "$LOCK")      # stop it; rerunning this script resumes"
  exit 1
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

run () {   # run <label> <expected-rows> <out.jsonl> <args...>
  local label=$1 want=$2 out=$3; shift 3
  local have; have=$(wc -l < "$out" 2>/dev/null || echo 0)
  if [ "$have" -ge "$want" ]; then
    echo "[$(date +%H:%M)] $label already complete ($have/$want) - skipping"
    return 0
  fi
  echo "[$(date +%H:%M)] $label starting ($have/$want done)"
  python experiments/e01_gate.py "$@" --out "$out" >> "${out%.jsonl}.log" 2>&1
  local rc=$? have2; have2=$(wc -l < "$out" 2>/dev/null || echo 0)
  echo "[$(date +%H:%M)] $label exit $rc, $have2/$want rows"
  # A non-zero exit with rows on disk is a survivable interruption, not a reason to
  # abandon the queue - the next invocation resumes it. A run that produced nothing is
  # a real failure and stops the queue so it gets looked at.
  if [ "$have2" -eq 0 ]; then
    echo "[$(date +%H:%M)] $label produced no rows - stopping the queue"
    return 1
  fi
}

echo "=== B-series queue started $(date) ==="

# B-1 is normally already running when this starts; the guard makes the queue safe to
# launch at any time.
run "B-1  gpt2 L6"        100 results/b1_stability_gpt2.jsonl \
    --restarts 5 || exit 1

# B-1b, the PRIMARY arm - see preregistration-b1-addendum-1-power.md. 2 restarts, not 5,
# for three reasons: the incumbent AUCs (R2 0.906, disagreement 0.802) were measured at 2,
# so holding the optimiser fixed is the fair comparison; 2 restarts is what a practitioner
# running two seeds actually holds; and C13's 22% failure rate at 2 restarts yields ~66
# failures at n=300, inside the range the power work requires. The 5-restart n=100 arm
# above is retained as the sensitivity arm, not the primary.
#
# The 300 units EXTEND the C13 draw - same rng, same seed - so the first 100 are the
# existing set and no unit is reselected.
run "B-1b gpt2 n=300 r=2" 300 results/b1b_primary_gpt2.jsonl     --restarts 2 --neurons 300 || exit 1

run "B-2b pythia n=300 r=2" 300 results/b1b_primary_pythia.jsonl     --restarts 2 --neurons 300 --model EleutherAI/pythia-160m --layer 6 --d-mlp 3072 || exit 1

run "B-2s pythia n=100 r=5" 100 results/b1_stability_pythia.jsonl     --restarts 5 --model EleutherAI/pythia-160m --layer 6 --d-mlp 3072 || exit 1

# B-7, layer sweep. L6 is already covered by B-1, so only the two ends are new.
run "B-7  gpt2 L2"         50 results/b7_layer02_gpt2.jsonl \
    --restarts 5 --layer 2 --neurons 50 || exit 1

run "B-7  gpt2 L10"        50 results/b7_layer10_gpt2.jsonl \
    --restarts 5 --layer 10 --neurons 50 || exit 1

# B-8, a third family at the same scale and shape as GPT-2: 768/3072/12, different
# training corpus. Isolates the corpus from the architecture.
run "B-8  gpt-neo-125m L6" 100 results/b8_gptneo125m.jsonl \
    --restarts 5 --model EleutherAI/gpt-neo-125M --layer 6 --d-mlp 3072 || exit 1

# B-9a, the first rung of the Pythia scale ladder. 410m is d_model 1024 and ~1.6 GB in
# fp32, which fits comfortably. 1.4b is 5.6 GB and is NOT queued here - see the notebook.
# B-9a (pythia-410m) MOVED TO KAGGLE as a rung of the B-11 scale ladder - see
# preregistration-b11-pythia-ladder.md and NEXT_SESSION_LADDER.md. The ladder runs every
# rung on one GPU, so it is internally device-consistent, and it carries pythia-1.4b which
# cannot run here at all: ~5.6 GB in fp32 against 16 GB total with 2.6 GB free under load.
# Running 410m locally too would only produce a number that cannot be pooled with either
# set.

echo "=== queue finished $(date) ==="
