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
# Stop:      kill the process; rerunning this script resumes.
#
# Written to use shell builtins only. A `$(...)` per line looked harmless and cost this
# queue the previous session: MSYS2 has to reserve a large heap block to fork, it does not
# always get one on a 16 GB box, and when it does not bash spins on
# "dofork: errno 11 / fork: Resource temporarily unavailable". The 2026-09-10 log shows
# that killing a run mid-flight this way, so the queue looks idle while doing nothing.
# Nothing here forks except the single `python` per run. Timestamps come from printf's
# %()T and line counts from a read loop, both builtins.

set -u
# Resolve the repo root from this script's own path. Guarded because `${0%/*}` strips on
# forward slashes only: invoked as `bash experiments/run_queue.sh` it yields `experiments`,
# but handed a Windows path (`C:\...\run_queue.sh`) it matches nothing, returns the whole
# string, and the cd lands somewhere without a results/ directory. That is exactly how the
# first detached launch exited 1 in silence.
case "$0" in
  */*) cd "${0%/*}/.." ;;
esac
export PYTHONPATH=.
mkdir -p results

printf '=== B-series queue started %(%Y-%m-%d %H:%M)T ===\n' -1

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
# Liveness is checked with `tasklist`, NOT `kill -0`. Measured on this machine: `kill -0`
# reports "not alive" for a running Windows python AND for an absurd PID, because MSYS
# maintains its own process table and cannot see native Windows processes. Every earlier
# version of this guard used `kill -0`, which means it never once detected a live queue -
# it only ever worked by accident, when a stale lock happened to point at an MSYS PID.
# `tasklist` reads the real Windows process table and correctly separates live from dead.
alive () {
  tasklist //FI "PID eq $1" 2>/dev/null | grep -q "[0-9] $1 "
}

# A lockfile carrying the PID works on both, and a real liveness probe distinguishes a live
# owner from a stale file left by a killed run.
#
# The lock guards the QUEUE, which is not the same as guarding the RUN. When Task Scheduler
# aborted the chain (0x8007042B) the bash chain died but its python child survived as an
# orphan, still writing rows to results/b8_gptneo125m.jsonl. A queue started at that moment
# would have found 31 rows against a target of 100, decided B-8 was incomplete, and launched
# a SECOND python against the same output file - which is failure 1 above, recreated by the
# very guard written to prevent it. So each run also claims its own output file, and the
# claim is checked before the row count is trusted.
LOCK="results/.queue.lock"
if [ -f "$LOCK" ]; then
  read -r owner < "$LOCK" || owner=""
  if [ -n "${owner:-}" ] && alive "$owner"; then
    printf 'queue already running as PID %s - refusing to start a second.\n' "$owner"
    printf '  kill %s      # stop it; rerunning this script resumes\n' "$owner"
    exit 1
  fi
fi
printf '%s\n' "$$" > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

# claiming <out.jsonl> - refuse to start a run whose output file another process owns.
#
# Records the PID of the python we launch, so an orphaned child of a dead queue is still
# detected. The claim is written before the row count is consulted, because the row count
# is the thing that cannot distinguish "not started yet" from "already running".
CLAIM_SUFFIX=".claim"
claiming () {
  local out=$1 claim="${out%.jsonl}$CLAIM_SUFFIX" owner=""
  if [ -f "$claim" ]; then
    read -r owner < "$claim" || owner=""
    if [ -n "${owner:-}" ] && alive "$owner"; then
      stamp "REFUSING: PID $owner already holds $out (run it or kill it, not both)"
      return 1
    fi
  fi
  # Also guard against orphans that predate claims: any process alive writing this exact output.
  # Scan every process; this is slow enough to be rare but correct on Windows.
  # wmic is deprecated; use powershell to get CommandLine quickly in a loop is costly; do a single pass.
  if powershell -NoProfile -Command "\$ps = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object { \$_.CommandLine -ne \$null -and \$_.CommandLine -match 'e01_gate' -and \$_.CommandLine -match [regex]::Escape(\"$out\") }; if (\$ps) { exit 1 } else { exit 0 }" >/dev/null 2>&1; then
    : # no match
  else
    # find which one
    m=$(powershell -NoProfile -Command "\$ps = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object { \$_.CommandLine -ne \$null -and \$_.CommandLine -match 'e01_gate' -and \$_.CommandLine -match [regex]::Escape(\"$out\") }; \$ps | ForEach-Object { \$_.ProcessId }" 2>/dev/null | head -1)
    m=$(echo "$m" | tr -d '\r\n')
    if [ -n "$m" ] && alive "$m"; then
      stamp "REFUSING: PID $m already writing $out (no claim)"
      return 1
    fi
  fi
  return 0
}

# Row count into ROWS, tolerating a file that does not exist. `wc -l < missing` fails the
# redirection rather than the command, so the old `|| echo 0` fallback left the shell's
# own error in the log before it recovered. Testing for the file first is quieter and
# does not depend on which failure the shell chooses to report.
ROWS=0
count_rows () {
  ROWS=0
  [ -f "$1" ] || return 0
  local line
  while IFS= read -r line || [ -n "$line" ]; do ROWS=$((ROWS + 1)); done < "$1"
}

# stamp <message> - print a timestamped progress line.
stamp () { printf '[%(%H:%M)T] %s\n' -1 "$1"; }

# attempt <out.jsonl> <script> <args...> - one resumable run, retried once if it wrote
# nothing.
#
# A run at 5 restarts writes nothing until its first batch of 32 completes, which is over
# an hour, so a power loss at minute 74 looks identical to a script that died on import.
# The queue stopped on a dead battery and sat idle until someone looked. So: retry once.
# A genuine failure fails again immediately; an interruption gets to continue.
#
# Each launch records its own PID in the claim file, so if this queue dies mid-run the
# orphaned python is still identifiable and the next queue will not double-write its output.
attempt () {
  local out=$1 script=$2; shift 2
  # Built from $out, not $1: the shift above means $1 is now the first experiment argument
  # (--restarts), so deriving the claim path from it produced "--restarts.claim" and an
  # `rm: unknown option` on every retry.
  local log="${out%.jsonl}.log" claim="${out%.jsonl}$CLAIM_SUFFIX"
  claiming "$out" || return 1
  python "$script" "$@" --out "$out" >> "$log" 2>&1 &
  local py=$!
  printf '%s\n' "$py" > "$claim"
  wait "$py"
  local rc=$?
  rm -f "$claim"
  [ "$rc" -ne 0 ] || return 0
  count_rows "$out"
  if [ "$ROWS" -eq 0 ]; then
    stamp "  produced no rows (exit $rc) - retrying once"
    python "$script" "$@" --out "$out" >> "$log" 2>&1 &
    py=$!
    printf '%s\n' "$py" > "$claim"
    wait "$py"
    rc=$?
    rm -f "$claim"
    count_rows "$out"
    if [ "$ROWS" -eq 0 ]; then
      stamp "  no rows twice (exit $rc) - giving up on this run"
      return 1
    fi
  fi
  count_rows "$out"
  stamp "  exit $rc, $ROWS rows (resumable; rerun to continue)"
  return 0
}

# run <label> <expected-rows> <out.jsonl> <args...> - e01_gate.py, gated on row count.
run () {
  local label=$1 want=$2 out=$3; shift 3
  claiming "$out" || return 1
  count_rows "$out"
  if [ "$ROWS" -ge "$want" ]; then
    stamp "$label already complete ($ROWS/$want) - skipping"
    return 0
  fi
  stamp "$label starting ($ROWS/$want done)"
  attempt "$out" experiments/e01_gate.py "$@"
}

# marker <label> <completion-file> <out.jsonl> <script> <args...> - for the runs whose
# output is a summary rather than a row count.
#
# A killed run never reaches its analysis block, so the summary file is written only
# after every cell is finished. That makes it a truer completion test than a row count,
# which a partially-alive unit set can never reach.
marker () {
  local label=$1 done=$2 out=$3 script=$4; shift 4
  claiming "$out" || return 1
  if [ -f "$done" ]; then
    stamp "$label already complete ($done) - skipping"
    return 0
  fi
  count_rows "$out"
  stamp "$label starting ($ROWS rows on disk)"
  attempt "$out" "$script" "$@"
}

# B-1, B-1b and B-2b are complete; kept so the guard below still reports them and so the
# ordering of the programme of record stays readable.
run "B-1  gpt2 L6"        100 results/b1_stability_gpt2.jsonl --restarts 5 || exit 1
run "B-1b gpt2 n=300 r=2" 300 results/b1b_primary_gpt2.jsonl --restarts 2 --neurons 300 || exit 1
run "B-2b pythia n=300"    300 results/b1b_primary_pythia.jsonl --restarts 2 --neurons 300 \
    --model EleutherAI/pythia-160m --layer 6 --d-mlp 3072 || exit 1

# ---------------------------------------------------------------- 1. B-8, third family
#
# GPT-Neo-125m: same shape as GPT-2 (768/3072/12) on a different corpus, which isolates
# the corpus from the architecture. The leg is "does the failure rate and the R2-beats-
# restart ordering hold on a third corpus". 2 restarts, not 5, so the AUC ordering is
# directly comparable to B-1b's - a variance component measured at a different optimiser
# setting is not comparable to the components already tabulated.
#
# LAYER 10, not layer 6 as first drafted. Measured, not assumed: at matched settings
# (restarts 1, steps 200, pool 300) the planted direction does not come back at all on
# Neo layer 6 - median alignment 0.0026, 0/4 units passing - while the same four-unit
# protocol on GPT-2 layer 6 gives 0.9872 and 3/4. That is not a hook bug. The hooks are
# right: the captured MLP input is bit-identical to ln_2's output on all three models, and
# corr(r, s.w) on Neo L6 is 0.698 against GPT-2's 0.666. The data and the weight column
# are both correct, so layer 6 is simply a layer where this estimator cannot identify the
# direction on this model. Response std climbs with depth on Neo (0.076 / 0.180 / 0.584 at
# L2/L6/L10) far faster than on GPT-2 (0.150 / 0.168 / 0.281), and the recovery tracks it:
#
#     layer        Neo L2     Neo L6     Neo L10      GPT-2 L2    L6       L10
#     align     .12/.04/.15  .02/.00/.00  .52/.89/.87   .80/.00/.71  .99/.99/.98  .74/.94/.43
#
# A run at layer 6 would have returned ~0% pass rate and looked like a spectacular finding
# about a third family when it is only evidence that layer 6 was the wrong depth.
#
# Layer 10 is also the depth-matched comparison: both models have 12 blocks, so L10 is
# 10/12 in each, and B-7 is already measuring GPT-2 at L10. Read B-8 against GPT-2 L10.
#
# Verified at production settings on 6 units: median alignment 0.9862, 5/6 passing,
# 93.3 s/unit, so n=100 is ~2.6 h.
run "B-8  gpt-neo-125m L10" 100 results/b8_gptneo125m.jsonl \
    --restarts 2 --neurons 100 --model EleutherAI/gpt-neo-125M --layer 10 --d-mlp 3072 || exit 1

# ---------------------------------------------------------------- 2. B-7, layer sweep
#
# GPT-2 layers 2, 6, 10, n=50 each, restarts 5.
#
# --neuron-pool 300 is REQUIRED here and was missing until now. numpy's
# choice(replace=False) is not nested in `size`, so `choice(3072, 50)` shares exactly ONE
# unit with `choice(3072, 100)`. Without the flag these three legs would have compared
# three disjoint sets of 50 units and the depth profile would have been unanswerable.
# With it, all three legs are prefixes of one ordering, so the sweep is paired on units.
#
# L6 is a re-fit rather than a reuse of B-1: B-1's last 8 units were fitted in a batch of
# 8 because that run resumed at 92/100, so its init differs from a fresh run's. B-7 is
# internally consistent, and B-1/B-1b remain the primary arms.
run "B-7  gpt2 L2"  50 results/b7_layer02_gpt2.jsonl \
    --restarts 5 --layer 2 --neurons 50 --neuron-pool 300 || exit 1
run "B-7  gpt2 L6"  50 results/b7_layer06_gpt2.jsonl \
    --restarts 5 --layer 6 --neurons 50 --neuron-pool 300 || exit 1
run "B-7  gpt2 L10" 50 results/b7_layer10_gpt2.jsonl \
    --restarts 5 --layer 10 --neurons 50 --neuron-pool 300 || exit 1

# ---------------------------------------------------------------- 3. B-10, required-N
#
# Token sweep 2k/4k/8k/16k at n=50, restarts 2. The 8,000-token leg lands on the same 50
# units at the same operating point as B-1b, so the sweep is anchored to an analysed run
# and the 8k leg doubles as a reproducibility check on the primary arm.
#
# Underpowered by construction and says so on its face: at the 17% failure rate, 50 units
# give ~8 failures, so the AUC standard error is near 0.10 and differences below roughly
# 0.15 are not resolvable at any N on this grid. Every AUC is reported with a bootstrap
# CI and the script refuses to call a ranking change inside overlapping CIs.
marker "B-10 required-N" results/b10_required_n_summary.json results/b10_required_n.jsonl \
    experiments/b10_required_n.py --neurons 50 --neuron-pool 300 --restarts 2 || exit 1

# ---------------------------------------------------------------- 4. B-12, batch invariance
#
# Closes the defect B-0's n=50 extension found. Same 50 units at batch sizes 32, 8, 1
# with per-neuron seeding ON, so a unit's initialisation no longer depends on how many
# units share its batch. Criterion, filed before running: zero units change pass/fail
# side. Both outcomes are reportable; flips surviving a bitwise-identical initialisation
# would mean float32 reduction order alone can flip a silent-failure classification, which
# is a larger claim than batching changing the answer.
#
# Measured on this machine: batch size barely affects cost (17.0 / 18.3 / 17.4 s/unit at
# 32 / 8 / 1), so all three arms together are ~5.4 h and not the 14 h a 5-10x speedup
# assumption would predict. The largest batch runs first so a usable partial exists early.
marker "B-12 batch invariance" results/b12_batch_invariance.json results/b12_batch_invariance.json \
    experiments/b12_batch_invariance.py --neurons 50 --restarts 2 --batches 32 8 1 || exit 1

# ---------------------------------------------------------------- 5. S1-2, K>1 deflation
#
# The Study 2 gate, and the only item here with a deadline (end of November). Pre-
# registered criterion: median subspace alignment at K=2, N=8000 above 0.80, where joint
# estimation gives 0.5213. Four arms per cell because "deflation works" and "we ran a
# robust rank-1 fit repeatedly" are different claims.
#
# A one-unit probe at the real settings already reads joint 0.512 / plain 0.746 /
# cascade 0.740, which reproduces the archived 0.5213 and points at the middle branch
# (0.60-0.80: one planted direction per trait). The run is what makes that a measurement
# over 24 units and four cells rather than an anecdote.
marker "S1-2 deflation" results/s1_2_deflation.json results/s1_2_deflation.jsonl \
    experiments/s1_2_deflation.py || exit 1

# Deferred, not run: B-2s (pythia n=100 at 5 restarts) exited 1 twice with no rows on
# 2026-09-10 and is not on the current programme. B-2b already carries pythia at n=300.

printf '=== queue finished %(%Y-%m-%d %H:%M)T ===\n' -1