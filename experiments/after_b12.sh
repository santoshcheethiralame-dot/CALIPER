#!/usr/bin/env bash
# One-shot follow-on to the B-series queue: wait for it to finish, rewrite B-12's report
# under the per-arm verdict, then start B-14 (docs/preregistration-b14-primary-rerun.md).
#
# Launched from Task Scheduler through after_b12.cmd for the same reason the queue is:
# a process started from an interactive shell sits in that shell's job object and dies
# with it.
#
# "Finished" means no process is running the queue, B-12 or a gate fit. Rows on disk
# cannot tell running from done, and kill -0 cannot see native Windows processes from
# MSYS, so the check reads the Windows process table.
case "$0" in
  */*) cd "${0%/*}/.." ;;
esac
export PYTHONPATH=.
LOG=results/after_b12.log
say () { printf '[%(%Y-%m-%d %H:%M)T] %s\n' -1 "$*" >> "$LOG"; }

busy () {
  powershell -NoProfile -Command "\$p = Get-CimInstance Win32_Process | Where-Object { \$_.CommandLine -match 'run_queue|b12_batch_invariance|e01_gate' -and \$_.ProcessId -ne \$PID }; if (\$p) { exit 0 } else { exit 1 }" >/dev/null 2>&1
}

say "waiting for the B-series queue to finish"
while busy; do sleep 120; done
say "queue idle"

CTL=results/b12_ctl_b032_r5.jsonl
rows=0
if [ -f "$CTL" ]; then
  while IFS= read -r line; do [ -n "$line" ] && rows=$((rows + 1)); done < "$CTL"
fi
if [ "$rows" -ge 50 ]; then
  say "B-12 control complete ($rows rows); rewriting the report under the per-arm verdict"
  python experiments/b12_batch_invariance.py --neurons 50 --restarts 2 --batches 32 8 1 \
    >> "$LOG" 2>&1
  say "B-12 report exit $?"
else
  say "B-12 control has $rows/50 rows; report NOT rewritten, rerun the queue to finish it"
fi

say "starting B-14"
python experiments/e01_gate.py --neurons 300 --restarts 2 --steps 1600 --tokens 8000 \
  --batch 32 --layer 6 --independent-units --out results/b14_primary_gpt2_indep.jsonl \
  >> results/b14_primary_gpt2_indep.log 2>&1
say "B-14 exit $? (resumable: rerun this script or the command to continue)"
