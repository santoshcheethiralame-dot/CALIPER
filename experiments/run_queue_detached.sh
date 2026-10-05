#!/usr/bin/env bash
# Detached entry point. Kept as a file rather than an inline
# `Start-Process bash -c "... >> log 2>&1"` because PowerShell re-tokenises that command
# line: the first attempt reached bash with the redirection mangled, wrote nothing at all,
# and left three idle processes behind.
case "$0" in
  */*) cd "${0%/*}/.." ;;
esac
bash experiments/run_queue.sh >> results/queue.log 2>&1