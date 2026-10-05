@echo off
rem Task Scheduler entry point for after_b12.sh. Same shape as caliper_queue.cmd: cd first,
rem forward-slash script path, so the bash side can resolve the repo root.
cd /d C:\Users\carbo\projects\caliper
"C:\Program Files\Git\bin\bash.exe" experiments/after_b12.sh
