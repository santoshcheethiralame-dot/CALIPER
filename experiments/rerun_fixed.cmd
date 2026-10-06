@echo off
rem Task Scheduler entry point for rerun_fixed.sh. Same shape as after_b12.cmd.
cd /d C:\Users\carbo\projects\caliper
"C:\Program Files\Git\bin\bash.exe" experiments/rerun_fixed.sh
