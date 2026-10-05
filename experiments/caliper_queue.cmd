@echo off
rem Task Scheduler entry point for the B-series queue.
rem A .cmd wrapper exists because schtasks mangles a /tr argument containing the quoted
rem "C:\Program Files\Git\bin\bash.exe" path; this path has no spaces, so it passes through.
rem The task is what makes the queue survive: a process started by Start-Process or WMI
rem Create is still inside the caller's job object and gets torn down with it, which killed
rem two launch attempts mid-run. Task Scheduler runs this from the service instead.
rem
rem cd first and pass the script path with forward slashes. The bash side resolves the repo
rem root from ${0%/*}, which strips on forward slashes only; handed the backslash path it
rem matched nothing and the queue exited 1 in silence with no log line at all.
cd /d C:\Users\carbo\projects\caliper
"C:\Program Files\Git\bin\bash.exe" experiments/run_queue_detached.sh