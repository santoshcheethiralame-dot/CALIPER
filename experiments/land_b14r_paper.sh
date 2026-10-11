#!/usr/bin/env bash
# Paper 1 after B-14r lands: every B-14r-dependent number, in order. Run once queue 8 is done.
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=.
grep -q "queue 8 done" results/rerun_queue8.log || { echo "B-14r not finished"; exit 1; }
python experiments/land_b14r.py > /dev/null
python experiments/analyse_round3.py --items B1 B6
python experiments/make_p1_lnnull_table.py
python experiments/make_p1_numbers.py
cd paper1 && pdflatex -interaction=nonstopmode main.tex > /dev/null; pdflatex -interaction=nonstopmode main.tex > /dev/null || true
grep "^!" main.log || echo "build clean"; grep -c "tbd{bR" tables/numbers.tex || echo "no B-14r TBD left"
