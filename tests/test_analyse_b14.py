"""B-14's analysis must reproduce B-1b exactly and stay null on scrambled labels.

The script is run on B-14 once, so it is tested here on B-1b, whose numbers are already
known (DeLong restart-minus-R2 = -0.164, p = 1.8e-07). The scrambled run is the blind-
analysis check: with alignments permuted across units, nothing should come out significant.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments" / "analyse_b14.py"
B1B = ROOT / "results" / "b1b_primary_gpt2.jsonl"
sys.path.insert(0, str(ROOT / "experiments"))


def run(tmp_path, *extra):
    if not B1B.exists():
        pytest.skip("B-1b rows not present")
    out = tmp_path / "a.json"
    cmd = [sys.executable, str(SCRIPT), "--rows", str(B1B), "--no-model",
           "--n-perm", "20", "--n-boot", "50", "--out", str(out), *extra]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=600,
                          env={"PYTHONPATH": str(ROOT), **__import__("os").environ})
    assert proc.returncode == 0, proc.stderr
    return json.loads(out.read_text())


def test_reproduces_b1b_primary(tmp_path):
    rep = run(tmp_path)
    p = rep["primary"]
    assert rep["n"] == 300 and rep["n_failures"] == 52
    assert round(p["auc_r2"], 3) == 0.942 and round(p["auc_restart"], 3) == 0.778
    assert round(p["diff_restart_minus_r2"], 3) == -0.164
    assert p["p"] < 1e-6
    assert p["verdict"].startswith("HEADLINE STANDS")
    assert rep["control_cheating_signal_auc"] == 1.0


def test_scrambled_labels_are_null(tmp_path):
    rep = run(tmp_path, "--scramble", "0")
    assert rep["primary"]["p"] > 0.05
    assert rep["primary"]["verdict"].startswith("NOT SIGNIFICANT")


def test_decision_table():
    from analyse_b14 import decision
    assert decision(-0.164, 1e-7, 52).startswith("HEADLINE STANDS")
    assert decision(-0.05, 0.01, 52).startswith("RANKING STANDS")
    assert decision(-0.20, 0.2, 52).startswith("NOT SIGNIFICANT")
    assert decision(0.10, 0.01, 52).startswith("REVERSAL")
    assert "UNDERPOWERED" in decision(-0.2, 1e-4, 20)
