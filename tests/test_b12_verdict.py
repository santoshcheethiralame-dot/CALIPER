"""Exercise B-12's verdict logic without spending compute on the real arms.

B-12's criterion was changed on 4 Oct 2026 from "zero flips" to "no more flips than a
restart-only control on the same units", because the estimator's verdicts are only ~80%
reproducible under a restart change. The logic that decides PASS/FAIL therefore encodes a
measurement claim, and the failure mode is silent: a wrong verdict would be reported as a
result about float32 reduction order.

`--results-dir` lets the script find pre-seeded arm files and skip every subprocess, so the
comparison and verdict path runs for real without touching results/ or the queue's files.
"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments" / "b12_batch_invariance.py"
PASS = 0.95
UNITS = 8
ABOVE = 0.99
BELOW = 0.949


def all_pass() -> list[float]:
    return [ABOVE] * UNITS


def with_fails(*idx: int) -> list[float]:
    """An arm where the listed units fall below the bar and the rest sit clearly above."""
    vals = list(all_pass())
    for i in idx:
        vals[i] = BELOW
    return vals


def arm(tmp_path: Path, name: str, aligns: list[float]) -> None:
    """Write an arm file shaped like e01_gate.py output: one row per unit."""
    rows = [{"align_selected": a, "_key": 100 + i} for i, a in enumerate(aligns)]
    (tmp_path / name).write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def run(tmp_path: Path, ref_aligns, batch_aligns, control_aligns) -> dict:
    """Run B-12 with both batch arms plus an optional control, all pre-seeded."""
    arm(tmp_path, "b12_fix_b004.jsonl", ref_aligns)
    arm(tmp_path, "b12_fix_b001.jsonl", batch_aligns)
    if control_aligns is not None:
        arm(tmp_path, "b12_ctl_b004_r5.jsonl", control_aligns)
    out = tmp_path / "report.json"
    cmd = [
        sys.executable, str(SCRIPT),
        "--neurons", str(UNITS), "--batches", "4", "1",
        "--control-restarts", "5" if control_aligns is not None else "0",
        "--results-dir", str(tmp_path), "--out", str(out),
    ]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr
    return json.loads(out.read_text())


def test_zero_flips_is_a_pass(tmp_path):
    rep = run(tmp_path, all_pass(), all_pass(), all_pass())
    assert rep["n_flips_total"] == 0
    assert rep["VERDICT"].startswith("PASS")


def test_flips_within_control_floor_pass(tmp_path):
    """One batch flip and one control flip: batch size did no worse than restarts did."""
    rep = run(tmp_path, with_fails(0), all_pass(), all_pass())
    assert rep["n_flips_total"] == 1
    assert rep["noise_floor"]["n_flips"] == 1
    assert rep["VERDICT"].startswith("PASS"), rep["VERDICT"]
    assert "within the 1-flip noise floor" in rep["VERDICT"]


def test_flips_beyond_control_floor_fail(tmp_path):
    """Two batch flips against a clean control: beyond the floor, so it is a real effect.

    The control repeats the reference exactly, so its own flip count is 0 and any batch
    flip at all is attributable to the batch change rather than to restart noise.
    """
    rep = run(tmp_path, with_fails(0), with_fails(1), with_fails(0))
    assert rep["n_flips_total"] == 2
    assert rep["noise_floor"]["n_flips"] == 0
    assert rep["VERDICT"].startswith("FAIL"), rep["VERDICT"]
    assert "reduction order" in rep["VERDICT"]


def test_control_floor_is_reported_with_external_comparison(tmp_path):
    rep = run(tmp_path, all_pass(), all_pass(), with_fails(0))
    floor = rep["noise_floor"]
    assert floor["kind"].startswith("restart-only")
    assert floor["control_restarts"] == 5 and floor["reference_restarts"] == 2
    assert floor["flip_rate"] == round(1 / UNITS, 4)
    # the independent cross-run estimate must be present so the two can be compared
    assert rep["external_floor"]["flip_rate"] == 0.20


def test_disabling_control_restores_strict_zero_flip_criterion(tmp_path):
    """A flip with no control available must fail loudly rather than pass by default."""
    rep = run(tmp_path, with_fails(0), all_pass(), None)
    assert "noise_floor" not in rep
    assert rep["n_flips_total"] == 1
    assert rep["VERDICT"].startswith("FAIL")
    assert "zero-flip criterion" in rep["VERDICT"]


def test_flip_rate_is_about_the_bar_not_about_magnitude(tmp_path):
    """A large change well above the bar is not a flip; only crossing the bar counts."""
    rep = run(tmp_path, [0.96] * UNITS, [0.999] * UNITS, [0.96] * UNITS)
    assert rep["n_flips_total"] == 0
    assert rep["comparisons"]["b1_vs_b4"]["max_abs_delta"] > 0.03


def test_pass_bar_matches_the_gate(tmp_path):
    """Guard the hardcoded 0.95 against drifting away from b1_signal_calibration.PASS."""
    sys.path.insert(0, str(ROOT / "experiments"))
    from b1_signal_calibration import PASS as GATE_PASS

    assert GATE_PASS == PASS
