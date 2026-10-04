"""Guard the B-7 paired analysis against silently reading the wrong thing.

The finding this file protects is easy to lose: B-7's three layer legs ran the SAME 50
units, so their failure sets can be compared within unit, and the answer (no unit fails at
every depth) is invisible in the aggregate pass rates. If the pairing or the pass rule
drifts, the analysis keeps running and returns a plausible-looking wrong reading, which is
how the first write-up of this result went astray.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
ANALYSIS = ROOT / "experiments" / "analyse_b7_depth.py"
LAYER_FILES = {
    "02": RESULTS / "b7_layer02_gpt2.jsonl",
    "06": RESULTS / "b7_layer06_gpt2.jsonl",
    "10": RESULTS / "b7_layer10_gpt2.jsonl",
}


def load(path: Path) -> dict[int, dict]:
    rows = {}
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                row = json.loads(line)
                rows[int(row["_key"])] = row
    return rows


@pytest.fixture(scope="module")
def analysis() -> dict[str, dict[int, dict]]:
    if not all(p.exists() for p in LAYER_FILES.values()):
        pytest.skip("B-7 layer outputs not present")
    return {layer: load(path) for layer, path in LAYER_FILES.items()}


def test_layer_legs_are_paired(analysis):
    """The whole analysis rests on this. Same units at every layer, not just same n."""
    keysets = [set(rows) for rows in analysis.values()]
    assert len(keysets) == 3
    assert keysets[0] == keysets[1] == keysets[2], (
        "B-7 legs are no longer paired - per-unit overlap is meaningless and the "
        "depth reading falls back to an unpaired comparison"
    )
    assert len(keysets[0]) == 50


def test_no_unit_fails_at_every_depth(analysis):
    """The load-bearing finding: the failure sets are near-disjoint, not nested."""
    def passed(row: dict) -> bool:
        return row["align_selected"] > 0.95 and row["k2_gain"] < 0.01

    always_fail = [
        u for u in analysis["02"]
        if not any(passed(analysis[layer][u]) for layer in analysis)
    ]
    assert always_fail == [], (
        f"{len(always_fail)} units fail at all three layers - a shared hard core "
        "contradicts the (unit, layer) reading and needs the write-up revisited"
    )


def test_depth_legs_have_no_faithful_subset_ordering(analysis):
    """Depth is not monotone per unit. If this starts passing, the docstring's
    DEPTH-BROKEN reading is stale."""
    def passed(row: dict) -> bool:
        return row["align_selected"] > 0.95 and row["k2_gain"] < 0.01

    units = sorted(analysis["02"])
    broke = [u for u in units if passed(analysis["02"][u]) and not passed(analysis["10"][u])]
    assert broke, (
        "no unit regresses from L2 to L10 any more; the README claim that depth can break "
        "a unit it handled at L2 would be stale"
    )


def test_analysis_script_runs_and_reports_paired_structure():
    if not all(p.exists() for p in LAYER_FILES.values()):
        pytest.skip("B-7 layer outputs not present")
    proc = subprocess.run(
        [sys.executable, str(ANALYSIS)], capture_output=True, text=True, timeout=300
    )
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout
    assert "shared core" in out
    assert "restart stability at fixed layer" in out
    assert "fail at all three layers (shared core) : 0/" in out
