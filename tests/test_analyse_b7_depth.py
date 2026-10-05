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


def test_layer_legs_share_indices_not_neurons(analysis):
    """Same 50 indices at every layer - which is NOT the same 50 neurons.

    Neuron i at layer 2 and neuron i at layer 6 have different weights and nothing in
    common but a number. The 4 Oct write-up read the shared indices as a paired design;
    that reading was withdrawn on 5 Oct. This test keeps the index fact pinned so nobody
    rediscovers it as a pairing.
    """
    keysets = [set(rows) for rows in analysis.values()]
    assert len(keysets) == 3
    assert keysets[0] == keysets[1] == keysets[2]
    assert len(keysets[0]) == 50


def _passed(row: dict) -> bool:
    return row["align_selected"] > 0.95


def test_pattern_counts_match_independent_layers(analysis):
    """The cross-layer pass/fail patterns are what three independent draws produce.

    This is the check that withdrew the paired reading: "no unit fails at every depth"
    and "depth changes which units fail" both follow from independence alone. If a
    pattern ever lands far from its independence expectation, there IS shared structure
    across index-matched units and that would need explaining.
    """
    from itertools import product

    units = sorted(analysis["02"])
    rate = {l: sum(_passed(analysis[l][u]) for u in units) / len(units) for l in analysis}
    for pat in product([False, True], repeat=3):
        obs = sum(tuple(_passed(analysis[l][u]) for l in ("02", "06", "10")) == pat
                  for u in units)
        exp = len(units)
        for l, v in zip(("02", "06", "10"), pat):
            exp *= rate[l] if v else 1 - rate[l]
        assert abs(obs - exp) <= 3, f"pattern {pat}: observed {obs}, expected {exp:.1f}"


def test_analysis_script_runs_and_reports_the_independence_null():
    if not all(p.exists() for p in LAYER_FILES.values()):
        pytest.skip("B-7 layer outputs not present")
    proc = subprocess.run(
        [sys.executable, str(ANALYSIS)], capture_output=True, text=True, timeout=300
    )
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout
    assert "expected if independent" in out
    assert "restart stability at fixed layer" in out
