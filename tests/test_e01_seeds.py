"""Seed flags must not change any earlier run: defaults reproduce B-14's direct route.

Only the direct-route fields are compared. B-14's cascade fits drew their head
initialisation from torch's global RNG (fixed 6 Oct 2026), so its cascade numbers depend on
run history and cannot be reproduced. Run at the default thread count, which B-14 used.

Slow (fits real GPT-2 units for a few minutes), so it runs only with CALIPER_SLOW=1.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
B14 = ROOT / "results" / "b14_primary_gpt2_indep.jsonl"

pytestmark = pytest.mark.skipif(os.environ.get("CALIPER_SLOW") != "1",
                                reason="slow; set CALIPER_SLOW=1")


def test_default_seeds_reproduce_b14(tmp_path):
    if not B14.exists():
        pytest.skip("B-14 rows not present")
    out = tmp_path / "seed0.jsonl"
    cmd = [sys.executable, "experiments/e01_gate.py", "--neurons", "2", "--neuron-pool", "300",
           "--restarts", "2", "--independent-units", "--no-save-directions",
           "--fit-seed", "0", "--corpus-seed", "0", "--split-seed", "0", "--out", str(out)]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=3600,
                          env={**os.environ, "PYTHONPATH": str(ROOT)})
    assert proc.returncode == 0, proc.stderr[-2000:]
    ref = {r["_key"]: r for r in map(json.loads, B14.read_text().splitlines()) if r}
    rows = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    assert len(rows) == 2
    for r in rows:
        b = ref[r["_key"]]
        for k in ("align_direct", "stability"):
            assert r[k] == b[k], (k, r[k], b[k])
