import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "experiments"))
import analyse_b15 as b15  # noqa: E402


def test_icc_is_one_for_identical_and_low_for_unrelated():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(200)
    assert abs(b15.icc_a1(x, x) - 1) < 1e-12
    assert abs(b15.icc_a1(x, rng.standard_normal(200))) < 0.2
    # absolute agreement penalises a constant offset; consistency would not
    assert b15.icc_a1(x, x + 3) < 0.2


def test_clopper_pearson_known_values():
    lo, hi = b15.clopper_pearson(2, 30)
    assert abs(lo - 0.0082) < 1e-3 and abs(hi - 0.2207) < 1e-3
    assert b15.clopper_pearson(0, 10)[0] == 0.0


def test_within_run_passes_reads_best_r2_of_first_k(tmp_path):
    d = 8
    w = np.eye(d)[0]
    dirs = tmp_path / "arm_dirs"
    dirs.mkdir()
    # unit 0: restart 3 is the right one and has the best R2, so pass@2 fails, pass@5 passes
    good, bad = w, np.eye(d)[1]
    np.savez(dirs / "n0.npz", w=w, direct_restarts=np.stack([bad, bad, bad, good, bad]),
             direct_r2_restarts=np.array([0.5, 0.6, 0.4, 0.99, 0.3]))
    # unit 1: restart 0 is right and best: passes at every k
    np.savez(dirs / "n1.npz", w=w, direct_restarts=np.stack([good, bad, bad, bad, bad]),
             direct_r2_restarts=np.array([0.99, 0.5, 0.5, 0.5, 0.5]))
    rows = tmp_path / "arm.jsonl"
    rows.write_text("")
    out = b15.within_run_passes(str(rows))
    assert out["units"] == 2
    assert out["pass@2"] == 0.5 and out["pass@3"] == 0.5 and out["pass@4"] == 1.0
    assert out["pass@5"] == 1.0
