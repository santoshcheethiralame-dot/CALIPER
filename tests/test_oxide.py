"""OXIDE's scoring on directions whose answers are known, and its command line on real GPT-2 weights."""
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from caliper import oxide  # noqa: E402


def test_alignments_ignore_the_null_component():
    rng = np.random.default_rng(0)
    d, k = 12, 5
    n = np.zeros(d)
    n[3] = 1.0
    W = rng.standard_normal((d, k))
    V = W.T + 10 * n                      # exact fits plus a large component along n
    euc, ident = oxide.alignments(V, W, n)
    assert np.allclose(ident, 1.0) and (euc < 0.9).all()


def test_no_null_direction_means_both_labels_agree():
    rng = np.random.default_rng(1)
    W = rng.standard_normal((8, 4))
    euc, ident = oxide.alignments(W.T + 0.1 * rng.standard_normal((4, 8)), W, None)
    assert np.allclose(euc, ident)


def test_evaluate_scores_checks_against_failures():
    units = np.arange(40)
    ident = np.r_[np.full(30, 0.99), np.full(10, 0.5)]
    good = np.r_[np.full(30, 0.99), np.full(10, 0.6)]     # separates perfectly
    noise = np.random.default_rng(2).random(40)            # carries nothing
    rep = oxide.evaluate(units, ident, ident, {"good": good, "noise": noise})
    assert rep["failures_identifiable"] == 10
    assert rep["checks"]["good"]["auc"] == 1.0
    assert rep["difference_from_good"]["noise"]["diff"] < -0.2


def test_cli_on_gpt2_true_columns(tmp_path):
    import torch
    from caliper.activations import _blocks, _mlp_in_weight, load_model
    model, _ = load_model("gpt2", dtype=torch.float32)
    units = np.array([5, 7, 11])
    W = _mlp_in_weight(_blocks(model)[6])[:, units]
    np.savez(tmp_path / "d.npz", units=units, directions=W.T)
    with open(tmp_path / "c.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["unit", "fit"])
        w.writerows([[5, 0.99], [7, 0.98], [11, 0.97]])
    out = subprocess.run([sys.executable, "-m", "caliper.oxide", "--model", "gpt2", "--layer", "6",
                          "--directions", str(tmp_path / "d.npz"), "--checks", str(tmp_path / "c.csv"),
                          "--out", str(tmp_path / "r.json")], cwd=ROOT, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-500:]
    rep = json.load(open(tmp_path / "r.json"))
    assert rep["failures_identifiable"] == 0
    assert all(abs(u["identifiable"] - 1) < 1e-6 for u in rep["per_unit"])
