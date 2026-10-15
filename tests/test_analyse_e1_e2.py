"""E-1 and E-2 scorers on synthetic arms whose answers are known.

The scorers read rows, saved directions and the stimulus covariance, and take n = 1/gamma from the
model's weights. Here n is fixed and every file is written to a temporary folder, so the tests need
no model and check the scoring logic only.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import analyse_e1 as e1  # noqa: E402
import analyse_ln_null as ln  # noqa: E402

D = 16


def unit(x):
    return x / np.linalg.norm(x)


@pytest.fixture
def n_dir(monkeypatch):
    # an axis direction in the middle of the spectrum, so projecting it out keeps the covariance
    # diagonal and its lowest-variance axes known
    n = np.zeros(D)
    n[7] = 1.0
    monkeypatch.setattr(ln, "null_direction", lambda model, layer: (n, 1.0))
    return n


def write_arm(root, stem, units, n):
    """Units 0-9 pass (fit = w); units 10-19 fail (fit far from w). Fails have low restart agreement
    and low test R^2 on half of them, so both checks carry some signal."""
    rng = np.random.default_rng(0)
    (root / f"{stem}_dirs").mkdir(parents=True)
    np.savez(root / f"{stem}_stimulus.npz", cov=np.diag(np.linspace(1, 0.001, D)), mean=np.zeros(D))
    rows = []
    for u in range(units):
        w = rng.standard_normal(D)
        w -= n * (n @ w)
        fail = u >= units // 2
        v = unit(rng.standard_normal(D)) if fail else unit(w)
        np.savez(root / f"{stem}_dirs/n{u}.npz", w=w, restarts=np.stack([v, v]), selected=0)
        rows.append({"_key": u, "align_euclid": abs(unit(v) @ unit(w)),
                     "restart_agreement": 0.5 if fail and u % 2 else 0.99,
                     "test_r2": 0.6 if fail else 0.999, "val_r2": 0.6 if fail else 0.999})
    with open(root / f"{stem}.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)


def test_identifiable_label_and_classes(tmp_path, monkeypatch, n_dir):
    monkeypatch.setattr(e1, "ROOT", tmp_path)
    write_arm(tmp_path, "arm", 20, n_dir)
    out, rows = e1.score_arm("arm", "m", 0)
    assert out["failures_identifiable"] == 10
    assert out["classes"] == {"under_fitted": 10, "converged_wrong": 0}
    # test R^2 separates perfectly; restart agreement only half the failures, so Delta < 0
    c = out["primary"]
    assert c["auc_r2"] == pytest.approx(1.0)
    assert c["auc_restart"] < c["auc_r2"] and c["diff"] < 0


def test_null_component_is_invisible_to_the_identifiable_label(tmp_path, monkeypatch, n_dir):
    monkeypatch.setattr(e1, "ROOT", tmp_path)
    write_arm(tmp_path, "arm", 4, n_dir)
    # unit 0: add a large component along n to an otherwise perfect fit
    z = dict(np.load(tmp_path / "arm_dirs/n0.npz"))
    z["restarts"] = np.stack([unit(z["w"] + 5 * np.linalg.norm(z["w"]) * n_dir)] * 2)
    np.savez(tmp_path / "arm_dirs/n0.npz", **z)
    _, rows = e1.score_arm("arm", "m", 0)
    r0 = rows[0]
    assert r0["align_ident"] == pytest.approx(1.0, abs=1e-9)
    assert r0["null_share"] > 0.9


def test_low_variance_share_matches_the_paper_definition(n_dir):
    cov = np.diag(np.linspace(1, 0.001, D))
    rng = np.random.default_rng(1)
    w = rng.standard_normal(D)
    w -= n_dir * (n_dir @ w)
    lam, Q = np.linalg.eigh(cov)
    low = e1.low_variance_share(w + 0.05 * np.linalg.norm(w) * Q[:, 0], w, n_dir, cov)
    high = e1.low_variance_share(w + 0.05 * np.linalg.norm(w) * Q[:, -1], w, n_dir, cov)
    assert low > 0.9 and high < 0.1


def test_practitioner_rule_counts_and_sign():
    rows = ([{"val_r2": 0.999, "test_r2": 0.999, "restart_agreement": 0.99, "align_ident": 0.99}] * 40 +
            [{"val_r2": 0.999, "test_r2": 0.999, "restart_agreement": 0.5, "align_ident": 0.99}] * 40 +
            [{"val_r2": 0.9, "test_r2": 0.9, "restart_agreement": 0.99, "align_ident": 0.5}] * 5)
    out = e1.practitioner_rule(rows)["val_r2"]
    assert out["r2_rule"] == {"accepted": 80, "correct": 80, "wrong": 0, "fcr": 0.0}
    assert out["restart_rule"]["wrong"] == 5 and out["restart_rule"]["correct"] == 40
    assert out["fcr_difference"] < 0 and out["prediction_holds"]


def test_reading_rules():
    assert e1.reading(None, 30) == "underpowered"
    assert e1.reading({"boot_ci95": [-0.2, -0.05]}, 9) == "underpowered"
    assert e1.reading({"boot_ci95": [-0.2, -0.05]}, 10) == "holds"
    assert e1.reading({"boot_ci95": [-0.2, 0.05]}, 10) == "not replicated"
    assert e1.reading({"boot_ci95": [0.01, 0.2]}, 10) == "reversed"


def test_e2_cross_family_and_mcnemar(tmp_path, monkeypatch, n_dir):
    import analyse_e2 as e2
    monkeypatch.setattr(e1, "ROOT", tmp_path)
    monkeypatch.setattr(e2, "ROOT", tmp_path)
    write_arm(tmp_path, "mid", 20, n_dir)
    write_arm(tmp_path, "ln", 20, n_dir)
    monkeypatch.setattr(e2, "ARMS", {"a": ("mid", "ln", "m", 0, 20)})
    (tmp_path / "results").mkdir()
    e2.main()
    rep = json.load(open(tmp_path / "results/e2_analysis.json"))
    o = rep["a"]
    # identical synthetic arms: every unit has the same verdict under both families
    assert o["mcnemar"]["ln_only"] == 0 and o["mcnemar"]["mid_only"] == 0
    assert o["mcnemar"]["pass_ln"] == o["mcnemar"]["pass_mid"] == 10
