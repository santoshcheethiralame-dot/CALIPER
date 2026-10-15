"""Score E-1 once, when both arms are complete (docs/preregistration-e1-fair-comparison.md).

    python experiments/analyse_e1.py          # writes results/e1_analysis.json

Label: a fit fails when |cos(P v, P w)| < 0.95 with P = I - n n', n = 1/gamma (normalised) of the layer
norm feeding the MLP (the identifiable label); the Euclidean |cos(v, w)| is the secondary label.
Primary: on the GPT-2 arm, AUC(restart agreement) - AUC(test R^2) at predicting failure, with DeLong,
a stratified bootstrap (failures and passes resampled separately) and the paired permutation test.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT))
import analyse_ln_null as ln  # noqa: E402

PASS, CW, MIN_FAIL = 0.95, 0.99, 10
ARMS = {"gpt2_l6": ("results/e1_gpt2_l6", "gpt2", 6, 200),
        "gptneo_l10": ("results/e1_gptneo_l10", "EleutherAI/gpt-neo-125M", 10, 100),
        # Amendment 1: the same GPT-2 units (100 of the 200) on Python source code. Not in P3's pool.
        "gpt2_l6_code": ("results/e1_gpt2_l6_code", "gpt2", 6, 100)}
POOLED = ("gpt2_l6", "gptneo_l10")
RULE = {"r2": 0.995, "restart_agreement": 0.97}


def load(stem):
    return [json.loads(l) for l in open(ROOT / f"{stem}.jsonl", encoding="utf-8") if l.strip()]


def unit(v):
    return v / np.linalg.norm(v)


def reading(c, n_fail):
    if c is None or n_fail < MIN_FAIL:
        return "underpowered"
    lo, hi = c["boot_ci95"]
    return "holds" if hi < 0 else ("reversed" if lo > 0 else "not replicated")


def low_variance_share(v, w, n, cov):
    """The paper's err_low_ident (analyse_a4_spectrum): share of the error between the unit, sign-aligned
    fit and reference, with n projected out, in the directions carrying the bottom 1% of stimulus
    variance (covariance with n projected out and its eigenvector dropped)."""
    from analyse_a4_spectrum import bottom
    P = np.eye(len(n)) - np.outer(n, n)
    pv, pw = unit(P @ v), unit(P @ w)
    e = np.sign(pv @ pw) * pv - pw
    L = bottom(P @ cov @ P, drop=n)
    return float(((L.T @ e) ** 2).sum() / (e @ e)) if e @ e > 0 else 0.0


def score_arm(stem, model, layer):
    R = load(stem)
    n, _ = ln.null_direction(model, layer)
    cov = np.load(ROOT / f"{stem}_stimulus.npz")["cov"]
    rows = []
    for r in R:
        z = np.load(ROOT / f"{stem}_dirs/n{r['_key']}.npz")
        v, w = z["restarts"][int(z["selected"])].astype(np.float64), z["w"].astype(np.float64)
        P = np.eye(len(n)) - np.outer(n, n)
        ident = abs(unit(P @ v) @ unit(P @ w))
        rows.append({**r, "align_ident": float(ident), "null_share": float((n @ unit(v)) ** 2),
                     "low_var_share": low_variance_share(v, w, n, cov)})
    g = lambda k: np.array([x[k] for x in rows], dtype=float)
    fail_i, fail_e = g("align_ident") < PASS, g("align_euclid") < PASS
    ra, te, va = g("restart_agreement"), g("test_r2"), g("val_r2")
    out = {"units": len(rows), "failures_identifiable": int(fail_i.sum()),
           "failures_euclidean": int(fail_e.sum()),
           "primary": ln.compare(-ra, -te, fail_i),
           "euclidean": ln.compare(-ra, -te, fail_e),
           "restart_vs_validation": ln.compare(-ra, -va, fail_i),
           "validation_vs_test": ln.compare(-va, -te, fail_i),
           "median_test_minus_val": float(np.median(te - va))}
    out["reading"] = reading(out["primary"], out["failures_identifiable"])
    cw = fail_i & (te > CW)
    out["classes"] = {"under_fitted": int((fail_i & (te <= CW)).sum()), "converged_wrong": int(cw.sum())}
    keep = te > CW
    if cw.sum() >= 5 and (keep & ~fail_i).sum() >= 5:
        from analyse_review_round1 import auc
        lv = g("low_var_share")
        out["low_variance_diagnostic"] = {"converged_wrong": int(cw.sum()),
                                          "auc_low_var_share": float(auc(lv[keep], cw[keep])),
                                          "auc_test_r2": float(auc(-te[keep], cw[keep])),
                                          "auc_restart_agreement": float(auc(-ra[keep], cw[keep]))}
    else:
        out["low_variance_diagnostic"] = "not scored: fewer than 5 converged-wrong fits or passes"
    out["verdicts_changed_by_null_removal"] = int((fail_i != fail_e).sum())
    return out, rows


def rule(rows, key, thr):
    acc = np.array([x[key] >= thr for x in rows])
    wrong = np.array([x["align_ident"] < PASS for x in rows])
    return acc, wrong


def practitioner_rule(rows):
    """The paper's rule, prospective: accept fits above a fixed threshold, pooled over both arms."""
    out = {}
    rng = np.random.default_rng(0)
    for score in ("val_r2", "test_r2"):
        a_r, w_ = rule(rows, score, RULE["r2"])
        a_s, _ = rule(rows, "restart_agreement", RULE["restart_agreement"])
        fcr = lambda acc, idx: float(w_[idx][acc[idx]].mean()) if acc[idx].any() else float("nan")
        idx = np.arange(len(rows))
        d = []
        for _ in range(2000):
            b = rng.choice(idx, len(idx))
            d.append(fcr(a_r, b) - fcr(a_s, b))
        out[score] = {"r2_rule": {"accepted": int(a_r.sum()), "correct": int((a_r & ~w_).sum()),
                                  "wrong": int((a_r & w_).sum()), "fcr": fcr(a_r, idx)},
                      "restart_rule": {"accepted": int(a_s.sum()), "correct": int((a_s & ~w_).sum()),
                                       "wrong": int((a_s & w_).sum()), "fcr": fcr(a_s, idx)},
                      "fcr_difference": fcr(a_r, idx) - fcr(a_s, idx),
                      "fcr_difference_ci95": [float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))]}
        o = out[score]
        o["prediction_holds"] = bool(o["r2_rule"]["correct"] > o["restart_rule"]["correct"]
                                     and o["fcr_difference"] <= 0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", nargs=3, metavar=("STEM", "MODEL", "LAYER"),
                    help="score one arbitrary arm without the row-count check (smoke tests)")
    a = ap.parse_args()
    if a.dev:
        rep, _ = score_arm(a.dev[0], a.dev[1], int(a.dev[2]))
        print(json.dumps(rep, indent=1, default=float)[:3000])
        return
    rep, pooled = {}, []
    for name, (stem, model, layer, n_units) in ARMS.items():
        if len(load(stem)) < n_units:
            raise SystemExit(f"{name}: {len(load(stem))} of {n_units} rows; score only once all rows exist")
        rep[name], rows = score_arm(stem, model, layer)
        if name in POOLED:
            pooled += rows
        else:
            code_rows = rows
    rep["practitioner_rule"] = practitioner_rule(pooled)
    # Amendment 1, descriptive: the same units' verdicts on prose and on code.
    prose = {r["_key"]: r for r in score_arm(*ARMS["gpt2_l6"][:3])[1]}
    paired = [(prose[r["_key"]]["align_ident"] >= PASS, r["align_ident"] >= PASS) for r in code_rows if r["_key"] in prose]
    rep["code_vs_prose_same_units"] = {"units": len(paired),
                                       "pass_prose": int(sum(p for p, _ in paired)),
                                       "pass_code": int(sum(c for _, c in paired)),
                                       "pass_prose_fail_code": int(sum(p and not c for p, c in paired)),
                                       "fail_prose_pass_code": int(sum(c and not p for p, c in paired))}
    json.dump(rep, open(ROOT / "results/e1_analysis.json", "w"), indent=1, default=float)
    for name in ARMS:
        c = rep[name]["primary"]
        print(name, rep[name]["reading"], c and round(c["diff"], 3), c and [round(x, 3) for x in c["boot_ci95"]])
    print("rule", {k: v["prediction_holds"] for k, v in rep["practitioner_rule"].items()})


if __name__ == "__main__":
    main()
