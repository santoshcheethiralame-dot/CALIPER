"""N-1: do the checks keep their ordering when the attainable fit is unknown?
(docs/preregistration-n1-response-noise.md)

    python experiments/analyse_n1.py                      # the three filed arms
    python experiments/analyse_n1.py --arm dev=results/b15b_corpusseed1.jsonl \\
        --out results/n1_dev.json                         # development on noiseless rows

Primary: N-1a's AUC(restart agreement) - AUC(held-out R2), DeLong, stratified bootstrap CI and
paired permutation p. Secondary: the same for N-1b/c beside B-15's noiseless re-fits;
SNR-normalised R2 (not ground-truth-free: it needs the ceiling); failure rate; the
converged-wrong / under-fitted split with the class boundary at 0.99 of each unit's ceiling.
Rows without an "snr" field are treated as noiseless (ceiling 1).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from analyse_review_round1 import auc, boot_diff, paired_perm  # noqa: E402
from b1_signal_calibration import PASS, delong  # noqa: E402

ARMS = {"N-1a (SNR 1-19 per unit, primary)": "results/n1a_snr_mixed.jsonl",
        "N-1b (SNR 19)": "results/n1b_snr19.jsonl",
        "N-1c (SNR 4)": "results/n1c_snr4.jsonl",
        "B-15c (noiseless, sequence split)": "results/b15c_seqsplit.jsonl",
        "B-15b (noiseless, corpus seed 1)": "results/b15b_corpusseed1.jsonl"}


def analyse(path):
    R = [json.loads(l) for l in open(ROOT / path, encoding="utf-8") if l.strip()]
    fail = np.array([r["align_selected"] < PASS for r in R])
    ceil = np.array([r["snr"] / (1 + r["snr"]) if "snr" in r else 1.0 for r in R])
    st = np.array([-r["stability"] for r in R])
    r2 = np.array([-r["r2_k1"] for r in R])
    out = {"n": len(R), "failures": int(fail.sum()), "failure_rate": float(fail.mean()),
           "ceiling_range": [float(ceil.min()), float(ceil.max())]}
    if not 0 < fail.sum() < len(R):
        return out
    res = delong(list(st), list(r2), list(fail))
    out.update({"auc_restart": auc(st, fail), "auc_r2": auc(r2, fail),
                "diff": auc(st, fail) - auc(r2, fail), "delong_p": res[5],
                "boot_ci95": boot_diff(st, r2, fail), "paired_perm_p": paired_perm(st, r2, fail),
                "auc_r2_over_ceiling (not ground-truth-free)": auc(r2 / ceil, fail)})
    rel = -r2 / ceil
    out["class_split"] = {"converged_wrong": int((fail & (rel > 0.99)).sum()),
                          "under_fitted": int((fail & (rel <= 0.99)).sum())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", default=None, help="name=path, replaces the arms")
    ap.add_argument("--out", default="results/n1_analysis.json")
    a = ap.parse_args()
    arms = dict(x.split("=", 1) for x in a.arm) if a.arm else ARMS
    rep = {}
    for name, path in arms.items():
        if not (ROOT / path).exists():
            rep[name] = "missing"
            continue
        rep[name] = analyse(path)
        v = rep[name]
        print(name, {k: v[k] for k in ("n", "failures", "diff", "boot_ci95") if k in v})
    prim = rep.get("N-1a (SNR 1-19 per unit, primary)")
    if isinstance(prim, dict) and "boot_ci95" in prim:
        rep["primary_verdict"] = ("R2 keeps its lead" if prim["boot_ci95"][1] < 0
                                  else "R2's lead is not established when the ceiling is unknown")
        print(rep["primary_verdict"])
    json.dump(rep, open(ROOT / a.out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
