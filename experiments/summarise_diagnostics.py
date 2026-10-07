"""Exploratory: diagnose_units.py output per arm, crossed with the filed failure classes.

    python experiments/summarise_diagnostics.py

Descriptive only (docs/preregistration-b17.md, "Exploratory, flagged as such"). For each arm
with saved directions it joins the run's rows to its diagnostics and reports:
  - the B-17 reading-rule class of every failing unit (geometry / optimisation / other);
  - that class crossed with B-14 Addendum 2's split (converged-wrong: held-out R2 > 0.99);
  - how many failures the stimulus cannot tell apart from the truth (sigma_align >= 0.99);
  - the same on a fresh, document-disjoint token sample (diagnose_units.py --corpus-seed 1),
    with the fitted direction's attainable R2 shortfall there (r2_true - r2_fitted);
  - AUC of held-out R2 and restart agreement under the filed Euclidean label and under two
    functional labels from the fresh sample. The R2-shortfall label shares its quantity with
    the held-out R2 check, so the sigma label is the less circular of the two.
Writes results/diagnostics_summary.json.
"""
import json
import os
import sys
from collections import Counter

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from b1_signal_calibration import PASS, roc  # noqa: E402

WRONG_BASIN_R2 = 0.99
ARMS = {
    "B-8b GPT-Neo-125m L10": ("results/b8b_gptneo125m_indep.jsonl", "results/b8b_diagnostics"),
    "B-15a GPT-2 L6 (fit seed 1)": ("results/b15a_fitseed1.jsonl", "results/b15a_diagnostics"),
    "B-15b GPT-2 L6 (corpus seed 1)": ("results/b15b_corpusseed1.jsonl",
                                        "results/b15b_diagnostics"),
    "B-15c GPT-2 L6 (sequence split)": ("results/b15c_seqsplit.jsonl",
                                         "results/b15c_diagnostics"),
    "B-2c Pythia-160m L6": ("results/b2c_pythia160m_indep.jsonl", "results/b2c_diagnostics"),
}


def b17_class(d):
    if d["r2_fitted"] < d["r2_true"] - 0.01:
        return "optimisation"
    if d["sigma_align"] >= 0.99:
        return "geometry"
    return "other"


def med(xs):
    return float(np.median(xs)) if xs else None


def arm(rows_path, diag_stem):
    rows = {r["_key"]: r for r in map(json.loads, open(os.path.join(ROOT, rows_path),
                                                      encoding="utf-8")) if r}
    diag = json.load(open(os.path.join(ROOT, diag_stem + ".json"), encoding="utf-8"))
    units = {d["unit"]: d for d in diag["units"]}
    fresh = {d["unit"]: d for d in json.load(open(os.path.join(ROOT, diag_stem + "_fresh.json"),
                                                   encoding="utf-8"))["units"]}
    assert set(units) == set(rows), "rows and diagnostics cover different units"
    gap = max(abs(units[k]["euclid_align"] - rows[k]["align_selected"]) for k in rows)
    out = {"n": len(rows), "pc1_share": diag["pc1_share"],
           "max_r2_exact_shortfall": 1 - min(d["r2_exact"] for d in units.values()),
           "max_align_mismatch": round(gap, 4)}
    fails = [k for k in rows if rows[k]["align_selected"] < PASS]
    split = {k: "converged_wrong" if rows[k]["r2_k1"] > WRONG_BASIN_R2 else "under_fitted"
             for k in fails}
    out["failures"] = len(fails)
    out["b17_class"] = dict(Counter(b17_class(units[k]) for k in fails))
    out["crossed"] = {f"{s} x {c}": n for (s, c), n in sorted(Counter(
        (split[k], b17_class(units[k])) for k in fails).items())}
    for s in ("converged_wrong", "under_fitted"):
        ks = [k for k in fails if split[k] == s]
        out[s] = {"n": len(ks),
                  "euclid_align_median": med([units[k]["euclid_align"] for k in ks]),
                  "sigma_align_median": med([units[k]["sigma_align"] for k in ks]),
                  "stimulus_equivalent": sum(units[k]["sigma_align"] >= 0.99 for k in ks),
                  "fresh_sigma_align_median": med([fresh[k]["sigma_align"] for k in ks]),
                  "fresh_sigma_align_min": min([fresh[k]["sigma_align"] for k in ks], default=None),
                  "fresh_equivalent": sum(fresh[k]["sigma_align"] >= 0.99 for k in ks),
                  "fresh_r2_shortfall_median": med([fresh[k]["r2_true"] - fresh[k]["r2_fitted"]
                                                    for k in ks])}
    passes = [k for k in rows if rows[k]["align_selected"] >= PASS]
    out["passes_min_sigma_align"] = min(units[k]["sigma_align"] for k in passes)
    out["passes_fresh_sigma_align_median"] = med([fresh[k]["sigma_align"] for k in passes])
    ks = sorted(rows)
    labels = {"euclid_lt_0.95": [rows[k]["align_selected"] < PASS for k in ks],
              "fresh_sigma_lt_0.99": [fresh[k]["sigma_align"] < 0.99 for k in ks],
              "fresh_r2_shortfall_gt_0.01": [fresh[k]["r2_true"] - fresh[k]["r2_fitted"] > 0.01
                                             for k in ks]}
    out["auc_by_label"] = {}
    for lab, fail in labels.items():
        a_r2 = roc([-rows[k]["r2_k1"] for k in ks], fail)[1]
        a_st = roc([-rows[k]["stability"] for k in ks], fail)[1]
        out["auc_by_label"][lab] = {"failures": sum(fail), "held-out R2": a_r2,
                                    "restart agreement": a_st, "diff": a_st - a_r2}
    return out


def main():
    rep = {name: arm(*paths) for name, paths in ARMS.items()}
    json.dump(rep, open(os.path.join(ROOT, "results/diagnostics_summary.json"), "w"), indent=2)
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
