"""E-2b: does a second estimator family (MID) fail at the near-constant layers as the least-squares
estimator did? (docs/preregistration-e2b-mid-artifact.md)

    PYTHONPATH=. python experiments/analyse_e2b.py          # the filed runs -> results/e2b_analysis.json
    PYTHONPATH=. python experiments/analyse_e2b.py --dev    # development on E-2's GPT-2 layer-6 arm

Per run: passes on the Euclidean and identifiable labels (0.95 bar), and each selected MID direction's
share of squared norm on the near-constant coordinates (SD < 1% of the median SD, from the run's own
saved stimulus covariance; the lowest-SD coordinate where there is none), with restart agreement of
the fits that sit on them.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from caliper.oxide import alignments, reference  # noqa: E402

RUNS = {"GPT-Neo L6": ("results/e2b_mid_gptneo_l6", "EleutherAI/gpt-neo-125M", 6),
        "GPT-2 L3": ("results/e2b_mid_gpt2_l3", "gpt2", 3)}
DEV = {"E-2 GPT-2 L6 (development)": ("results/e2_mid_gpt2_l6", "gpt2", 6)}


def summarise(stem, model, layer):
    R = {r["_key"]: r for r in map(json.loads, open(ROOT / f"{stem}.jsonl", encoding="utf-8"))}
    ks = sorted(R)
    sd = np.sqrt(np.diag(np.load(ROOT / f"{stem}_stimulus.npz")["cov"]))
    const = np.flatnonzero(sd < 1e-2 * np.median(sd))
    coords = const if len(const) else np.array([int(sd.argmin())])
    V = []
    for k in ks:
        z = np.load(ROOT / f"{stem}_dirs/n{k}.npz")
        V.append(z["restarts"][int(z["selected"])].reshape(-1))
    V = np.array(V)
    W, n = reference(model, layer, ks)
    euc, ident = alignments(V, W, n)
    U = V / np.linalg.norm(V, axis=1, keepdims=True)
    share = (U[:, coords] ** 2).sum(1)
    agree = np.array([R[k]["restart_agreement"] for k in ks])
    on = share > 0.5
    return {"units": len(ks), "near_constant_coords": int(len(const)),
            "share_measured_on": "near-constant coordinates" if len(const) else "lowest-SD coordinate",
            "passes": int((euc >= 0.95).sum()), "median_alignment": float(np.median(euc)),
            "passes_identifiable": int((ident >= 0.95).sum()),
            "median_alignment_identifiable": float(np.median(ident)),
            "median_share": float(np.median(share)), "fits_with_share_gt_0.5": int(on.sum()),
            "median_restart_agreement_of_those": float(np.median(agree[on])) if on.any() else None,
            "median_restart_agreement_all": float(np.median(agree))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", action="store_true")
    a = ap.parse_args()
    rep = {name: summarise(*spec) for name, spec in (DEV if a.dev else RUNS).items()}
    if not a.dev:
        neo, g3 = rep["GPT-Neo L6"], rep["GPT-2 L3"]
        rep["P1 MID collapses at GPT-Neo L6"] = {"holds": neo["fits_with_share_gt_0.5"] >= 10}
        rep["P1b MID collapses at GPT-2 L3"] = {"holds": g3["fits_with_share_gt_0.5"] >= 5}
        r = neo["median_restart_agreement_of_those"]
        rep["P2 restart agreement certifies it"] = {"holds": r is not None and r >= 0.99}
    print(json.dumps(rep, indent=1))
    json.dump(rep, open(ROOT / ("results/e2b_dev.json" if a.dev else "results/e2b_analysis.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
