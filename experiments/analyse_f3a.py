"""F-3a: does the layer map predict the standardisation artifact in a third family (OPT-125m, ReLU)?
(docs/preregistration-f3a-opt-standardisation.md)

    python experiments/analyse_f3a.py            # the filed runs -> results/f3a_analysis.json
    python experiments/analyse_f3a.py --dev      # development on B-17c/d (GPT-2 L3)

Per run, as in analyse_b17b.py: passes at the 0.95 bar, median Euclidean alignment, and each fitted
direction's share of squared norm on the near-constant coordinates (SD < 1% of the median SD, from the
run's own rebuilt stimulus); where a layer has none, the share on its lowest-SD coordinate. Added:
restart agreement of fits whose direct-route direction sits on those coordinates, and the
identifiable label (1/gamma of the layer norm feeding the MLP removed; caliper/oxide.py).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT))
from analyse_review_round1 import stimulus  # noqa: E402
from caliper.oxide import alignments, reference  # noqa: E402

OPT = "facebook/opt-125m"
RUNS = {  # name: (stem, model, layer)
    "L5, as usual": ("results/f3a_opt125m_l5", OPT, 5),
    "L5, coordinate dropped": ("results/f3a_opt125m_l5_drop", OPT, 5),
    "L3, as usual": ("results/f3a_opt125m_l3", OPT, 3),
    "L6, as usual (control)": ("results/f3a_opt125m_l6", OPT, 6),
}
DEV = {
    "B-17c GPT-2 L3, as usual": ("results/b17c_gpt2_l3", "gpt2", 3),
    "B-17d GPT-2 L3, coordinate dropped": ("results/b17d_gpt2_l3_drop", "gpt2", 3),
}


def share_on(v, coords):
    v = v / np.linalg.norm(v)
    return float((v[coords] ** 2).sum())


def summarise(stem, model, layer):
    R = {r["_key"]: r for r in map(json.loads, open(ROOT / f"{stem}.jsonl", encoding="utf-8"))}
    ks = sorted(R)
    S, _ = stimulus(model, layer, ks)
    sd = S.std(0)
    const = np.flatnonzero(sd < 1e-2 * np.median(sd))
    coords = const if len(const) else np.array([int(sd.argmin())])
    W, n = reference(model, layer, ks)
    picked, direct, stab = [], [], []
    for k in ks:
        z = np.load(ROOT / f"{stem}_dirs/n{k}.npz")
        picked.append(z[R[k]["picked"]][:, 0])
        direct.append(z["direct"][:, 0])
        stab.append(R[k]["stability"])
    picked, direct, stab = np.array(picked), np.array(direct), np.array(stab)
    euc, ident = alignments(picked, W, n)
    sh = np.array([share_on(v, coords) for v in picked])
    sh_direct = np.array([share_on(v, coords) for v in direct])
    on = sh_direct > 0.5
    return {"units": len(ks), "near_constant_coords": int(len(const)),
            "share_measured_on": "near-constant coordinates" if len(const) else "lowest-SD coordinate",
            "passes": int((euc >= 0.95).sum()), "median_alignment": float(np.median(euc)),
            "passes_identifiable": int((ident >= 0.95).sum()),
            "median_alignment_identifiable": float(np.median(ident)),
            "median_share": float(np.median(sh)), "fits_with_share_gt_0.5": int((sh > 0.5).sum()),
            "direct_fits_with_share_gt_0.5": int(on.sum()),
            "median_restart_agreement_of_those": float(np.median(stab[on])) if on.any() else None,
            "median_restart_agreement_all": float(np.median(stab))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", action="store_true")
    a = ap.parse_args()
    rep = {name: summarise(*spec) for name, spec in (DEV if a.dev else RUNS).items()}
    if not a.dev:
        u5, d5 = rep["L5, as usual"], rep["L5, coordinate dropped"]
        u3, c6 = rep["L3, as usual"], rep["L6, as usual (control)"]
        rep["P1 map predicts the artifact at L5"] = {"holds": u5["fits_with_share_gt_0.5"] >= 10}
        rep["P1b map predicts the artifact at L3"] = {"holds": u3["fits_with_share_gt_0.5"] >= 5}
        r = u5["median_restart_agreement_of_those"]
        rep["P2 restart agreement certifies it"] = {"holds": r is not None and r >= 0.99}
        rep["P3 dropping the coordinate repairs it"] = {
            "holds": d5["median_alignment"] - u5["median_alignment"] >= 0.30 and d5["passes"] >= u5["passes"]}
        rep["P4 control layer clean"] = {"holds": c6["fits_with_share_gt_0.5"] <= 2}
    print(json.dumps(rep, indent=1))
    json.dump(rep, open(ROOT / ("results/f3a_dev.json" if a.dev else "results/f3a_analysis.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
