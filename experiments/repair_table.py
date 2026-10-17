"""Post hoc (17 Oct): how much of the standardisation failure a free repair recovers, by estimator.

    PYTHONPATH=. python experiments/repair_table.py      # writes results/repair_table.json

Not filed; no new fit. For every saved run at a near-constant layer, four scores of the same units on
the Euclidean bar (|cos| >= 0.95 with the weight column):
  as fitted          the direction the estimator returned
  zeroed             the same direction with the near-constant coordinates set to 0, renormalised
                     (the repair a practitioner can apply after the fact, knowing the coordinates
                     from caliper.lint or the weights)
  identifiable       the as-fitted direction scored with 1/gamma removed from fit and reference
  refit, dropped     a separate fit that never saw the coordinates (B-17b, B-17d, F-3a-2), where run
Near-constant coordinates: SD below 1% of the median over the run's own stimulus.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT))
from analyse_review_round1 import stimulus  # noqa: E402
from caliper.oxide import alignments, reference  # noqa: E402

NEO, OPT = "EleutherAI/gpt-neo-125M", "facebook/opt-125m"
RUNS = [  # (name, estimator, stem, model, layer, refit-with-drop stem or None, format)
    ("GPT-Neo L6 (B-17)", "least squares", "results/b17_gptneo125m_l6", NEO, 6, "results/b17b_gptneo125m_l6_drop", "e01"),
    ("GPT-2 L3 (B-17c)", "least squares", "results/b17c_gpt2_l3", "gpt2", 3, "results/b17d_gpt2_l3_drop", "e01"),
    ("OPT-125m L5 (F-3a)", "least squares", "results/f3a_opt125m_l5", OPT, 5, "results/f3a_opt125m_l5_drop", "e01"),
    ("OPT-125m L3 (F-3a)", "least squares", "results/f3a_opt125m_l3", OPT, 3, None, "e01"),
    ("GPT-Neo L6 (E-2b)", "MID", "results/e2b_mid_gptneo_l6", NEO, 6, None, "mid"),
    ("GPT-2 L3 (E-2b)", "MID", "results/e2b_mid_gpt2_l3", "gpt2", 3, None, "mid"),
]
BAR = 0.95


def directions(stem, fmt):
    R = {r["_key"]: r for r in map(json.loads, open(ROOT / f"{stem}.jsonl", encoding="utf-8"))}
    ks = sorted(R)
    V = []
    for k in ks:
        z = np.load(ROOT / f"{stem}_dirs/n{k}.npz")
        V.append(z[R[k]["picked"]][:, 0] if fmt == "e01" else z["restarts"][int(z["selected"])].reshape(-1))
    return ks, np.array(V)


def near_constant(stem, model, layer, ks, fmt):
    if fmt == "mid":
        sd = np.sqrt(np.diag(np.load(ROOT / f"{stem}_stimulus.npz")["cov"]))
    else:
        sd = stimulus(model, layer, ks)[0].std(0)
    return np.flatnonzero(sd < 0.01 * np.median(sd))


def main():
    rep = {}
    for name, est, stem, model, layer, drop, fmt in RUNS:
        ks, V = directions(stem, fmt)
        const = near_constant(stem, model, layer, ks, fmt)
        W, n = reference(model, layer, ks)
        euc, ident = alignments(V, W, n)
        Z = V.copy()
        Z[:, const] = 0
        zer, _ = alignments(Z, W, n)
        r = {"estimator": est, "units": len(ks), "near_constant": const.tolist(),
             "as_fitted": int((euc >= BAR).sum()), "zeroed": int((zer >= BAR).sum()),
             "identifiable": int((ident >= BAR).sum()),
             "median": {"as_fitted": float(np.median(euc)), "zeroed": float(np.median(zer)),
                        "identifiable": float(np.median(ident))}}
        if drop:
            kd, Vd = directions(drop, "e01")
            ed, _ = alignments(Vd, *reference(model, layer, kd))
            r["refit_dropped"] = int((ed >= BAR).sum())
            r["median"]["refit_dropped"] = float(np.median(ed))
            r["refit_units"] = len(kd)
        rep[name] = r
        print(name, {k: v for k, v in r.items() if k != "near_constant"}, flush=True)
    json.dump(rep, open(ROOT / "results/repair_table.json", "w"), indent=1)


if __name__ == "__main__":
    main()
