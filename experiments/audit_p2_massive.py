"""Post hoc audit (17 Oct): do a few massive residual coordinates drive Paper 2's distinctness check?

    PYTHONPATH=. python experiments/audit_p2_massive.py      # writes results/audit_p2_massive.json

Not filed. For each model of Paper 2's health table, the saved steering vectors of the three arms, the
filed live/dead labels (each run's analysis file), and:
  - each vector's share of squared norm on its largest coordinate, and how many vectors share it;
  - the 5 heavy coordinates: largest mean squared share over all of the model's vectors (chosen
    without the labels);
  - distinctness (minus the max |cos| to another vector of the same arm), as filed and with the heavy
    coordinates zeroed, and its AUC at telling live from dead vectors, with stratified bootstrap CIs.
Paper 2's checks use raw projections and unit-norm cosines, so per-coordinate standardisation (Paper
1's failure mode) does not arise; this asks the related question for massive activations.
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from analyse_s2 import ARMS, auc, boot_ci  # noqa: E402

MODELS = {  # name: (vector dir, file model tag, analysis file with the filed labels, file suffix)
    "Qwen2.5-3B": ("results/s2_qwen3b", "qwen3b", "results/s2_qwen3b_analysis.json", "_none"),
    "Qwen2.5-7B": ("results/s2_qwen7b", "qwen7b", "results/s2_qwen7b_analysis.json", ""),
    "Gemma-3-4B, KL grid": ("results/s2_gemma4b_kl", "gemma4b_kl", "results/s2_gemma4b_kl_analysis.json", "_none"),
    "Qwen3.5-2B": ("results/p2n", "qwen35_2b", "results/p2n/qwen35_2b_s2.json", "_none"),
    "Qwen3.5-0.8B": ("results/p2n", "qwen35_08b", "results/p2n/qwen35_08b_s2.json", "_none"),
}
K = 5


def distinct(U, arms):
    out = np.empty(len(U))
    for arm in set(arms):
        idx = np.flatnonzero(np.array(arms) == arm)
        G = np.abs(U[idx] @ U[idx].T)
        np.fill_diagonal(G, -np.inf)
        out[idx] = -G.max(1)
    return out


def main():
    rep = {}
    for name, (d, tag, lab, q) in MODELS.items():
        live_of = {(r["arm"], r["concept"]): r["live"] for r in json.load(open(ROOT / lab))["per_vector"]}
        V, arms, live = [], [], []
        for arm, (steer, _) in ARMS.items():
            z = np.load(ROOT / d / f"s2_{tag}_{steer.format(q=q)}.vectors.npz", allow_pickle=True)
            for c, v in zip(z["names"], z["vectors"]):
                V.append(np.asarray(v, float)); arms.append(arm); live.append(live_of[(arm, str(c))])
        V, live = np.array(V), np.array(live)
        U = V / np.linalg.norm(V, axis=1, keepdims=True)
        sq = U ** 2
        heavy = np.argsort(-sq.mean(0))[:K]
        Ur = U.copy()
        Ur[:, heavy] = 0
        Ur /= np.linalg.norm(Ur, axis=1, keepdims=True)
        top, n_top = Counter(sq.argmax(1)).most_common(1)[0]
        r = {"vectors": len(V), "live": int(live.sum()), "dimension": V.shape[1],
             "median_top1_share": float(np.median(sq.max(1))),
             "median_share_on_heavy": float(np.median(sq[:, heavy].sum(1))),
             "most_common_top_coordinate": [int(top), int(n_top)], "heavy_coordinates": heavy.tolist(),
             "by_arm": {}}
        for arm in ARMS:
            m = np.array(arms) == arm
            r["by_arm"][arm] = {"median_share_on_heavy": float(np.median(sq[m][:, heavy].sum(1))),
                                "median_max_cos": float(np.median(-distinct(U[m], [arm] * m.sum()))),
                                "median_max_cos_without_heavy": float(np.median(-distinct(Ur[m], [arm] * m.sum())))}
        if live.sum() >= 5 and (~live).sum() >= 5:
            for key, X in (("distinctness", U), ("distinctness_without_heavy", Ur)):
                s = distinct(X, arms)
                r[key] = {"auc": auc(s, live), "ci95": boot_ci(s, live)}
        rep[name] = r
        print(name, {k: v for k, v in r.items() if k not in ("by_arm", "heavy_coordinates")}, flush=True)
        for arm, b in r["by_arm"].items():
            print("   ", arm, {k: round(v, 3) for k, v in b.items()}, flush=True)
    json.dump(rep, open(ROOT / "results/audit_p2_massive.json", "w"), indent=1)


if __name__ == "__main__":
    main()
