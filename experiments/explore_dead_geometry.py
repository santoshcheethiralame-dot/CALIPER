"""Exploratory, not pre-registered: is a dead steering vector one that sits on directions every
concept shares, the Paper 2 analogue of Paper 1's low-variance share?

    python experiments/explore_dead_geometry.py                     # the three S-2 sessions
    python experiments/explore_dead_geometry.py --run gemma4b_kl:results/s2_gemma4b_kl:_none \
        --out results/p2g_gemma4b_kl.json        # P2-G, docs/preregistration-p2g-dead-geometry.md

--run takes model:dir:suffix, where results/s2_<model>_analysis.json holds analyse_s2's labels.

Per S-2 model, per real-arm vector (unit direction as injected):
  shared share   |cos(v, mean of the model's real vectors)|^2
  top-k share    share of squared norm on the k coordinates with the largest mean |v|
                 across all real vectors of that model (a proxy for massive-activation dims)
  participation  (sum v^2)^2 / sum v^4 / d, how spread the vector is over coordinates
AUC = P(statistic of a live vector > statistic of a dead vector), from analyse_s2's labels.
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = {"qwen3b": ("results/s2_qwen3b", "_none"), "qwen7b": ("results/s2_qwen7b", ""),
        "gemma4b": ("results/s2_gemma4b", "_none")}
ARMS = {"concept token": "concept_steer", "template tail": "tail_steer",
        "sentence mean": "sentence_steer_aperture"}


def auc(score, live):
    from scipy.stats import rankdata
    score, live = np.asarray(score, float), np.asarray(live, bool)
    m, n = live.sum(), (~live).sum()
    r = rankdata(score)
    return float((r[live].sum() - m * (m + 1) / 2) / (m * n))


def boot_ci(score, live, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    score, live = np.asarray(score, float), np.asarray(live, bool)
    li, di = np.where(live)[0], np.where(~live)[0]
    out = [auc(score[i], live[i]) for i in
           (np.concatenate([rng.choice(li, len(li)), rng.choice(di, len(di))]) for _ in range(n_boot))]
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def main(k=8):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", default=None)
    ap.add_argument("--out", default="results/explore_dead_geometry.json")
    a = ap.parse_args()
    runs = (dict((m, (d, q)) for m, d, q in (r.split(":") for r in a.run)) if a.run else RUNS)
    out = {}
    for model, (d, q) in runs.items():
        labels = {(r["arm"], r["concept"]): r["live"] for r in
                  json.load(open(ROOT / f"results/s2_{model}_analysis.json"))["per_vector"]}
        V, live, arm_of = [], [], []
        for arm, tag in ARMS.items():
            z = np.load(ROOT / d / f"s2_{model}_{tag}{q}.vectors.npz", allow_pickle=True)
            for name, v in zip(z["names"], z["vectors"]):
                V.append(v / np.linalg.norm(v))
                live.append(labels[(arm, str(name))])
                arm_of.append(arm)
        V, live = np.array(V, float), np.array(live)
        mean = V.mean(0)
        shared = (V @ (mean / np.linalg.norm(mean))) ** 2
        top = np.argsort(-np.abs(V).mean(0))[:k]
        topk = (V[:, top] ** 2).sum(1)
        part = (V ** 2).sum(1) ** 2 / (V ** 4).sum(1) / V.shape[1]
        rep = {"vectors": len(V), "live": int(live.sum()),
               "scored": bool(5 <= live.sum() <= len(live) - 5)}
        for name, s in (("shared share", shared), (f"top-{k} share", topk),
                        ("participation", part)):
            rep[name] = {"auc": auc(s, live), "ci95": boot_ci(s, live),
                         "median live": float(np.median(s[live])),
                         "median dead": float(np.median(s[~live]))}
            rep[name]["within-arm auc"] = {
                a: auc(s[np.array(arm_of) == a], live[np.array(arm_of) == a])
                for a in ARMS if 0 < live[np.array(arm_of) == a].sum() < (np.array(arm_of) == a).sum()}
        out[model] = rep
    print(json.dumps(out, indent=1))
    json.dump(out, open(ROOT / a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
