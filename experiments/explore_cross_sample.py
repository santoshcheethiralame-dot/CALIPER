"""Exploratory (8 Oct 2026): does agreement across token samples catch what restart agreement misses?

    python experiments/explore_cross_sample.py

B-15a (fit seed 1, corpus seed 0), B-15b (corpus seed 1) and B-15c (sequence-level split)
re-fit the same 100 GPT-2 layer-6 units and saved their selected directions. Cross-sample
agreement is |cos| between a unit's B-15a and B-15b directions; cross-split agreement uses
B-15c. Failure labels are B-15a's. Not pre-registered, and only 6 converged-wrong units:
a lead for a filed test, not a result. Writes results/cross_sample_agreement.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from analyse_review_round1 import auc  # noqa: E402

RUNS = {"a": "b15a_fitseed1", "b": "b15b_corpusseed1", "c": "b15c_seqsplit"}


def rows(stem):
    return {r["_key"]: r for r in map(json.loads, open(ROOT / f"results/{stem}.jsonl"))}


def direction(stem, r):
    v = np.load(ROOT / f"results/{stem}_dirs/n{r['_key']}.npz")[r["picked"]][:, 0]
    return v / np.linalg.norm(v)


def main():
    R = {k: rows(s) for k, s in RUNS.items()}
    ks = sorted(set(R["a"]) & set(R["b"]) & set(R["c"]))
    A = R["a"]
    fail = np.array([A[k]["align_selected"] < 0.95 for k in ks])
    cw = fail & np.array([A[k]["r2_k1"] > 0.99 for k in ks])
    uf = fail & ~cw
    scores = {"held-out R2": np.array([-A[k]["r2_k1"] for k in ks]),
              "restart agreement": np.array([-A[k]["stability"] for k in ks])}
    for other in ("b", "c"):
        scores[f"agreement with {RUNS[other]}"] = -np.array(
            [abs(direction(RUNS["a"], A[k]) @ direction(RUNS[other], R[other][k])) for k in ks])
    out = {"units": len(ks), "failures": int(fail.sum()), "converged_wrong": int(cw.sum()),
           "auc": {}}
    for name, s in scores.items():
        out["auc"][name] = {"all failures": auc(s, fail),
                            "converged-wrong vs pass": auc(s[cw | ~fail], cw[cw | ~fail]),
                            "under-fitted vs pass": auc(s[uf | ~fail], uf[uf | ~fail])}
    print(json.dumps(out, indent=1))
    json.dump(out, open(ROOT / "results/cross_sample_agreement.json", "w"), indent=1)


if __name__ == "__main__":
    main()
