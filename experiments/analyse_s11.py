"""S-11 secondary: uncentred classification of the saved activations.

    python experiments/analyse_s11.py

Reads results/s11_activations.pt and results/s11_primary.jsonl. Re-derives the centred
predictions as a consistency check against the jsonl, then classifies the raw activations
with the same nearest-direction rule. Writes results/s11_analysis.json.
"""
import collections
import json
import os

import torch
from scipy.stats import beta, binomtest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def exact(k, n):
    return [0.0 if k == 0 else float(beta.ppf(.025, k, n - k + 1)),
            1.0 if k == n else float(beta.ppf(.975, k + 1, n - k))]


def score(pred):
    k, n = sum(pred[c] == c for c in pred), len(pred)
    return {"k": k, "n": n, "ci95": exact(k, n),
            "p_vs_chance": binomtest(k, n, 1 / n, alternative="greater").pvalue,
            "predicted": collections.Counter(pred.values()).most_common()}


def main():
    d = torch.load(os.path.join(ROOT, "results/s11_activations.pt"))
    acts, dirs = d["acts"], d["dirs"]
    rows = [json.loads(l) for l in open(os.path.join(ROOT, "results/s11_primary.jsonl"),
                                        encoding="utf-8") if l.strip()]
    nearest = lambda a: max(dirs, key=lambda c: float(a @ dirs[c]))
    stacked = torch.stack([acts[c] for c in acts])
    mean = stacked.mean(0)
    centred = {c: nearest(acts[c] - mean) for c in acts}
    raw = {c: nearest(acts[c]) for c in acts}
    out = {
        "centred_matches_jsonl": sum(centred[r["concept"]] == r["predicted"] for r in rows),
        "centred": score(centred),
        "uncentred": score(raw),
        "mean_norm": float(mean.norm()),
        "median_activation_norm": float(stacked.norm(dim=1).median()),
        "mean_cos_to_mean": float(torch.nn.functional.cosine_similarity(
            stacked, mean[None]).mean()),
    }
    json.dump(out, open(os.path.join(ROOT, "results/s11_analysis.json"), "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
