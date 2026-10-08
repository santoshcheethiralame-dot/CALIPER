"""B-17b/c/d: is the GPT-Neo layer-6 collapse a standardisation artifact, and does the layer
map predict it at GPT-2 layer 3? (docs/preregistration-b17b-scale-artifact.md)

    python experiments/analyse_b17b.py                    # the filed runs
    python experiments/analyse_b17b.py --dev              # development on B-17 alone

Per run: pass count at the 0.95 bar, median Euclidean alignment, and the median share of each
fitted direction's squared norm on the near-constant coordinates (SD < 1% of the median SD,
from the run's own rebuilt stimulus).
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

RUNS = {  # name: (stem, model, layer)
    "B-17 GPT-Neo L6, as run": ("results/b17_gptneo125m_l6", "EleutherAI/gpt-neo-125M", 6),
    "B-17b GPT-Neo L6, coordinate dropped": ("results/b17b_gptneo125m_l6_drop", "EleutherAI/gpt-neo-125M", 6),
    "B-17c GPT-2 L3, as usual": ("results/b17c_gpt2_l3", "gpt2", 3),
    "B-17d GPT-2 L3, coordinate dropped": ("results/b17d_gpt2_l3_drop", "gpt2", 3),
}


def summarise(stem, model, layer):
    R = {r["_key"]: r for r in map(json.loads, open(ROOT / f"{stem}.jsonl", encoding="utf-8"))}
    ks = sorted(R)
    S, _ = stimulus(model, layer, ks)
    sd = S.std(0)
    const = sd < 1e-2 * np.median(sd)
    share = []
    for k in ks:
        z = np.load(ROOT / f"{stem}_dirs/n{k}.npz")
        v = z[R[k]["picked"]][:, 0]
        v = v / np.linalg.norm(v)
        share.append(float((v[const] ** 2).sum()))
    al = np.array([R[k]["align_selected"] for k in ks])
    return {"units": len(ks), "near_constant_coords": int(const.sum()),
            "passes": int((al >= 0.95).sum()), "median_alignment": float(np.median(al)),
            "median_share_on_near_constant": float(np.median(share)),
            "fits_with_share_gt_0.5": int((np.array(share) > 0.5).sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", action="store_true")
    ap.add_argument("--out", default="results/b17b_analysis.json")
    a = ap.parse_args()
    runs = {k: v for k, v in RUNS.items() if not a.dev or k.startswith("B-17 ")}
    rep = {name: summarise(*spec) for name, spec in runs.items()}
    if not a.dev:
        b, b2 = rep["B-17b GPT-Neo L6, coordinate dropped"], rep["B-17 GPT-Neo L6, as run"]
        c, d = rep["B-17c GPT-2 L3, as usual"], rep["B-17d GPT-2 L3, coordinate dropped"]
        rep["P1 rescue"] = {"holds": b["passes"] >= 10 and b["median_alignment"] >= 0.90}
        rep["P2 map prediction"] = {"holds": c["near_constant_coords"] >= 1
                                    and c["fits_with_share_gt_0.5"] >= 5}
        rep["P3 fix at GPT-2 L3"] = {"holds": d["passes"] > c["passes"]
                                     and d["median_share_on_near_constant"] == 0.0}
    print(json.dumps(rep, indent=1))
    json.dump(rep, open(ROOT / (a.out if not a.dev else "results/b17b_dev.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
