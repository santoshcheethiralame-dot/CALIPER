"""P2-R filed secondaries left out of analyse_p2r.py, computed after the primary output was seen.

    python experiments/analyse_p2r_secondaries.py --dir <unzipped session>   # writes results/p2r_secondaries.json

1. Identify-live at strength 4 (R4) against S-1M's steering-gate labels: Cohen's kappa and the 2x2 table.
2. P(YES) on the probability scale per framing and strength (R1), for continuity with P2-F, and the
   per-concept change on factual_yes at strength 4 (Wilcoxon), the probability-scale twin of P1.
3. P1 and P2 for the content-free cells (R2 norm-matched, R3 impact-matched) at every strength, with
   analyse_p2r's own functions: does a random vector also flatten, and also give an introspective excess?
4. Median log-odds change per cell, framing and strength (for the paper's table and figure).
All are filed in docs/preregistration-p2r-released-protocol.md (Secondary); none was in the
scorer when the data arrived.
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import analyse_p2r as p2r  # noqa: E402


def rows(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def kappa(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    a = ap.parse_args()
    rep = {}
    st = rows(ROOT / "results/s1_gemma27/s1_gemma27_s1m_steer_macar-release.jsonl")
    base = {r["concept"]: r["steered"] for r in st if r["alpha"] == 0}
    s1m = {r["concept"]: bool(r["steered"]) and not base[r["concept"]] for r in st if r["alpha"] == 4}
    live = json.load(open(ROOT / "results/p2r_analysis.json"))["identify R4"]["strength 4"]["live_by_concept"]
    cs = sorted(set(s1m) & set(live))
    x, y = [live[c] for c in cs], [s1m[c] for c in cs]
    rep["identify_vs_s1m_gate"] = {
        "concepts": len(cs), "kappa": kappa(x, y),
        "both_live": sum(p and q for p, q in zip(x, y)), "identify_only": sum(p and not q for p, q in zip(x, y)),
        "gate_only": sum(q and not p for p, q in zip(x, y)), "both_dead": sum(not p and not q for p, q in zip(x, y)),
        "gate_dead_concepts": [c for c in cs if not s1m[c]]}
    R1 = rows(glob.glob(f"{a.dir}/*_forced_macar-release_p2r_fromtrial.jsonl")[0])
    py = {}
    for r in R1:
        py.setdefault(r["framing"], {}).setdefault(r["alpha"], {})[r["concept"]] = r["p_yes"]
    rep["p_yes_median"] = {f: {str(s): float(np.median(list(v.values()))) for s, v in sorted(d.items())}
                           for f, d in py.items()}
    fy = py["factual_yes"]
    d = np.array([fy[4.0][c] - fy[0.0][c] for c in fy[0.0]])
    rep["P1_probability_scale"] = {"median_change": float(np.median(d)),
                                   "wilcoxon_p": float(wilcoxon(d).pvalue) if np.any(d) else 1.0,
                                   "concepts_down": int((d < 0).sum()), "concepts": len(d)}
    T = {k: p2r.table(p2r.load(a.dir, tag)) for k, tag in p2r.CELLS.items() if k in ("R1", "R2", "R3")}
    for k in ("R2", "R3"):
        rep[f"P1 {k} by strength"] = {f"strength {x:g}": p2r.p1(T[k], x) for x in p2r.STRENGTHS}
        rep[f"P2 {k} by strength"] = {f"strength {x:g}": p2r.excess(T[k], x, p2r.FACTUAL) for x in p2r.STRENGTHS}
    rep["median_logodds_change"] = {k: {fr: {f"{x:g}": float(np.median(p2r.change(t, fr, x)[1])) for x in p2r.STRENGTHS}
                                        for fr in t} for k, t in T.items()}
    json.dump(rep, open(ROOT / "results/p2r_secondaries.json", "w"), indent=1)
    print(json.dumps(rep, indent=1)[:2500])


if __name__ == "__main__":
    main()
