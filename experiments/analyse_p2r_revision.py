"""P2-R analyses requested by the second internal review of Paper 2 (exploratory, after the data).

    python experiments/analyse_p2r_revision.py      # writes results/p2r_revision.json

1. Introspective excess for real, norm-matched and impact-matched random vectors, and the paired
   per-concept difference real minus each random control (Wilcoxon; concept bootstrap of the median).
2. The baseline-matched regression's intercept and slope with a concept bootstrap: a positive
   intercept is a shift toward YES that the factual baselines do not predict.
3. An item-level answering gate (YES+NO mass >= 0.5 at strength 0 and at the strength scored),
   applied alike to every framing and endpoint: P1 on factual-YES, the change on every framing, P3.
4. The content-free share (median random change / median real change) with a concept bootstrap.
5. The identify stage at every strength: the filed 2-nat rule and a rank rule (concept token among
   the top 10 first tokens), real against random, and the AUC of the change in log P(concept).
6. Answers: concepts whose introspective P(YES) > 0.5 while YES+NO mass >= 0.5, per cell and strength.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import analyse_p2r as p2r  # noqa: E402

D = str(ROOT / "results/p2r_gemma27")
NB, GATE, RANK = 2000, 0.5, 10
lo = p2r.lo


def residuals(t, s, cs, fit_cs=None):
    """Per-concept introspective change minus the factual regression's prediction at its baseline."""
    fit_cs = cs if fit_cs is None else fit_cs
    X = np.array([lo(t[fr][0.0][c]) for c in fit_cs for fr in p2r.FACTUAL])
    Y = np.array([lo(t[fr][s][c]) - lo(t[fr][0.0][c]) for c in fit_cs for fr in p2r.FACTUAL])
    b = np.linalg.lstsq(np.c_[np.ones(len(X)), X], Y, rcond=None)[0]
    di = np.array([lo(t["released_introspective"][s][c]) - lo(t["released_introspective"][0.0][c]) for c in cs])
    bi = np.array([lo(t["released_introspective"][0.0][c]) for c in cs])
    return di - (b[0] + b[1] * bi), b


def pct(x):
    x = [v for v in x if np.isfinite(v)]
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def answering(t, fr, s, c):
    return t[fr][0.0][c]["yesno_mass"] >= GATE and t[fr][s][c]["yesno_mass"] >= GATE


def wil(x):
    x = np.asarray(x)
    return float(wilcoxon(x).pvalue) if len(x) > 5 and np.any(x != 0) else None


def main():
    T = {k: p2r.table(p2r.load(D, tag)) for k, tag in p2r.CELLS.items() if k in ("R1", "R2", "R3")}
    cs = sorted(T["R1"]["released_introspective"][0.0])
    rng = np.random.default_rng(0)
    boots = [rng.integers(0, len(cs), len(cs)) for _ in range(NB)]
    rep = {"note": "exploratory; requested by the second internal review of Paper 2 (11 Oct 2026)"}
    for s in p2r.STRENGTHS:
        k = f"strength {s:g}"
        out = {}
        res = {c: residuals(T[c], s, cs) for c in T}
        for c in T:
            r, b = res[c]
            bb = [residuals(T[c], s, [cs[i] for i in ix]) for ix in boots]
            out[f"{c} excess"] = {"median": float(np.median(r)), "ci95": pct([np.median(x[0]) for x in bb]),
                                  "intercept": float(b[0]), "intercept_ci95": pct([x[1][0] for x in bb]),
                                  "slope": float(b[1])}
        for c in ("R2", "R3"):
            d = res["R1"][0] - res[c][0]
            out[f"R1 minus {c} excess, paired"] = {"median": float(np.median(d)), "wilcoxon_p": wil(d),
                                                   "ci95": pct([np.median(d[ix]) for ix in boots])}
        # item-level answering gate
        gate = {}
        for fr in T["R1"]:
            ok = [c for c in cs if answering(T["R1"], fr, s, c)]
            d = [lo(T["R1"][fr][s][c]) - lo(T["R1"][fr][0.0][c]) for c in ok]
            gate[fr] = {"answering": len(ok), "median_change": float(np.median(d)) if d else None, "wilcoxon_p": wil(d)}
        ok = [c for c in cs if answering(T["R1"], "released_introspective", s, c)
              and answering(T["R3"], "released_introspective", s, c)]
        dr = np.array([lo(T["R1"]["released_introspective"][s][c]) - lo(T["R1"]["released_introspective"][0.0][c]) for c in ok])
        dz = np.array([lo(T["R3"]["released_introspective"][s][c]) - lo(T["R3"]["released_introspective"][0.0][c]) for c in ok])
        gate["P3 real minus impact-matched, answering in both"] = {
            "concepts": len(ok), "median_real": float(np.median(dr)) if len(ok) else None,
            "median_random": float(np.median(dz)) if len(ok) else None, "wilcoxon_p": wil(dr - dz) if len(ok) else None}
        out["answering gate"] = gate
        # content-free share with a concept bootstrap
        for c in ("R2", "R3"):
            dr = np.array([lo(T["R1"]["released_introspective"][s][x]) - lo(T["R1"]["released_introspective"][0.0][x]) for x in cs])
            dz = np.array([lo(T[c]["released_introspective"][s][x]) - lo(T[c]["released_introspective"][0.0][x]) for x in cs])
            sh = lambda ix: np.median(dz[ix]) / np.median(dr[ix]) if np.median(dr[ix]) else np.nan
            out[f"content-free share {c}"] = {"share": float(sh(np.arange(len(cs)))), "ci95": pct([sh(ix) for ix in boots])}
        # answers
        out["introspective YES answers"] = {
            c: int(sum(T[c]["released_introspective"][s][x]["p_yes"] > 0.5 and
                       T[c]["released_introspective"][s][x]["yesno_mass"] >= GATE for x in cs)) for c in T}
        rep[k] = out
    # identify at every strength
    R4 = p2r.table(p2r.load(D, p2r.CELLS["R4"]))["released_identify"]
    R4r = p2r.table(p2r.load(D, p2r.CELLS["R4-random"]))["released_identify"]
    idf = {}
    for s in p2r.STRENGTHS:
        dr = np.array([R4[s][c]["logp_concept"] - R4[0.0][c]["logp_concept"] for c in cs])
        dz = np.array([R4r[s][c]["logp_concept"] - R4r[0.0][c]["logp_concept"] for c in cs])
        idf[f"strength {s:g}"] = {"two_nat_real": int((dr >= 2).sum()), "two_nat_random": int((dz >= 2).sum()),
                                  "rank_top10_real": int(sum(R4[s][c]["concept_rank"] < RANK for c in cs)),
                                  "rank_top10_random": int(sum(R4r[s][c]["concept_rank"] < RANK for c in cs)),
                                  "auc_change_real_vs_random": p2r.auc(dr, dz)}
    rep["identify"] = idf
    json.dump(rep, open(ROOT / "results/p2r_revision.json", "w"), indent=1)
    for s in ("strength 1", "strength 2", "strength 4"):
        o = rep[s]
        print(s, "| excess R1/R2/R3", [round(o[f'{c} excess']['median'], 2) for c in ('R1', 'R2', 'R3')],
              "| R1-R3", round(o["R1 minus R3 excess, paired"]["median"], 2), o["R1 minus R3 excess, paired"]["wilcoxon_p"],
              [round(x, 2) for x in o["R1 minus R3 excess, paired"]["ci95"]],
              "| R1-R2", round(o["R1 minus R2 excess, paired"]["median"], 2), o["R1 minus R2 excess, paired"]["wilcoxon_p"],
              "| icpt R1", round(o["R1 excess"]["intercept"], 2), [round(x, 2) for x in o["R1 excess"]["intercept_ci95"]],
              "R3", round(o["R3 excess"]["intercept"], 2), [round(x, 2) for x in o["R3 excess"]["intercept_ci95"]])
        print("   gate", {f: (v.get("answering", v.get("concepts")), v.get("median_change", v.get("median_real")) and round(v.get("median_change") or v.get("median_real"), 2), v["wilcoxon_p"] and round(v["wilcoxon_p"], 4)) for f, v in o["answering gate"].items()})
        print("   share R3", round(o["content-free share R3"]["share"], 2), [round(x, 2) for x in o["content-free share R3"]["ci95"]],
              "R2", round(o["content-free share R2"]["share"], 2), "| YES answers", o["introspective YES answers"])
    print("identify", json.dumps(idf))


if __name__ == "__main__":
    main()
