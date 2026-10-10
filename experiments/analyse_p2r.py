"""P2-R scorer (docs/preregistration-p2r-released-protocol.md), written before the data.

    python experiments/analyse_p2r.py --dir <unzipped session> [--out results/p2r_analysis.json]

Finds the four cells by their file tags, runs the manipulation checks, then P1 (yes-bias against
flattening), P2 (framing-specific excess over a baseline-matched factual regression) and P3 (real
against impact-matched random at the released strength), all on log-odds, plus the filed
secondaries.
"""
import argparse
import glob
import json
import os

import numpy as np
from scipy.stats import rankdata, wilcoxon

CELLS = {"R1": "_forced_macar-release_p2r_fromtrial", "R2": "_forced_random_macar-release_p2r_fromtrial",
         "R3": "_forced_random-impact_macar-release_p2r_fromtrial",
         "R4": "_identify_macar-release_fromtrial", "R4-random": "_identify_random_macar-release_fromtrial"}
FACTUAL = ("factual_no", "factual_yes", "factual_contested")
STRENGTH = 4.0
NB = 2000


def load(d, tag):
    f = [p for p in glob.glob(os.path.join(d, "**", f"*{tag}.jsonl"), recursive=True)]
    if not f:
        return None
    return [json.loads(l) for l in open(f[0], encoding="utf-8") if l.strip()]


def table(R):
    """framing -> strength -> concept -> row."""
    t = {}
    for r in R:
        t.setdefault(r["framing"], {}).setdefault(r["alpha"], {})[r["concept"]] = r
    return t


def lo(r):
    return r["logp_yes"] - r["logp_no"]


def change(t, fr, s):
    c = sorted(set(t[fr][0.0]) & set(t[fr][s]))
    return c, np.array([lo(t[fr][s][k]) - lo(t[fr][0.0][k]) for k in c]), np.array([lo(t[fr][0.0][k]) for k in c])


def wil(x):
    x = np.asarray(x)
    return float(wilcoxon(x).pvalue) if np.any(x != 0) else 1.0


def auc(x1, x0):
    x = np.r_[x1, x0]
    r = rankdata(x)
    m, n = len(x1), len(x0)
    return float((r[:m].sum() - m * (m + 1) / 2) / (m * n))


def checks(cells):
    out = {}
    base = {k: table(v) for k, v in cells.items() if v and k in ("R1", "R2", "R3")}
    if len(base) > 1:
        ref = base.get("R1") or next(iter(base.values()))
        same = all(abs(lo(ref[fr][0.0][c]) - lo(t[fr][0.0][c])) < 1e-6
                   for t in base.values() for fr in ref for c in ref[fr][0.0] if c in t.get(fr, {}).get(0.0, {}))
        out["strength0_identical_across_cells"] = bool(same)
    t = base.get("R1")
    if t:
        out["kl_zero_at_strength0"] = all(r["kl"] == 0 for fr in t for r in t[fr][0.0].values())
        out["median_mass_at_strength0"] = {fr: float(np.median([r["yesno_mass"] for r in t[fr][0.0].values()]))
                                           for fr in t}
        out["framings_passing_mass"] = [fr for fr, m in out["median_mass_at_strength0"].items() if m >= 0.5]
        med = {fr: float(np.median([lo(r) for r in t[fr][0.0].values()])) for fr in ("factual_no", "factual_yes")}
        out["factual_baselines_opposite_sign"] = bool(med["factual_no"] < 0 < med["factual_yes"])
        out["factual_baseline_median_logodds"] = med
    return out


def p1(t, s):
    c, d, _ = change(t, "factual_yes", s)
    return {"median_change": float(np.median(d)), "wilcoxon_p": wil(d), "concepts": len(c),
            "reading": ("affirmative shift" if np.median(d) > 0 else "flattening") if wil(d) < 0.05 else "neither"}


def excess(t, s, fit_on):
    X, Y = [], []
    for fr in fit_on:
        _, d, b = change(t, fr, s)
        X += list(b)
        Y += list(d)
    X, Y = np.array(X), np.array(Y)
    ci, di, bi = change(t, "released_introspective", s)
    rng = np.random.default_rng(0)

    def stat(ix_f, ix_i):
        A = np.c_[np.ones(len(ix_f)), X[ix_f]]
        beta = np.linalg.lstsq(A, Y[ix_f], rcond=None)[0]
        return float(np.median(di[ix_i] - (beta[0] + beta[1] * bi[ix_i])))
    point = stat(np.arange(len(X)), np.arange(len(di)))
    boots = [stat(rng.integers(0, len(X), len(X)), rng.integers(0, len(di), len(di))) for _ in range(NB)]
    lo_, hi_ = np.percentile(boots, [2.5, 97.5])
    return {"excess_median": point, "ci95": [float(lo_), float(hi_)], "framing_specific": bool(lo_ > 0),
            "fit_items": int(len(X)), "introspective_baseline_median": float(np.median(bi))}


def p3(real, rand, s):
    c = sorted(set(real["released_introspective"][s]) & set(rand["released_introspective"][s]))
    dr = np.array([lo(real["released_introspective"][s][k]) - lo(real["released_introspective"][0.0][k]) for k in c])
    dz = np.array([lo(rand["released_introspective"][s][k]) - lo(rand["released_introspective"][0.0][k]) for k in c])
    return {"real_median_change": float(np.median(dr)), "random_median_change": float(np.median(dz)),
            "wilcoxon_p_real_minus_random": wil(dr - dz),
            "content_free_share": float(np.median(dz) / np.median(dr)) if np.median(dr) else None}


def identify(R, s):
    t = table(R)["released_identify"]
    c = sorted(set(t[0.0]) & set(t[s]))
    d = np.array([t[s][k]["logp_concept"] - t[0.0][k]["logp_concept"] for k in c])
    return {"median_change_logp_concept": float(np.median(d)), "identify_live": int((d >= 2).sum()),
            "concepts": len(c), "live_by_concept": {k: bool(x >= 2) for k, x in zip(c, d)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", default="results/p2r_analysis.json")
    a = ap.parse_args()
    cells = {k: load(a.dir, tag) for k, tag in CELLS.items()}
    rep = {"cells_found": {k: (len(v) if v else 0) for k, v in cells.items()}, "checks": checks(cells)}
    T = {k: table(v) for k, v in cells.items() if v and k in ("R1", "R2", "R3")}
    ok = set(rep["checks"].get("framings_passing_mass", []))
    if "R1" in T and {"factual_yes"} <= ok:
        rep["P1"] = {f"strength {s:g}": p1(T["R1"], s) for s in (1.0, 2.0, 4.0, 8.0)}
        for k in ("R2", "R3"):
            if k in T:
                rep[f"P1 {k}"] = p1(T[k], STRENGTH)
    if "R1" in T and {"released_introspective", *FACTUAL} <= ok:
        rep["P2"] = {f"strength {s:g}": {"all factual": excess(T["R1"], s, FACTUAL),
                                          "contested only": excess(T["R1"], s, ("factual_contested",))}
                     for s in (1.0, 2.0, 4.0, 8.0)}
    if {"R1", "R3"} <= set(T) and "released_introspective" in ok:
        rep["P3"] = {f"strength {s:g}": {"impact-matched": p3(T["R1"], T["R3"], s),
                                          **({"norm-matched": p3(T["R1"], T["R2"], s)} if "R2" in T else {})}
                     for s in (1.0, 2.0, 4.0, 8.0)}
    if "R1" in T and "released_neutral" in T["R1"]:
        n = T["R1"]["released_neutral"]
        rep["neutral_injected_vs_uninjected_auc"] = {
            f"strength {s:g}": auc([lo(r) for r in n[s].values()], [lo(r) for r in n[0.0].values()])
            for s in sorted(n) if s}
    if "R1" in T:
        rep["yesno_mass_median"] = {fr: {f"{s:g}": float(np.median([r["yesno_mass"] for r in T["R1"][fr][s].values()]))
                                         for s in sorted(T["R1"][fr])} for fr in T["R1"]}
    for k in ("R4", "R4-random"):
        if cells[k]:
            rep[f"identify {k}"] = {f"strength {s:g}": identify(cells[k], s) for s in (1.0, 2.0, 4.0, 8.0)
                                    if any(r["alpha"] == s for r in cells[k])}
    print(json.dumps({k: v for k, v in rep.items() if k.startswith(("cells", "checks", "P"))}, indent=1)[:4000])
    json.dump(rep, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
