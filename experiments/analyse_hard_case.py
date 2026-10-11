"""Paper 1, round-3 review items: the hard case defined without either compared check, and
unit-clustered intervals for the diagnostics (exploratory, post hoc).

    PYTHONPATH=. python experiments/analyse_hard_case.py     # writes results/hard_case.json

Hard case. On the identifiable label (1/gamma removed), a failing fit is "data-limited" when at least
half of its error energy lies in the eigen-directions carrying the bottom 1% of stimulus variance
(err_low_ident, from analyse_a4_spectrum.py), and "other" otherwise. The definition uses the ground
truth and the stimulus only: neither restart agreement nor held-out R2. Thresholds 0.3 and 0.7 are
reported as sensitivity. Coupling, stated: a fit can predict almost perfectly only if its error lies
where the stimulus barely varies, so held-out R2 is expected to rank data-limited failures as safe;
the class is independent of the checks by definition, not by mechanism.

Per arm: class counts, overlap with the R2-defined converged-wrong class, and both checks' AUCs for
data-limited failures against passes. Class stability: the four GPT-2 layer-6 fits with saved
directions (B-14r, B-15a/b/c) share their first 100 units.

Clustered intervals. The diagnostics (null share, low-variance share with u removed) and both checks
are scored on the identifiable label for converged-wrong fits against passes, per arm and pooled. In
the pooled bootstrap a cluster is a (model, layer, unit): every fit of a unit is resampled together,
so B-15a/b/c and X-1a/b no longer count as independent units.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
import analyse_a4_spectrum as a4  # noqa: E402
from analyse_review_round1 import auc  # noqa: E402

PASS, WB, NB = 0.95, 0.99, 2000
B14R = ("GPT-2 L6 primary replicate (B-14r)", "results/b14r_primary_gpt2_dirs", "gpt2", 6, 0, False, False)
SUB = {"GPT-2 L6": "gpt2-L6", "GPT-Neo-125m L10": "neo-L10", "GPT-Neo-125m L6": "neo-L6", "Pythia-160m L6": "pythia-L6"}


def sub(arm):
    return next(v for k, v in SUB.items() if arm.startswith(k))


def safe_auc(score, y):
    return auc(score, y) if 0 < y.sum() < len(y) else None


def boot(fn, clusters, rng):
    ids = np.unique(clusters)
    idx = {c: np.flatnonzero(clusters == c) for c in ids}
    out = []
    for _ in range(NB):
        pick = np.concatenate([idx[c] for c in rng.choice(ids, len(ids))])
        v = fn(pick)
        if v is not None:
            out.append(v)
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))] if len(out) > NB // 2 else None


def table(recs):
    a = {k: np.array([r[k] for r in recs]) for k in ("ident", "r2", "restart", "err_low_ident", "null_share",
                                                      "low_share_u_out")}
    a["cluster"] = np.array([r["cluster"] for r in recs])
    return a


def hard_case(a, thr, rng):
    fail = a["ident"] < PASS
    hard = fail & (a["err_low_ident"] >= thr)
    cw = fail & (a["r2"] > WB)
    keep = hard | ~fail
    y = hard[keep]

    def diff(ix):
        k = keep[ix]
        yy = hard[ix][k]
        if not 0 < yy.sum() < len(yy):
            return None
        return auc(-a["restart"][ix][k], yy) - auc(-a["r2"][ix][k], yy)
    res = {"failures": int(fail.sum()), "data_limited": int(hard.sum()), "other_failures": int((fail & ~hard).sum()),
           "data_limited_and_r2_above_0.99": int((hard & cw).sum()), "converged_wrong_by_r2": int(cw.sum()),
           "auc_restart": safe_auc(-a["restart"][keep], y), "auc_r2": safe_auc(-a["r2"][keep], y)}
    if res["auc_restart"] is not None and hard.sum() >= 2:
        res["diff"] = res["auc_restart"] - res["auc_r2"]
        res["diff_ci95_clustered"] = boot(diff, a["cluster"], rng)
    return res


def diagnostics(a, rng):
    fail = a["ident"] < PASS
    cw = fail & (a["r2"] > WB)
    hi = (cw | ~fail) & (a["r2"] > WB)  # converged-wrong against passes with R2 > 0.99, as in a4
    y = cw[hi]
    if not 0 < y.sum() < len(y):
        return {"converged_wrong": int(cw.sum())}
    out = {"converged_wrong": int(cw.sum()), "passes_kept": int((~y).sum()),
           "clusters": int(len(np.unique(a["cluster"][hi])))}
    for name, s in (("null_share", a["null_share"]), ("low_share_u_out", a["low_share_u_out"]),
                    ("held_out_r2", -a["r2"]), ("restart", -a["restart"])):
        def f(ix, s=s):
            k = hi[ix]
            yy = cw[ix][k]
            return auc(s[ix][k], yy) if 0 < yy.sum() < len(yy) else None
        out[name] = {"auc": auc(s[hi], y), "ci95_clustered": boot(f, a["cluster"], rng)}
    return out


def main():
    spec = json.load(open(ROOT / "results/a4_spectrum.json"))
    arms = {k: v["units"] for k, v in spec.items() if "units" in v}
    rec, _ = a4.arm(*B14R)
    arms = {B14R[0]: rec, **arms}
    for name, recs in arms.items():
        for r in recs:
            r["cluster"] = f"{sub(name)}:{r['unit']}"
    rng = np.random.default_rng(0)
    rep = {"definition": "identifiable failure with >= thr of its error energy in the bottom-1% variance directions",
           "per_arm": {}, "pooled": {}}
    for name, recs in arms.items():
        a = table(recs)
        rep["per_arm"][name] = {f"thr {t}": hard_case(a, t, rng) for t in (0.3, 0.5, 0.7)}
        rep["per_arm"][name]["diagnostics"] = diagnostics(a, rng)
    allrec = [r for recs in arms.values() for r in recs]
    a = table(allrec)
    rep["pooled"] = {f"thr {t}": hard_case(a, t, rng) for t in (0.3, 0.5, 0.7)}
    rep["pooled"]["diagnostics"] = diagnostics(a, rng)
    rep["pooled"]["fits"], rep["pooled"]["clusters"] = len(allrec), int(len(np.unique(a["cluster"])))
    gpt = ["GPT-2 L6 primary replicate (B-14r)", "GPT-2 L6 re-fit, fit seed 1 (B-15a)",
           "GPT-2 L6 re-fit, corpus seed 1 (B-15b)", "GPT-2 L6 re-fit, sequence split (B-15c)"]
    by = [{r["unit"]: r for r in arms[g]} for g in gpt]
    shared = sorted(set.intersection(*[set(b) for b in by]))
    k = [sum(b[u]["ident"] < PASS and b[u]["err_low_ident"] >= 0.5 for b in by) for u in shared]
    f = [sum(b[u]["ident"] < PASS for b in by) for u in shared]
    rep["class_stability_gpt2"] = {"fits": len(gpt), "shared_units": len(shared),
                                   "data_limited_in_k_fits": {str(i): k.count(i) for i in range(len(gpt) + 1)},
                                   "failing_in_k_fits": {str(i): f.count(i) for i in range(len(gpt) + 1)}}
    json.dump(rep, open(ROOT / "results/hard_case.json", "w"), indent=1, default=float)
    p = rep["pooled"]
    print(json.dumps({"pooled thr 0.5": p["thr 0.5"], "pooled diagnostics": p["diagnostics"],
                      "stability": rep["class_stability_gpt2"]}, indent=1, default=float))


if __name__ == "__main__":
    main()
