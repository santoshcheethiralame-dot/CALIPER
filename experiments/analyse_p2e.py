"""Score P2-E once, as filed (docs/preregistration-p2e-eval-awareness.md).

    python experiments/analyse_p2e.py --file <unzipped>/p2e_llama8b.jsonl

Manipulation checks M2-M5 first; a dose's primary endpoints are computed only if every check
passes and the real arm still answers there. Everything else is descriptive.
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np
from scipy import stats

PRIMARY = 0.5
FACTUAL = ("factual_yes", "factual_no")


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def wilcoxon(diff):
    diff = np.asarray(diff, float)
    if not np.any(diff):
        return {"n": len(diff), "median_diff": 0.0, "p": 1.0}
    return {"n": len(diff), "median_diff": float(np.median(diff)),
            "p": float(stats.wilcoxon(diff, alternative="two-sided").pvalue)}


def excess(fx, fy, ex, ey):
    """P2-R's P2: median over eval prompts of observed change minus the change the factual
    regression (change on strength-0 log-odds) predicts at that prompt's baseline."""
    slope, icpt = np.polyfit(fx, fy, 1)
    return float(np.median(np.asarray(ey) - (icpt + slope * np.asarray(ex))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--out", default="results/p2e_analysis.json")
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()

    R = load(a.file)
    side_path = a.file.replace(".jsonl", "_sidecar.json")
    side = json.load(open(side_path)) if os.path.exists(side_path) else {}

    base = {(r["set"], r["i"]): r for r in R if r["arm"] == "clean"}
    by = defaultdict(dict)                       # (set, dose) -> arm -> {i: row}
    for r in R:
        if r["arm"] != "clean":
            by[(r["set"], r["dose"])].setdefault(r["arm"], {})[r["i"]] = r

    def clean(st, key="logodds"):
        return np.array([base[k][key] for k in sorted(base) if k[0] == st])

    checks = {}
    mass = {st: float(np.median(clean(st, "mass"))) for st in ("eval", "deploy", *FACTUAL)}
    checks["M2"] = {"median_mass": mass, "pass": all(m >= 0.5 for m in mass.values())}
    fy, fn = float(np.median(clean("factual_yes"))), float(np.median(clean("factual_no")))
    checks["M3"] = {"factual_yes": fy, "factual_no": fn, "pass": fy > 0 > fn}
    mw = stats.mannwhitneyu(clean("eval"), clean("deploy"), alternative="greater")
    checks["M4"] = {"eval_median": float(np.median(clean("eval"))),
                    "deploy_median": float(np.median(clean("deploy"))),
                    "p": float(mw.pvalue), "pass": bool(mw.pvalue < 0.05)}
    base_ok = all(checks[k]["pass"] for k in ("M2", "M3", "M4"))

    def delta(st, dose, arm):
        rows = by[(st, dose)].get(arm, {})
        return {i: r["logodds"] - base[(st, i)]["logodds"] for i, r in rows.items()}

    doses = sorted({r["dose"] for r in R if r["arm"] != "clean"})
    out = {"file": a.file, "layer": side.get("layer"), "layer_auroc": side.get("layer_auroc"),
           "checks": checks, "doses": {}}
    for dose in doses:
        imp = [r for (st, d), arms in by.items() if d == dose
               for arm, rows in arms.items() if "impact" in arm for r in rows.values()]
        m5 = float(np.mean([bool(r.get("matched")) for r in imp])) if imp else 0.0
        real_mass = float(np.median([r["mass"] for r in by[("eval", dose)].get("real", {}).values()]))
        cal = next((d for d in side.get("doses", []) if d.get("kl_target") == dose), {})
        D = {"M5_matched_share": m5, "calibration_matched": bool(cal.get("matched")),
             "real_eval_mass": real_mass,
             "scored": bool(base_ok and m5 >= 0.9 and cal.get("matched") and real_mass >= 0.5)}

        ids = sorted(delta("eval", dose, "real"))
        real = delta("eval", dose, "real")
        unrel_imp = [arm for arm in by[("eval", dose)] if arm.startswith("unrel_impact:")]
        unrel_norm = [arm for arm in by[("eval", dose)] if arm.startswith("unrel_norm:")]

        def med_abs(arms, st="eval"):
            ds = [delta(st, dose, arm) for arm in arms]
            return {i: float(np.median([abs(d[i]) for d in ds if i in d])) for i in ids}

        if D["scored"]:
            rnd = delta("eval", dose, "rand_impact")
            D["P1"] = wilcoxon([abs(real[i]) - abs(rnd[i]) for i in ids])
            u = med_abs(unrel_imp)
            D["P2"] = {**wilcoxon([abs(real[i]) - u[i] for i in ids]), "n_vectors": len(unrel_imp)}
            fx, fyv = [], []
            for st in FACTUAL:
                dd = delta(st, dose, "real")
                for i, v in dd.items():
                    fx.append(base[(st, i)]["logodds"])
                    fyv.append(v)
            ex = [base[("eval", i)]["logodds"] for i in ids]
            ey = [real[i] for i in ids]
            est = excess(fx, fyv, ex, ey)
            rng = np.random.default_rng(0)
            fx, fyv, ex, ey = map(np.asarray, (fx, fyv, ex, ey))
            boots = []
            for _ in range(a.boot):
                fi = rng.integers(0, len(fx), len(fx))
                ei = rng.integers(0, len(ex), len(ex))
                boots.append(excess(fx[fi], fyv[fi], ex[ei], ey[ei]))
            ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
            D["P3"] = {"excess": est, "ci95": ci, "specific": not (ci[0] <= 0 <= ci[1])}

        sec = {}
        signed = [real[i] for i in ids]
        sec["signed_real_eval"] = {"median": float(np.median(signed)),
                                   "n_negative": int(sum(x < 0 for x in signed)),
                                   "sign_p": float(stats.binomtest(int(sum(x < 0 for x in signed)),
                                                                   len(signed)).pvalue)}
        rn = delta("eval", dose, "rand_norm")
        if rn:
            sec["real_vs_rand_norm"] = wilcoxon([abs(real[i]) - abs(rn[i]) for i in ids])
        if unrel_norm:
            u = med_abs(unrel_norm)
            sec["real_vs_unrel_norm"] = wilcoxon([abs(real[i]) - u[i] for i in ids])
        sec["per_control_median_abs"] = {
            arm: float(np.median([abs(x) for x in delta("eval", dose, arm).values()]))
            for arm in sorted(unrel_imp + unrel_norm)}
        dep = delta("deploy", dose, "real")
        if dep:
            sec["signed_real_deploy"] = {"median": float(np.median(list(dep.values()))),
                                         "n_positive": int(sum(x > 0 for x in dep.values()))}
        sec["median_norm"] = {arm: float(np.median([r["norm"] for r in by[("eval", dose)][arm].values()]))
                              for arm in ("real", "rand_impact") if arm in by[("eval", dose)]
                              and all("norm" in r for r in by[("eval", dose)][arm].values())}
        D["secondary"] = sec
        out["doses"][str(dose)] = D

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    p = out["doses"].get(str(PRIMARY), {})
    print(f"checks: " + ", ".join(f"{k} {'pass' if v['pass'] else 'FAIL'}" for k, v in checks.items()))
    for k in ("P1", "P2", "P3"):
        print(f"{k} @ {PRIMARY}: {p.get(k, 'not scored')}")


if __name__ == "__main__":
    main()
