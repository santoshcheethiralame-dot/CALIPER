"""Paper 2, C1 and C3 restated after the second internal review (exploratory, after the data).

    python experiments/analyse_p2_c1_c3.py      # writes results/p2_c1_c3.json

C1. Per health check, the within-recipe, concrete-only AUCs (results/p2_review_analysis.json, A3)
are pooled with inverse-variance weights (Hanley-McNeil variances, AUC clamped to [0.05, 0.95]
for the variance so that a cell at 0 or 1 does not get zero variance) and tested for heterogeneity
(Cochran's Q). A check whose cells agree around 0.5 is consistently uninformative; one whose cells
disagree beyond sampling noise is inconsistent. Cells with no live or no dead vector are skipped.
Also the concrete-only (all recipes) AUCs per session.

C3. Fisher's combination of the per-session P(YES)-shift inversion tests, one-sided (as stored in A4)
and two-sided (the filed S-2 prediction was two-sided: CI including 0.5), and the AUCs by recipe.
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import chi2, norm

ROOT = Path(__file__).resolve().parents[1]


def hm_var(a, n1, n0):
    q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
    return (a * (1 - a) + (n1 - 1) * (q1 - a * a) + (n0 - 1) * (q2 - a * a)) / (n1 * n0)


def main():
    d = json.load(open(ROOT / "results/p2_review_analysis.json"))
    a3, a4 = d["A3 C1 by concept type and recipe"], d["A4 C3 inversion"]
    cells = {}
    for sess, v in a3.items():
        for arm, checks in v.get("within_recipe_concrete_only", {}).items():
            for chk, c in checks.items():
                n1, n0 = c["live"], c["n"] - c["live"]
                if n1 and n0:
                    cells.setdefault(chk, []).append({"cell": f"{sess} / {arm}", "auc": c["auc"], "live": n1, "dead": n0,
                                                      "var": hm_var(min(max(c["auc"], 0.05), 0.95), n1, n0)})
    c1 = {}
    for chk, cs in cells.items():
        a = np.array([c["auc"] for c in cs])
        w = 1 / np.array([c["var"] for c in cs])
        pooled = float((w * a).sum() / w.sum())
        se = float(1 / np.sqrt(w.sum()))
        q = float((w * (a - pooled) ** 2).sum())
        c1[chk] = {"cells": len(cs), "range": [float(a.min()), float(a.max())], "pooled_auc": pooled,
                   "pooled_ci95": [pooled - 1.96 * se, pooled + 1.96 * se],
                   "pooled_p_vs_0.5": float(2 * norm.sf(abs(pooled - 0.5) / se)),
                   "cochran_q": q, "heterogeneity_p": float(chi2.sf(q, len(cs) - 1)) if len(cs) > 1 else None}
    concrete = {sess: {chk: v["concrete_only"][chk]["auc"] for chk in v.get("concrete_only", {})
                       if isinstance(v["concrete_only"][chk], dict)} for sess, v in a3.items() if "concrete_only" in v}
    one = [v["all"]["p_one_sided"] for k, v in a4.items() if isinstance(v, dict) and "all" in v]
    two = [min(1.0, 2 * min(p, 1 - p)) for p in one]
    fisher = lambda ps: float(chi2.sf(-2 * np.sum(np.log(ps)), 2 * len(ps)))
    c3 = {"fisher_one_sided": fisher(one), "fisher_two_sided": fisher(two), "per_session_one_sided": one,
          "per_session_two_sided": two,
          "by_recipe": {k: {arm: round(x["auc"], 3) for arm, x in v["by_recipe"].items()}
                        for k, v in a4.items() if isinstance(v, dict) and "by_recipe" in v}}
    rep = {"C1 within recipe, concrete only": c1, "C1 concrete only, all recipes": concrete, "C3": c3}
    json.dump(rep, open(ROOT / "results/p2_c1_c3.json", "w"), indent=1)
    for chk, v in c1.items():
        print(f"{chk:26s} cells {v['cells']} range {v['range'][0]:.2f}-{v['range'][1]:.2f} pooled {v['pooled_auc']:.2f} "
              f"[{v['pooled_ci95'][0]:.2f}, {v['pooled_ci95'][1]:.2f}] p(0.5) {v['pooled_p_vs_0.5']:.3f} "
              f"Q p {v['heterogeneity_p']:.3f}")
    print("concrete only", json.dumps(concrete))
    print("C3", json.dumps(c3))


if __name__ == "__main__":
    main()
