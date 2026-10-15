"""Score E-2 once, when both arms and E-1's matching arms are complete (docs/preregistration-e2-mid.md).

    python experiments/analyse_e2.py          # writes results/e2_analysis.json

Each arm is scored with E-1's scorer unchanged (identifiable label primary). Added for E-2:
  cross_family   |cos| between the MID fit and E-1's fit of the same unit, as a check on the MID fit
  mcnemar        per-unit verdicts of the two estimator families on the same units
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT))
import analyse_e1 as e1  # noqa: E402
import analyse_ln_null as ln  # noqa: E402

ARMS = {"gpt2_l6": ("results/e2_mid_gpt2_l6", "results/e1_gpt2_l6", "gpt2", 6, 200),
        "gptneo_l10": ("results/e2_mid_gptneo_l10", "results/e1_gptneo_l10", "EleutherAI/gpt-neo-125M", 10, 100)}


def main():
    rep = {}
    for name, (stem, e1_stem, model, layer, n_units) in ARMS.items():
        for s in (stem, e1_stem):
            if len(e1.load(s)) < n_units:
                raise SystemExit(f"{s}: {len(e1.load(s))} of {n_units} rows; score only once all rows exist")
        out, rows = e1.score_arm(stem, model, layer)
        _, rows_ln = e1.score_arm(e1_stem, model, layer)
        ln_by = {r["_key"]: r for r in rows_ln}
        cross = []
        for r in rows:
            a = np.load(ROOT / f"{stem}_dirs/n{r['_key']}.npz")
            b = np.load(ROOT / f"{e1_stem}_dirs/n{r['_key']}.npz")
            va, vb = a["restarts"][int(a["selected"])], b["restarts"][int(b["selected"])]
            cross.append(abs(float(va @ vb) / (np.linalg.norm(va) * np.linalg.norm(vb))))
        cross = np.array(cross)
        fail = np.array([r["align_ident"] < e1.PASS for r in rows])
        ra = np.array([r["restart_agreement"] for r in rows])
        te = np.array([r["test_r2"] for r in rows])
        out["cross_family"] = {"restart_vs_cross": ln.compare(-ra, -cross, fail),
                               "cross_vs_test": ln.compare(-cross, -te, fail)}
        both = [(ln_by[r["_key"]]["align_ident"] >= e1.PASS, r["align_ident"] >= e1.PASS)
                for r in rows if r["_key"] in ln_by]
        b_only, c_only = sum(x and not y for x, y in both), sum(y and not x for x, y in both)
        out["mcnemar"] = {"units": len(both), "pass_ln": int(sum(x for x, _ in both)),
                          "pass_mid": int(sum(y for _, y in both)), "ln_only": int(b_only), "mid_only": int(c_only),
                          "p": float(binomtest(b_only, b_only + c_only, 0.5).pvalue) if b_only + c_only else 1.0}
        c = out["cross_family"]["restart_vs_cross"]
        out["P2_reading"] = e1.reading(c, out["failures_identifiable"])
        rep[name] = out
    json.dump(rep, open(ROOT / "results/e2_analysis.json", "w"), indent=1, default=float)
    for name, o in rep.items():
        c, x = o["primary"], o["cross_family"]["restart_vs_cross"]
        print(name, "fails", o["failures_identifiable"], "P1", o["reading"], c and round(c["diff"], 3),
              "P2", o["P2_reading"], x and round(x["diff"], 3), "mcnemar", o["mcnemar"])


if __name__ == "__main__":
    main()
