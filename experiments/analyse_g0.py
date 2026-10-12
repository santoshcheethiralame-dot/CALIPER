"""Score G-0 (docs/preregistration-g0-gain-law.md) once, when all 50 rows exist.

    PYTHONPATH=. python experiments/analyse_g0.py      # writes results/g0_analysis.json

Primary: among fits that pass the identifiable label (1/gamma of ln_2 removed from fit and
reference), the share that fail the filed Euclidean label. Two rival bands at the observed number
of identifiable passes n: the gain law's (gain_law.forecast, c resampled from the GPT-2 layer-6
calibration, units resampled) and the smallest-gain rule's (binomial at the pooled GPT-2 layer-6 rate,
54/707 = 0.0764, frozen at filing).
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binom

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import gain_law as gl  # noqa: E402

STEM, N, LAYER, P_L6, MIN_PASS = "results/g0_gpt2_l0", 50, 0, 54 / 707, 20


def main():
    R = [json.loads(l) for l in open(ROOT / f"{STEM}.jsonl", encoding="utf-8") if l.strip()]
    if len(R) != N:
        raise SystemExit(f"{len(R)} of {N} rows; score only once all rows exist")
    g, _ = gl.weights(LAYER)
    u = (1 / g) / np.linalg.norm(1 / g)
    P = lambda x: x - u * (u @ x)
    rows = []
    for r in R:
        z = np.load(ROOT / f"{STEM}_dirs/n{r['_key']}.npz")
        w, v = z["w"].astype(float), z[r["picked"]][:, 0].astype(float)
        pv, pw = P(v), P(w)
        rows.append({"unit": r["_key"], "raw": abs(v @ w) / np.linalg.norm(v) / np.linalg.norm(w),
                     "ident": abs(pv @ pw) / np.linalg.norm(pv) / np.linalg.norm(pw),
                     "null_share": float((u @ v) ** 2 / (v @ v)), "restart": r["stability"], "r2": r["r2_k1"]})
    passes = [x for x in rows if x["ident"] >= gl.BAR]
    n = len(passes)
    share = float(np.mean([x["raw"] < gl.BAR for x in passes])) if n else None
    units = [x["unit"] for x in rows]
    fc = gl.forecast(LAYER, units, n_pass=n) if n else gl.forecast(LAYER, units)
    sg = [float(binom.ppf(0.025, n, P_L6) / n), float(binom.ppf(0.975, n, P_L6) / n)] if n else None
    if n < MIN_PASS:
        reading = "underpowered"
    else:
        gb = fc["band95_at_n"]["band"]
        in_g, in_sg = gb[0] <= share <= gb[1], sg[0] <= share <= sg[1]
        reading = ("gain law confirmed" if in_g and not in_sg else
                   "gain law not replicated" if in_sg and not in_g else "partial")
    # secondaries: c * sqrt(d) of the layer-0 fits, and restart agreement on filed failures
    G = dict(zip(units, gl.gains(LAYER, units)[0]))
    c = [np.sqrt(x["null_share"] / (1 - x["null_share"])) / G[x["unit"]] * np.sqrt(len(g)) for x in rows]
    filed_fail = [x for x in rows if x["raw"] < gl.BAR]
    rep = {"units": len(rows), "identifiable_passes": n, "filed_fail_share_among_passes": share,
           "gain_law": fc, "smallest_gain_band": sg, "reading": reading,
           "c_sqrt_d_median_layer0": float(np.median(c)), "c_sqrt_d_median_calibration": fc["c_sqrt_d_median"],
           "restart_agreement_median_on_filed_failures": float(np.median([x["restart"] for x in filed_fail])) if filed_fail else None,
           "filed_failures": len(filed_fail), "rows": rows}
    json.dump(rep, open(ROOT / "results/g0_analysis.json", "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in rep.items() if k != "rows"}, indent=1, default=float))


if __name__ == "__main__":
    main()
