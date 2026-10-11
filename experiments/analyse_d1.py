"""Score D-1 (docs/preregistration-d1-fresh-units.md) once, when all 200 rows exist.

    PYTHONPATH=. python experiments/analyse_d1.py                        # writes results/d1_analysis.json
    PYTHONPATH=. python experiments/analyse_d1.py --stem results/b15a_fitseed1 --dev   # development run

Primary: on the identifiable label (1/gamma of ln_2 removed from fit and reference), AUC(restart
agreement) minus AUC(held-out R2) over all failures vs passes; it holds when the stratified bootstrap
95% interval lies below 0. Fewer than 10 identifiable failures: reported as underpowered.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT))
import analyse_ln_null as ln  # noqa: E402

STEM, N, MIN_FAIL = "results/d1_gpt2_l6_fresh", 200, 10


def reading(c, n_fail):
    if n_fail < MIN_FAIL or c is None:
        return "underpowered"
    lo, hi = c["boot_ci95"]
    return "holds" if hi < 0 else ("reversed" if lo > 0 else "not replicated")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", default=STEM)
    ap.add_argument("--dev", action="store_true", help="score an earlier arm; skips the row-count check")
    a = ap.parse_args()
    R = [json.loads(l) for l in open(ROOT / f"{a.stem}.jsonl", encoding="utf-8") if l.strip()]
    if not a.dev and len(R) != N:
        raise SystemExit(f"{len(R)} of {N} rows; score only once all rows exist")
    arm = ln.arm("D-1", a.stem, "gpt2", 6)
    ident, euc = arm["identifiable"], arm["as filed"]
    own = np.array([float(np.max(np.load(ROOT / f"{a.stem}_dirs/n{r['_key']}.npz")["direct_r2_restarts"]))
                    for r in R])
    st = np.array([r["stability"] for r in R])
    lab = np.array([r["align_direct"] < ln.PASS for r in R])
    rep = {"stem": a.stem, "units": len(R), "dev": a.dev,
           "primary": {"label": "identifiable", "comparison": ident["primary_comparison"],
                       "reading": reading(ident["primary_comparison"], ident["failures"])},
           "euclidean": {"comparison": euc["primary_comparison"],
                         "reading": reading(euc["primary_comparison"], euc["failures"])},
           "direct_route_own_r2": ln.compare(-st, -own, lab),
           "arm": arm}
    out = ROOT / ("results/d1_dev.json" if a.dev else "results/d1_analysis.json")
    json.dump(rep, open(out, "w"), indent=1, default=float)
    for k in ("primary", "euclidean"):
        c = rep[k]["comparison"]
        print(k, rep[k]["reading"], c and round(c["diff"], 3), c and [round(x, 3) for x in c["boot_ci95"]])


if __name__ == "__main__":
    main()
