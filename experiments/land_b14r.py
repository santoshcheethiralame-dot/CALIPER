"""Land B-14r (queue 8): the primary GPT-2 layer-6 run repeated with fitted directions saved.

    PYTHONPATH=. python experiments/land_b14r.py      # writes results/b14r_landing.json

B-14r runs the filed B-14 command unchanged except --out. Its direct-route fits reproduce B-14 to
the digit; its cascade-route fits do not, because B-14 ran before the cascade's head was seeded
per unit (babd0f5, 6 Oct), so B-14's cascade draws came from the process-wide random state and
cannot be reproduced. B-14r is therefore a replicate of the primary that differs only in the
cascade's initial draw. This script reports:
  1. completeness and row identity by route;
  2. verdict flips between B-14 and B-14r, which isolate the cascade-initialisation component of
     the single-unit error budget;
  3. B-14r's own pre-registered comparison (DeLong, stratified bootstrap) on the Euclidean label;
  4. B-14r on the identifiable label (1/gamma removed), as for every other arm (analyse_ln_null).
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT))
import analyse_ln_null as ln  # noqa: E402

FILED, RERUN = "results/b14_primary_gpt2_indep", "results/b14r_primary_gpt2_dirs"
PASS = 0.95


def rows(stem):
    return {json.loads(l)["_key"]: json.loads(l) for l in open(ROOT / f"{stem}.jsonl", encoding="utf-8")
            if l.strip()}


def main():
    a, b = rows(FILED), rows(RERUN)
    ks = sorted(set(a) & set(b))
    rep = {"filed_rows": len(a), "rerun_rows": len(b), "complete": len(b) == len(a) == 300}
    same = lambda f: int(sum(abs(a[k][f] - b[k][f]) < 1e-9 for k in ks))
    rep["identity"] = {"direct_route_identical": same("align_direct"),
                       "cascade_route_identical": same("align_cascade"),
                       "restart_agreement_identical": same("stability"),
                       "picked_route_changed": int(sum(a[k]["picked"] != b[k]["picked"] for k in ks)),
                       "of": len(ks)}
    fa = np.array([a[k]["align_selected"] < PASS for k in ks])
    fb = np.array([b[k]["align_selected"] < PASS for k in ks])
    n = len(ks)
    rep["verdicts"] = {"filed_failures": int(fa.sum()), "rerun_failures": int(fb.sum()),
                       "flips": int((fa != fb).sum()), "flip_rate": float((fa != fb).mean()),
                       "fail_to_pass": int((fa & ~fb).sum()), "pass_to_fail": int((~fa & fb).sum()),
                       "flips_among_cascade_picked_in_either": int(sum(
                           (a[k]["align_selected"] < PASS) != (b[k]["align_selected"] < PASS)
                           for k in ks if "cascade" in (a[k]["picked"], b[k]["picked"])))}
    if rep["complete"]:
        rep["rerun_as_an_arm"] = ln.arm("GPT-2 L6, primary replicate (B-14r)", RERUN, "gpt2", 6)
    print(json.dumps({k: v for k, v in rep.items()}, indent=1, default=float)[:3000])
    json.dump(rep, open(ROOT / "results/b14r_landing.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
