"""Post hoc (17 Oct): the earlier conditions pooled on test R2 instead of same-document held-out R2.

    PYTHONPATH=. python experiments/pool_test_r2.py      # writes results/pool_test_r2.json

Not filed. The filed pool (analyse_every_run.py) cannot be re-run on test R2: B-7 and B-11c kept no
fitted directions. This pools every arm of results/test_r2_saved_arms.json that has independent units,
at least 10 identifiable failures and no near-constant coordinate, one arm per unit set (B-14r for
B-14/B-15's units, D-1, B-8b, X-1a, B-2c), fixed before the pooled values were computed. Both pools
use the same units and rows, so they differ only in which R2 is the check. DeLong SEs are recovered
from the stored two-sided p values; the random-effects pool and its HKSJ interval are
analyse_review_round1.hksj, unchanged.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import analyse_review_round1 as r1  # noqa: E402

ARMS = ("b14r", "d1", "b8b", "x1a", "b2c")


def main():
    t = json.load(open(ROOT / "results/test_r2_saved_arms.json"))
    rep = {"arms": list(ARMS), "label": "identifiable"}
    for check in ("restart_vs_heldout", "restart_vs_test"):
        d, se, per = [], [], {}
        for a in ARMS:
            c = t[a]["ident"][check]
            z = norm.isf(c["delong_p"] / 2)
            d.append(c["diff"])
            se.append(abs(c["diff"]) / z)
            per[a] = {"diff": c["diff"], "boot_ci95": c["boot_ci95"], "failures": c["failures"]}
        rep[check] = {"per_arm": per, **r1.hksj(d, se)}
        print(check, round(rep[check]["pooled"], 4), rep[check]["hksj_ci95"], flush=True)
    json.dump(rep, open(ROOT / "results/pool_test_r2.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
