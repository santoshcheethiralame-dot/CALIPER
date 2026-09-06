"""Pre-registered analysis of the forced-choice run.

Criteria are fixed in `docs/preregistration-s3-forced-choice.md` and were filed before
the run. The primary test is a paired Wilcoxon signed-rank on P(YES) at alpha=6 versus
alpha=0 under the introspective framing, two-sided, 0.05.

Pairing is by concept, which also neutralises the trial-number confound in the
free-generation run: concept identity is perfectly confounded with trial index, and
comparing a concept against itself across strengths holds that constant.
"""

import json
import sys

import numpy as np
from scipy.stats import wilcoxon

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def load(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    out = {}
    for r in rows:
        out.setdefault((r["framing"], r["alpha"]), {})[r["concept"]] = r["p_yes"]
    return out


def paired(a, b):
    """Same concepts, same order, so the test is genuinely paired."""
    keys = sorted(set(a) & set(b))
    return np.array([a[k] for k in keys]), np.array([b[k] for k in keys])


def main(forced_path, gen_path=None):
    d = load(forced_path)
    alphas = sorted({a for (_f, a) in d})

    print("PRIMARY TEST (pre-registered): introspective, alpha=6 vs alpha=0\n")
    x, y = paired(d[("introspective", 6.0)], d[("introspective", 0.0)])
    stat, p = wilcoxon(x, y)
    print(f"  alpha=0  mean P(YES) = {y.mean():.5f}   median = {np.median(y):.5f}")
    print(f"  alpha=6  mean P(YES) = {x.mean():.5f}   median = {np.median(x):.5f}")
    print(f"  Wilcoxon W = {stat:.1f}, p = {p:.2e}, n = {len(x)} paired concepts")
    print(f"  concepts where P(YES) rose: {int((x > y).sum())}/{len(x)}")
    verdict = "A - first-token sensitivity EXISTS" if p < 0.05 and x.mean() > y.mean() \
        else ("C - injection SUPPRESSES the answer" if p < 0.05 else "B - no first-token shift")
    print(f"\n  PRE-REGISTERED OUTCOME: {verdict}")

    print("\n\nFull dose-response, both framings\n")
    print(f"{'alpha':>6} | {'introspective':>26} | {'neutral':>26}")
    print(f"{'':>6} | {'mean':>8}{'P>0.5':>8}{'p vs a=0':>10} | "
          f"{'mean':>8}{'P>0.5':>8}{'p vs a=0':>10}")
    print("-" * 68)
    for a in alphas:
        cells = []
        for f in ("introspective", "neutral"):
            v = d[(f, a)]
            arr = np.array(list(v.values()))
            if a == 0:
                ptxt = "-"
            else:
                xx, yy = paired(v, d[(f, 0.0)])
                ptxt = f"{wilcoxon(xx, yy).pvalue:.1e}"
            cells.append(f"{arr.mean():>8.3f}{(arr > 0.5).mean():>8.0%}{ptxt:>10}")
        print(f"{a:>6} | {cells[0]} | {cells[1]}")

    print("\n\nSECONDARY (pre-registered): is the shift specific to self-directed framing?\n")
    for a in alphas:
        if a == 0:
            continue
        xi, yi = paired(d[("introspective", a)], d[("introspective", 0.0)])
        xn, yn = paired(d[("neutral", a)], d[("neutral", 0.0)])
        print(f"  alpha={a}: introspective rises {yi.mean():.3f} -> {xi.mean():.3f} "
              f"(+{xi.mean()-yi.mean():+.3f}) | "
              f"neutral {yn.mean():.3f} -> {xn.mean():.3f} ({xn.mean()-yn.mean():+.3f})")

    if gen_path:
        print("\n\nDoes the first-token measure agree with the generated answers?\n")
        gen = [json.loads(l) for l in open(gen_path, encoding="utf-8") if l.strip()]
        sys.path.insert(0, "experiments")
        from rescore_s3 import first_answer, names_concept
        print(f"{'alpha':>6}{'generated YES':>16}{'named concept':>16}{'first-token P(YES)':>21}")
        for a in alphas:
            g = [r for r in gen if r["alpha"] == a]
            if not g:
                continue
            gy = sum(first_answer(r["text"]) == "YES" for r in g) / len(g)
            gn = sum(names_concept(r["text"], r["concept"]) for r in g) / len(g)
            ft = np.mean(list(d[("introspective", float(a))].values()))
            print(f"{a:>6}{gy:>15.0%}{gn:>16.0%}{ft:>21.3f}")


if __name__ == "__main__":
    main(*sys.argv[1:])
