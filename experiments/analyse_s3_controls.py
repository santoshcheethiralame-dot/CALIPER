"""Pre-registered control analysis: is the first-token shift concept-specific?

Criteria fixed in the addendum to `docs/preregistration-s3-forced-choice.md`, filed before
the control run. Primary test: introspective P(YES) at alpha=6, real vectors versus
norm-matched random vectors, Wilcoxon signed-rank paired by concept, two-sided, 0.05.

  A1  real significantly above random  -> the shift carries concept information
  A2  real indistinguishable from random -> the model is responding to perturbation
                                            magnitude, not content
  A3  random significantly above real  -> anomaly, report without interpreting
"""

import json
import sys

import numpy as np
from scipy.stats import wilcoxon

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def load(path):
    out = {}
    for line in open(path, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            out.setdefault((r["framing"], r["alpha"]), {})[r["concept"]] = r["p_yes"]
    return out


def paired(a, b):
    ks = sorted(set(a) & set(b))
    return np.array([a[k] for k in ks]), np.array([b[k] for k in ks]), ks


def compare(a, b, label):
    x, y, _ = paired(a, b)
    stat, p = wilcoxon(x, y)
    return dict(label=label, mx=x.mean(), my=y.mean(), p=p,
                higher=int((x > y).sum()), n=len(x))


def main(real_path, random_path, shuffle_path=None):
    real, rnd = load(real_path), load(random_path)
    shuf = load(shuffle_path) if shuffle_path else None
    F = "introspective"

    print("PRIMARY TEST (pre-registered): real vs random, introspective, alpha=6\n")
    r = compare(real[(F, 6.0)], rnd[(F, 6.0)], "real vs random")
    print(f"  real   mean P(YES) = {r['mx']:.3f}")
    print(f"  random mean P(YES) = {r['my']:.3f}")
    print(f"  Wilcoxon p = {r['p']:.4f}, n = {r['n']} paired concepts, "
          f"real higher on {r['higher']}/{r['n']}")
    if r["p"] < 0.05:
        outcome = "A1 - real ABOVE random: the shift carries concept information" \
            if r["mx"] > r["my"] else "A3 - random ABOVE real: anomaly, do not interpret"
    else:
        outcome = ("A2 - real INDISTINGUISHABLE from random: the model is responding to "
                   "perturbation magnitude, not content")
    print(f"\n  PRE-REGISTERED OUTCOME: {outcome}")

    print("\n\nAll strengths, introspective framing\n")
    hdr = f"{'alpha':>6}{'real':>9}{'random':>9}{'shuffle':>9}{'p r-vs-rand':>13}{'p r-vs-shuf':>13}"
    print(hdr)
    print("-" * len(hdr))
    alphas = sorted({a for (f, a) in rnd if f == F})
    for a in alphas:
        if (F, a) not in real:
            continue
        mr = np.mean(list(real[(F, a)].values()))
        mn = np.mean(list(rnd[(F, a)].values()))
        ms = np.mean(list(shuf[(F, a)].values())) if shuf else float("nan")
        if a == 0:
            p1 = p2 = "-"
        else:
            p1 = f"{compare(real[(F, a)], rnd[(F, a)], '')['p']:.4f}"
            p2 = f"{compare(real[(F, a)], shuf[(F, a)], '')['p']:.4f}" if shuf else "-"
        print(f"{a:>6}{mr:>9.3f}{mn:>9.3f}{ms:>9.3f}{p1:>13}{p2:>13}")

    print("\n\nDid the controls move P(YES) off the floor at all?\n")
    for name, d in (("random", rnd), ("shuffle", shuf)):
        if d is None:
            continue
        print(f"  {name}:")
        for a in alphas:
            if a == 0:
                continue
            x, y, _ = paired(d[(F, a)], d[(F, 0.0)])
            print(f"    alpha={a}: {y.mean():.5f} -> {x.mean():.3f}, "
                  f"p = {wilcoxon(x, y).pvalue:.2e}, rose on {int((x > y).sum())}/{len(x)}")

    if shuf:
        print("\n\nDo the two controls agree with each other?\n")
        for a in alphas:
            if a == 0:
                continue
            c = compare(rnd[("introspective", a)], shuf[("introspective", a)], "")
            same = "agree" if c["p"] > 0.05 else "DIFFER"
            print(f"  alpha={a}: random {c['mx']:.3f} vs shuffle {c['my']:.3f}, "
                  f"p = {c['p']:.3f}  ({same})")


if __name__ == "__main__":
    main(*sys.argv[1:])
