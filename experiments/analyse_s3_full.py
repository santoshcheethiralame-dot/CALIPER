"""Full analysis of the Study 3 forced-choice design: real, random, shuffle, two framings.

All tests are Wilcoxon signed-rank, paired by concept, two-sided, 0.05 - as fixed in
docs/preregistration-s3-forced-choice.md and its addendum before the runs.

  Primary   : introspective, alpha=6, real vs random                (A1 / A2 / A3)
  Secondary : introspective vs neutral_matched, real vectors, each alpha
  Sweep     : every strength, both framings, all three vector types (declared, exploratory)
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
    return np.array([a[k] for k in ks]), np.array([b[k] for k in ks])


def test(a, b):
    x, y = paired(a, b)
    return x.mean(), y.mean(), wilcoxon(x, y).pvalue, int((x > y).sum()), len(x)


def main(real_path, random_path, shuffle_path):
    D = {"real": load(real_path), "random": load(random_path), "shuffle": load(shuffle_path)}
    alphas = [0.0, 2.0, 4.0, 6.0]
    I, N = "introspective", "neutral_matched"

    # Sanity: alpha=0 must be identical across files (no injection, deterministic model).
    for f in (I, N):
        a0 = [tuple(sorted(D[k][(f, 0.0)].items())) for k in D]
        assert a0[0] == a0[1] == a0[2], f"alpha=0 differs across runs for {f}"
    print("sanity: alpha=0 rows identical across all three runs (deterministic)\n")

    print("=" * 76)
    print("PRIMARY (pre-registered): introspective, alpha=6, real vs random")
    print("=" * 76)
    mx, my, p, hi, n = test(D["real"][(I, 6.0)], D["random"][(I, 6.0)])
    print(f"  real {mx:.3f}  random {my:.3f}  p = {p:.3f}  real higher {hi}/{n}")
    out = ("A1 real above random" if p < 0.05 and mx > my else
           "A3 random above real" if p < 0.05 else
           "A2 real indistinguishable from random")
    print(f"  OUTCOME: {out}\n")

    print("=" * 76)
    print("SWEEP: mean first-token P(YES), every cell")
    print("=" * 76)
    print(f"{'':>8}{'introspective':^30}{'neutral_matched':^30}")
    print(f"{'alpha':>8}{'real':>10}{'random':>10}{'shuffle':>10}{'real':>10}{'random':>10}{'shuffle':>10}")
    for a in alphas:
        row = f"{a:>8.0f}"
        for f in (I, N):
            for k in ("real", "random", "shuffle"):
                row += f"{np.mean(list(D[k][(f, a)].values())):>10.3f}"
        print(row)

    print("\n" + "=" * 76)
    print("Is the shift concept-specific?  real vs each control, per framing")
    print("=" * 76)
    print(f"{'framing':<17}{'alpha':>6}{'real':>8}{'random':>8}{'p':>9}{'shuffle':>9}{'p':>9}")
    for f in (I, N):
        for a in alphas[1:]:
            _, mr, p1, _, _ = test(D["real"][(f, a)], D["random"][(f, a)])
            mx, ms, p2, _, _ = test(D["real"][(f, a)], D["shuffle"][(f, a)])
            s1 = "*" if p1 < 0.05 else " "
            s2 = "*" if p2 < 0.05 else " "
            print(f"{f:<17}{a:>6.0f}{mx:>8.3f}{mr:>8.3f}{p1:>8.4f}{s1}{ms:>8.3f}{p2:>8.4f}{s2}")

    print("\n" + "=" * 76)
    print("SECONDARY (pre-registered): introspective vs neutral_matched, real vectors")
    print("=" * 76)
    print(f"{'alpha':>6}{'introspective':>15}{'neutral':>10}{'diff':>8}{'p':>9}{'intro higher':>14}")
    for a in alphas:
        mi, mn, p, hi, n = test(D["real"][(I, a)], D["real"][(N, a)])
        s = "*" if p < 0.05 else " "
        print(f"{a:>6.0f}{mi:>15.3f}{mn:>10.3f}{mi-mn:>+8.3f}{p:>8.4f}{s}{hi:>8}/{n}")

    print("\n" + "=" * 76)
    print("Rise over the no-injection baseline, by framing and vector type")
    print("=" * 76)
    print(f"{'framing':<17}{'vectors':<9}{'a=0':>7}{'a=2':>8}{'a=4':>8}{'a=6':>8}")
    for f in (I, N):
        for k in ("real", "random", "shuffle"):
            base = np.mean(list(D[k][(f, 0.0)].values()))
            row = f"{f:<17}{k:<9}{base:>7.3f}"
            for a in alphas[1:]:
                row += f"{np.mean(list(D[k][(f, a)].values())) - base:>+8.3f}"
            print(row)

    print("\n" + "=" * 76)
    print("Share of the real-vector effect that a content-free vector reproduces")
    print("=" * 76)
    for f in (I, N):
        base = np.mean(list(D["real"][(f, 0.0)].values()))
        parts = []
        for a in alphas[1:]:
            r = np.mean(list(D["real"][(f, a)].values())) - base
            c = np.mean([np.mean(list(D[k][(f, a)].values())) for k in ("random", "shuffle")]) - base
            parts.append(f"a={a:.0f}: {c / r:.0%}")
        print(f"  {f:<17}" + "   ".join(parts))


if __name__ == "__main__":
    main(*sys.argv[1:])
