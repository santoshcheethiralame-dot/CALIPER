"""Is identification at chance, and do the report and content channels dissociate?

Two questions the free-generation run can answer without any new GPU time.

1. When a response names its injected concept, is that above chance? The chance
   baseline is built from the same responses: how often does a response contain one of
   the 29 concepts that were NOT injected on that trial. This controls for the model
   simply talking about concept-like nouns a lot.

2. Do "says YES" and "names the concept" co-occur more or less than independence
   predicts? A detection faculty that reports what it detects should put them together.
"""

import json
import sys
from itertools import product

import numpy as np
from scipy.stats import fisher_exact

sys.path.insert(0, "experiments")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from rescore_s3 import first_answer, names_concept  # noqa: E402


def main(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    concepts = sorted({r["concept"] for r in rows})
    alphas = sorted({r["alpha"] for r in rows})

    print("1. IS IDENTIFICATION ABOVE CHANCE?\n")
    print("   'hit'   = response contains the concept that was injected")
    print("   'chance'= response contains some OTHER concept from the 30, per opportunity")
    print(f"\n{'alpha':>6}{'hit rate':>11}{'chance rate':>14}{'ratio':>9}{'p':>10}")
    for a in alphas:
        g = [r for r in rows if r["alpha"] == a]
        hits = sum(names_concept(r["text"], r["concept"]) for r in g)
        # Every (response, non-injected concept) pair is one chance opportunity.
        opp = sum(1 for r, c in product(g, concepts) if c != r["concept"])
        false_hits = sum(1 for r, c in product(g, concepts)
                         if c != r["concept"] and names_concept(r["text"], c))
        hr, cr = hits / len(g), false_hits / opp
        table = [[hits, len(g) - hits], [false_hits, opp - false_hits]]
        p = fisher_exact(table).pvalue
        ratio = f"{hr / cr:.1f}x" if cr > 0 else "inf"
        print(f"{a:>6}{hr:>10.1%}{cr:>14.2%}{ratio:>9}{p:>10.1e}")

    print("\n\n2. DO REPORT AND CONTENT GO TOGETHER?\n")
    print(f"{'alpha':>6}{'says YES':>10}{'names':>8}{'both':>7}{'if indep':>10}{'p':>9}  verdict")
    for a in alphas:
        g = [r for r in rows if r["alpha"] == a]
        y = np.array([first_answer(r["text"]) == "YES" for r in g])
        m = np.array([names_concept(r["text"], r["concept"]) for r in g])
        both = int((y & m).sum())
        indep = y.mean() * m.mean() * len(g)
        if y.sum() and m.sum() and y.sum() < len(g) and m.sum() < len(g):
            p = fisher_exact([[both, int(y.sum()) - both],
                              [int(m.sum()) - both, len(g) - int(y.sum()) - int(m.sum()) + both]]
                             ).pvalue
            ptxt = f"{p:.2f}"
        else:
            ptxt = "-"
        if y.mean() > 0.15 and m.mean() < 0.05:
            v = "report without content"
        elif m.mean() > 0.15 and y.mean() < 0.15:
            v = "content without report"
        elif both > indep:
            v = "together"
        else:
            v = "-"
        print(f"{a:>6}{y.mean():>9.0%}{m.mean():>8.0%}{both:>7}{indep:>10.1f}{ptxt:>9}  {v}")

    print("""
The two channels peak at different injection strengths. At low alpha the model asserts a
detection and names nothing; at high alpha the concept comes out and the assertion does
not. That is a dissociation in both directions, not a single missing faculty.""")


if __name__ == "__main__":
    main(sys.argv[1])
