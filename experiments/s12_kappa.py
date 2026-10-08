"""S-12: agreement between two labelling passes, and between the labels and a scorer.

    python experiments/s12_kappa.py docs/s12/s12_pass1.csv docs/s12/s12_pass2.csv \
        --key docs/s12/s12_key.csv [--judge docs/s12/s12_judge.csv]

For each label it reports, on the items both files share:
- Cohen's kappa between the two files (pass 1 and the blind pass-2 re-label), with a
  bootstrap 95% CI over items;
- the same for each file against the rule scorer (from the key file);
- the same against a judge, when one is given.

The gate (run plan, S-12) is pass-1 vs pass-2 kappa >= 0.6 on the label a judge would replace.
Below it, judge numbers drop to secondary, and rule-scored and first-token readouts carry the
claims.

Labels compared:
- detect: said (YES/NO/UNCLEAR), with UNCLEAR kept as its own category;
- steer: about_concept (Y/N/UNSURE);
- both tasks: coherent (Y/N).
"""
import argparse
import csv

import numpy as np


def kappa(a, b):
    cats = sorted(set(a) | set(b))
    idx = {c: i for i, c in enumerate(cats)}
    m = np.zeros((len(cats), len(cats)))
    for x, y in zip(a, b):
        m[idx[x], idx[y]] += 1
    n = m.sum()
    po = np.trace(m) / n
    pe = (m.sum(0) * m.sum(1)).sum() / n**2
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def boot(a, b, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a), np.asarray(b)
    ks = []
    for _ in range(n_boot):
        i = rng.integers(0, len(a), len(a))
        ks.append(kappa(list(a[i]), list(b[i])))
    return [round(float(np.percentile(ks, 2.5)), 3), round(float(np.percentile(ks, 97.5)), 3)]


def load(path):
    return {r["item"]: r for r in csv.DictReader(open(path, encoding="utf-8"))}


def norm(v):
    v = (v or "").strip().upper()
    return {"YES": "Y", "NO": "N", "TRUE": "Y", "FALSE": "N", "1": "Y", "0": "N"}.get(v, v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--key", required=True)
    ap.add_argument("--judge", default=None)
    args = ap.parse_args()
    A, B, K = load(args.a), load(args.b), load(args.key)
    J = load(args.judge) if args.judge else None
    labels = {"detect": ("said", "coherent"), "steer": ("about_concept", "coherent")}
    rule_field = {"said": "rule", "about_concept": "rule"}
    for task, fields in labels.items():
        items = [i for i in K if K[i]["task"] == task and i in A and i in B]
        print(f"\n{task}: {len(items)} items")
        for f in fields:
            a = [norm(A[i][f]) for i in items]
            b = [norm(B[i][f]) for i in items]
            if not any(a) or not any(b):
                print(f"  {f}: not labelled")
                continue
            k = kappa(a, b)
            print(f"  {f:<14} human-human kappa {k:.3f} {boot(a, b)}"
                  f"{'   GATE PASSED' if f != 'coherent' and k >= 0.6 else ''}")
            if f in rule_field:
                rule = ["Y" if K[i][rule_field[f]] == "1" else "N" for i in items]
                for name, lab in (("A", a), ("B", b)):
                    # UNCLEAR / UNSURE count as "not affirmed", the rule's reading
                    yn = ["Y" if x == "Y" else "N" for x in lab]
                    print(f"  {'':<14} {name}-rule kappa {kappa(yn, rule):.3f}")
            if J is not None:
                j = [norm(J[i][f]) for i in items if i in J]
                if len(j) == len(items):
                    print(f"  {'':<14} judge-A {kappa(j, a):.3f}, judge-B {kappa(j, b):.3f}")


if __name__ == "__main__":
    main()
