"""B1 (P2-R Amendment 1): what Gemma-3-27B says when YES and NO lose their first-token mass.

    python experiments/analyse_b1_tokens.py --dir <unzipped session>     # writes results/b1_tokens.json

Descriptive, filed before data. For each cell (released vectors, norm-matched random), framing and
strength: the most common top-1 first token and its share of rows; the median first-token mass of
the top-1 token; the share of continuations that open with YES or NO (word, any case), that name the
concept as a whole word, and that are degenerate (one token repeated over half the continuation).
Reproducibility check: logp_yes and logp_no against the P2-R rows (results/p2r_gemma27), median and
maximum absolute difference.
"""
import argparse
import collections
import glob
import json
import os
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CELLS = {"released": "_forced_macar-release_p2r_tok_fromtrial",
         "norm-matched random": "_forced_random_macar-release_p2r_tok_fromtrial"}
P2R = {"released": "_forced_macar-release_p2r_fromtrial",
       "norm-matched random": "_forced_random_macar-release_p2r_fromtrial"}
YN = re.compile(r"^\W*(yes|no)\b", re.I)


def load(d, tag):
    f = glob.glob(os.path.join(d, "**", f"*{tag}.jsonl"), recursive=True)
    return [json.loads(l) for l in open(f[0], encoding="utf-8") if l.strip()] if f else []


def degenerate(text):
    w = text.split()
    return bool(w) and collections.Counter(w).most_common(1)[0][1] > len(w) / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    a = ap.parse_args()
    rep = {}
    for cell, tag in CELLS.items():
        R = load(a.dir, tag)
        if not R:
            rep[cell] = "not run"
            continue
        old = {r["key"]: r for r in load(str(ROOT / "results/p2r_gemma27"), P2R[cell])}
        dy = [abs(r["logp_yes"] - old[r["key"]]["logp_yes"]) for r in R if r["key"] in old]
        dn = [abs(r["logp_no"] - old[r["key"]]["logp_no"]) for r in R if r["key"] in old]
        out = {"rows": len(R), "matched_p2r_rows": len(dy),
               "repro_abs_diff_logp": {"median": float(np.median(dy + dn)) if dy else None,
                                       "max": float(np.max(dy + dn)) if dy else None}}
        by = collections.defaultdict(list)
        for r in R:
            by[(r["framing"], r["alpha"])].append(r)
        for (fr, s), g in sorted(by.items()):
            top1 = collections.Counter(r["top_tokens"][0] for r in g)
            tok, n = top1.most_common(1)[0]
            cont = [r.get("continuation", "") for r in g]
            out[f"{fr} @ {s:g}"] = {
                "n": len(g), "top1_most_common": tok, "top1_share": n / len(g),
                "top1_distinct": len(top1), "median_top1_prob": float(np.median([np.exp(r["top_logp"][0]) for r in g])),
                "median_yesno_mass": float(np.median([r["yesno_mass"] for r in g])),
                "continuation_opens_yes_no": float(np.mean([bool(YN.match(c)) for c in cont])),
                "continuation_names_concept": float(np.mean([bool(re.search(rf"\b{re.escape(r['concept'])}", c, re.I))
                                                             for r, c in zip(g, cont)])),
                "continuation_degenerate": float(np.mean([degenerate(c) for c in cont]))}
        rep[cell] = out
    json.dump(rep, open(ROOT / "results/b1_tokens.json", "w"), indent=1)
    for cell, out in rep.items():
        if isinstance(out, str):
            print(cell, out)
            continue
        print(cell, out["rows"], "rows; repro", out["repro_abs_diff_logp"])
        for k, v in out.items():
            if "@" in k and k.startswith("released_introspective"):
                print(" ", k, v["top1_most_common"], round(v["top1_share"], 2), "YN-open", round(v["continuation_opens_yes_no"], 2),
                      "names", round(v["continuation_names_concept"], 2))


if __name__ == "__main__":
    main()
