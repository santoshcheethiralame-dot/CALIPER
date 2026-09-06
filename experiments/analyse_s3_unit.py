"""Unit-norm protocol (C21): onset search per the filed criterion.

Pre-registration: docs/preregistration-s3-forced-choice.md, Addendum 2,
filed 3 September 2026, before the run.

Onset criterion: the smallest alpha at which mean introspective P(YES) with
real vectors exceeds 0.10.
"""
import json
from pathlib import Path
from scipy.stats import wilcoxon

SRC = Path("data/s3/s3_unit_forced_norm1.jsonl")
OUT = Path("results/s3_unit_onset.json")
ONSET = 0.10
UNNORM_MEDIAN_NORM = 5002.36   # measured, C17-C20 vector-building line


def cell(rows, framing, alpha):
    return [r["p_yes"] for r in sorted(
        (r for r in rows if r["framing"] == framing and r["alpha"] == alpha),
        key=lambda r: r["concept"])]


def main():
    rows = [json.loads(l) for l in SRC.open(encoding="utf-8") if l.strip()]
    alphas = sorted({r["alpha"] for r in rows})
    out = {"n_rows": len(rows), "alphas": alphas, "onset_threshold": ONSET, "cells": []}

    for a in alphas:
        row = {"alpha": a}
        for fr in ("introspective", "neutral_matched"):
            g = cell(rows, fr, a)
            row[fr] = {"mean": sum(g) / len(g),
                       "frac_gt_half": sum(x > 0.5 for x in g) / len(g), "n": len(g)}
        out["cells"].append(row)

    intro = {c["alpha"]: c["introspective"]["mean"] for c in out["cells"]}
    reached = [a for a in alphas if intro[a] > ONSET]
    out["alpha_star"] = min(reached) if reached else None
    out["onset_reached"] = bool(reached)

    # paired test at the top of the sweep against alpha=0
    top = max(alphas)
    base, hi = cell(rows, "introspective", 0.0), cell(rows, "introspective", top)
    if any(x != y for x, y in zip(base, hi)):
        stat, p = wilcoxon(hi, base, alternative="greater")
        out["top_vs_zero"] = {"alpha": top, "W": float(stat), "p_one_sided": float(p),
                              "n_increased": sum(h > b for h, b in zip(hi, base))}

    # scale-equivalence check (notebook 7.1 A-1)
    out["scale_check"] = {
        "unnormalised_median_vector_norm": UNNORM_MEDIAN_NORM,
        "alpha_unit_per_alpha_unnorm": UNNORM_MEDIAN_NORM,
        "sweep_top_alpha_unit": top,
        "sweep_top_as_alpha_unnorm": top / UNNORM_MEDIAN_NORM,
        "alpha_unit_needed_to_match_unnorm_2": 2 * UNNORM_MEDIAN_NORM,
        "shortfall_factor": (2 * UNNORM_MEDIAN_NORM) / top,
    }
    OUT.write_text(json.dumps(out, indent=2))

    print(f"alpha* = {out['alpha_star']}   (threshold {ONSET})")
    if "top_vs_zero" in out:
        t = out["top_vs_zero"]
        print(f"alpha={t['alpha']:.0f} vs 0: W={t['W']:.1f} p={t['p_one_sided']:.3g} "
              f"rose on {t['n_increased']}/30")
    s = out["scale_check"]
    print(f"\nsweep top alpha={top:.0f} (unit) == alpha={s['sweep_top_as_alpha_unnorm']:.3f} unnormalised")
    print(f"to match unnormalised alpha=2 needs unit alpha={s['alpha_unit_needed_to_match_unnorm_2']:.0f}"
          f"  -> sweep fell {s['shortfall_factor']:.1f}x short")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
