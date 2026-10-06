"""S-4: Study 3 reanalysis. Equivalence tests, exact intervals, dose-response.

Governed by docs/preregistration-s4-reanalysis.md (margin +/-0.10 P(YES), paired TOST over
the 30 concepts, fixed before this script was run). No new data: every input is archived in
data/s3/.

    python experiments/analyse_s3_reanalysis.py --out results/s3_reanalysis.json
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
S3 = ROOT / "data" / "s3"
MARGIN = 0.10
RNG = np.random.default_rng(0)


def load(name):
    return [json.loads(l) for l in open(S3 / name, encoding="utf-8") if l.strip()]


def by_concept(rows, framing, alpha, tol=1e-3):
    """concept -> P(YES) at one alpha and framing (alpha matched within tol, relative)."""
    out = {}
    for r in rows:
        if r.get("framing") == framing and "p_yes" in r and \
                abs(r["alpha"] - alpha) <= tol * max(1.0, abs(alpha)):
            out[r["concept"]] = r["p_yes"]
    return out


def tost(a, b, margin=MARGIN):
    """Paired TOST on a - b. Equivalent if the 90% CI of the mean lies inside +/-margin."""
    keys = sorted(set(a) & set(b))
    d = np.array([a[k] - b[k] for k in keys])
    n = len(d)
    mean, se = d.mean(), d.std(ddof=1) / np.sqrt(n)
    t_lo = (mean + margin) / se          # H0: mean <= -margin
    t_hi = (mean - margin) / se          # H0: mean >= +margin
    p_lo = 1 - stats.t.cdf(t_lo, n - 1)
    p_hi = stats.t.cdf(t_hi, n - 1)
    ci90 = stats.t.interval(0.90, n - 1, loc=mean, scale=se)
    boots = [RNG.choice(d, n).mean() for _ in range(5000)]
    p_diff = stats.ttest_1samp(d, 0.0).pvalue
    equivalent = max(p_lo, p_hi) < 0.05
    if equivalent:
        verdict = "equivalent within +/-0.10"
    elif p_diff < 0.05:
        verdict = "different"
    else:
        verdict = "inconclusive"
    return {"n": n, "mean_diff": round(float(mean), 4),
            "ci90_t": [round(float(ci90[0]), 4), round(float(ci90[1]), 4)],
            "ci90_boot": [round(float(np.percentile(boots, 5)), 4),
                          round(float(np.percentile(boots, 95)), 4)],
            "p_tost": round(float(max(p_lo, p_hi)), 4), "p_difference": round(float(p_diff), 4),
            "verdict": verdict}


def cp(k, n, conf=0.95):
    a = (1 - conf) / 2
    lo = stats.beta.ppf(a, k, n - k + 1) if k > 0 else 0.0
    hi = stats.beta.ppf(1 - a, k + 1, n - k) if k < n else 1.0
    return [round(float(lo), 4), round(float(hi), 4)]


def dose(rows):
    cells = defaultdict(list)
    for r in rows:
        if "p_yes" in r:
            cells[(r["framing"], round(r["alpha"], 3))].append(r["p_yes"])
    out = {}
    for (fr, al), v in sorted(cells.items()):
        v = np.array(v)
        boots = [RNG.choice(v, len(v)).mean() for _ in range(2000)]
        out[f"{fr}|{al}"] = {"mean": round(float(v.mean()), 4), "n": len(v),
                             "ci95": [round(float(np.percentile(boots, 2.5)), 4),
                                      round(float(np.percentile(boots, 97.5)), 4)]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rep = {"margin": MARGIN, "equivalence": {}, "exact_ci": {}, "dose_response": {}}
    E = rep["equivalence"]

    real, rand, shuf = load("s3_forced_real.jsonl"), load("s3_forced_random.jsonl"), \
        load("s3_forced_shuffle.jsonl")
    E["C18 real vs random, alpha 6"] = tost(by_concept(real, "introspective", 6.0),
                                            by_concept(rand, "introspective", 6.0))
    for al in (2.0, 4.0, 6.0):
        E[f"C19 shuffle vs random, alpha {al:g}"] = tost(
            by_concept(shuf, "introspective", al), by_concept(rand, "introspective", al))
        E[f"C20 introspective vs neutral, real, alpha {al:g}"] = tost(
            by_concept(real, "introspective", al), by_concept(real, "neutral_matched", al))

    ur, ux = load("s3_unit_ext_forced_norm1.jsonl"), load("s3_unit_ext_forced_random_norm1.jsonl")
    E["C24 real vs random, alpha 32768"] = tost(by_concept(ur, "introspective", 32768.0),
                                                by_concept(ux, "introspective", 32768.0))

    gr, gx, gs = load("gw_forced_norm1.jsonl"), load("gw_forced_random_norm1.jsonl"), \
        load("gw_forced_span_norm1.jsonl")
    norm = 36244.97
    for frac in (0.30, 0.40, 0.50, 0.60):
        al = frac * norm
        E[f"C50-52 real vs random, frac {frac:.2f}"] = tost(
            by_concept(gr, "introspective", al), by_concept(gx, "introspective", al))
        E[f"C50-52 real vs span, frac {frac:.2f}"] = tost(
            by_concept(gr, "introspective", al), by_concept(gs, "introspective", al))

    rep["exact_ci"]["positive control, pre-registered scorer (C40)"] = {
        "k": 2, "n": 30, "rate": round(2 / 30, 4), "ci95": cp(2, 30),
        "macar_reported": 0.108}
    gen = load("s3_generation.jsonl")
    by_alpha = defaultdict(list)
    for r in gen:
        by_alpha[float(r["alpha"])].append(bool(r.get("detected")))
    rep["exact_ci"]["generation 'detected' field by alpha (C15/C16)"] = {
        f"{al:g}": {"k": int(sum(v)), "n": len(v), "ci95": cp(int(sum(v)), len(v))}
        for al, v in sorted(by_alpha.items())}

    for name in ("s3_forced_real.jsonl", "s3_forced_random.jsonl", "s3_forced_shuffle.jsonl",
                 "s3_unit_forced_norm1.jsonl", "s3_unit_ext_forced_norm1.jsonl",
                 "s3_unit_ext_forced_random_norm1.jsonl", "g2_forced_norm1.jsonl",
                 "g2_forced_random_norm1.jsonl", "g2_forced_span_norm1.jsonl",
                 "gw_forced_norm1.jsonl", "gw_forced_random_norm1.jsonl",
                 "gw_forced_span_norm1.jsonl"):
        rep["dose_response"][name] = dose(load(name))

    rep["vector_status"] = [
        {"runs": "C15-C24", "model": "Gemma-3-27B", "read_position": "template tail",
         "steering_control": "none run on these vectors"},
        {"runs": "C27-C31", "model": "Qwen2.5-32B", "read_position": "template tail",
         "steering_control": "C31: concept-in-text 1/30 at alpha 200, equal to alpha 0 (dead)"},
        {"runs": "C32-C34", "model": "Qwen2.5-32B", "read_position": "concept token",
         "steering_control": "C32: 1 -> 7 -> 14 of 30 (live)"},
        {"runs": "C45-C52", "model": "Gemma-3-27B", "read_position": "concept token",
         "steering_control": "C48: semantic 10/30 at 40% of norm (weak but live)"},
    ]

    print("EQUIVALENCE (paired TOST, margin +/-0.10 P(YES))")
    for k, v in E.items():
        print(f"  {k:<44} diff {v['mean_diff']:+.3f}  90% CI [{v['ci90_t'][0]:+.3f}, "
              f"{v['ci90_t'][1]:+.3f}]  p_tost {v['p_tost']:.3f}  p_diff {v['p_difference']:.3f}"
              f"  -> {v['verdict']}")
    pc = rep["exact_ci"]["positive control, pre-registered scorer (C40)"]
    print(f"POSITIVE CONTROL 2/30 = {pc['rate']:.3f}, 95% CI {pc['ci95']} (Macar 0.108)")
    if a.out:
        json.dump(rep, open(a.out, "w"), indent=2)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
