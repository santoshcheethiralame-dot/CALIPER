"""Numbers Paper 2's prose quotes that no earlier scorer wrote to a results file.

    python experiments/analyse_p2_numbers.py      # writes results/p2_numbers.json

c4_s1_27b      S-1 Gemma-3-27B, KL-calibrated grid: per recipe, the random-calibrated KL over the
               real vectors' median KL at each grid point, the top dose as a fraction of the
               residual norm, and steered counts per dose
p2r_kl_ratio   P2-R, released recipe: median next-token KL of norm-matched and impact-matched
               random vectors over real vectors, per framing and strength
identify_rank  P2-R identify stage: concepts whose token ranks within the top k first tokens,
               real against random vectors, per strength (k not fixed in advance)
s3_readout     Study 3's generated text, rescored with rescore_s3's rules: the filed scorer, a
               first YES, and a first YES that names the injected concept, per dose
Descriptive throughout.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from rescore_s3 import first_answer, names_concept  # noqa: E402

R = ROOT / "results"


def rows(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def median(xs):
    return float(np.median(xs)) if xs else None


def c4_s1_27b():
    out = {}
    for arm, stem in (("concept", "concept_steer"), ("tail", "tail_steer"), ("sentence", "sentence_steer_aperture")):
        cfg = json.load(open(R / f"s1_gemma27/s1_gemma27_4bit_{stem}.config.json"))
        rs = rows(R / f"s1_gemma27/s1_gemma27_4bit_{stem}.jsonl")
        grid = []
        for g in cfg["kl_calibration"]:
            real = median([r["kl"] for r in rs if r["alpha"] == g["alpha"]])
            grid.append({"kl_target": g["kl_target"], "random_kl": g["kl_achieved"], "real_median_kl": real,
                         "random_over_real": g["kl_achieved"] / real,
                         "fraction_of_norm": g["alpha"] / cfg["residual_norm_at_read_median"]})
        out[arm] = {"grid": grid, "steered": {f"{a:g}": sum(r["steered"] for r in rs if r["alpha"] == a)
                                               for a in sorted({r["alpha"] for r in rs})}}
    ratios = [g["random_over_real"] for v in out.values() for g in v["grid"]]
    tops = [v["grid"][-1]["fraction_of_norm"] for v in out.values()]
    out["range"] = {"random_over_real": [min(ratios), max(ratios)], "top_fraction": [min(tops), max(tops)]}
    return out


def p2r_kl_ratio():
    d = R / "p2r_gemma27"
    real = rows(d / "p2r_gemma27_forced_macar-release_p2r_fromtrial.jsonl")
    arms = {"norm-matched": rows(d / "p2r_gemma27_forced_random_macar-release_p2r_fromtrial.jsonl"),
            "impact-matched": rows(d / "p2r_gemma27_forced_random-impact_macar-release_p2r_fromtrial.jsonl")}
    med = lambda rs, f, a: median([r["kl"] for r in rs if r["framing"] == f and r["alpha"] == a])
    out = {}
    for f in sorted({r["framing"] for r in real}):
        out[f] = {}
        for a in sorted({r["alpha"] for r in real} - {0}):
            m = med(real, f, a)
            out[f][f"{a:g}"] = {k: med(v, f, a) / m for k, v in arms.items()}
    fac = [v["impact-matched"] for f in ("factual_yes", "factual_no", "factual_contested") for v in out[f].values()]
    out["impact_over_real_factual_range"] = [min(fac), max(fac)]
    return out


def identify_rank():
    out = {}
    for arm, tag in (("real", "identify_macar-release_fromtrial"), ("random", "identify_random_macar-release_fromtrial")):
        rs = rows(R / f"p2r_gemma27/p2r_gemma27_{tag}.jsonl")
        out[arm] = {f"{a:g}": {str(k): sum(r["concept_rank"] < k for r in rs if r["alpha"] == a) for k in (1, 10, 100, 1000)}
                    for a in sorted({r["alpha"] for r in rs})}
    return out


def s3_readout():
    rs = rows(R / "s3_generation/s3_results.jsonl")
    out = {}
    for a in sorted({r["alpha"] for r in rs}):
        g = [r for r in rs if r["alpha"] == a]
        out[f"{a:g}"] = {"n": len(g), "filed_scorer": sum(r["detected"] for r in g),
                         "first_yes": sum(first_answer(r["text"]) == "YES" for r in g),
                         "yes_and_names": sum(first_answer(r["text"]) == "YES" and names_concept(r["text"], r["concept"])
                                              for r in g),
                         "names": sum(names_concept(r["text"], r["concept"]) for r in g)}
    return out


def main():
    rep = {"c4_s1_27b": c4_s1_27b(), "p2r_kl_ratio": p2r_kl_ratio(), "identify_rank": identify_rank(),
           "s3_readout": s3_readout()}
    json.dump(rep, open(R / "p2_numbers.json", "w"), indent=1)
    print(json.dumps({k: rep["c4_s1_27b"]["range"] if k == "c4_s1_27b" else rep[k] for k in rep if k != "p2r_kl_ratio"},
                     indent=1)[:2500])
    print("impact/real factual range", rep["p2r_kl_ratio"]["impact_over_real_factual_range"])


if __name__ == "__main__":
    main()
