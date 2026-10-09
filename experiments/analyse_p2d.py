"""P2-D: does a steering dose transfer across models? (docs/flagship-plan.md §6)

    python experiments/analyse_p2d.py            # writes results/p2d_dose_transfer.json

For each S-2 session and arm, at each grid position (0 = no injection, then three doses):
  median next-token KL on the framing prompts (introspective and neutral-matched),
  coherent and steered shares on the steer stage,
the dose itself as a fraction of the residual norm, and the run's residual and concept-vector
norms. Two contrasts:
  transfer   the same fraction of the residual norm on Qwen-3B, Qwen-7B and Gemma-4B
  real vs random   on the KL-calibrated grid, real vectors' KL against random directions' KL
                   at the same dose (the calibration equalises random directions only)
Descriptive; nothing here is pre-registered.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SESSIONS = {  # name: (dir, file prefix, quant suffix, grid)
    "Qwen2.5-3B": ("results/s2_qwen3b", "s2_qwen3b", "_none", "alpha-frac"),
    "Qwen2.5-7B": ("results/s2_qwen7b", "s2_qwen7b", "", "alpha-frac"),
    "Gemma-3-4B": ("results/s2_gemma4b", "s2_gemma4b", "_none", "alpha-frac"),
    "Gemma-3-4B (KL grid)": ("results/s2_gemma4b_kl", "s2_gemma4b_kl", "_none", "kl"),
}
ARMS = {  # arm: (framing file tag, steer file tag)
    "concept": ("concept", "concept_steer"),
    "tail": ("tail", "tail_steer"),
    "sentence": ("sentence_aperture", "sentence_steer_aperture"),
    "random": ("random_random", "random_steer_random"),
    "shuffle": ("shuffle_shuffle", "shuffle_steer_shuffle"),
    "span": ("span_span", "span_steer_span"),
    "impact-matched": ("impact_random-impact", "impact_steer_random-impact"),
}


def rows(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def session(d, pre, q):
    out, meta = {}, {}
    for arm, (fr, st) in ARMS.items():
        f = ROOT / d / f"{pre}_{fr}{q}.jsonl"
        s = ROOT / d / f"{pre}_{st}{q}.jsonl"
        cfg = json.load(open(ROOT / d / f"{pre}_{fr}{q}.config.json"))
        al = sorted(cfg["alphas"])
        rn = cfg["residual_norm_at_read_median"]
        r_fr, r_st = rows(f), rows(s)
        per = []
        for i, a in enumerate(al):
            kl = [r["kl"] for r in r_fr if r["alpha"] == a and r.get("kl") is not None]
            st = [r for r in r_st if r["alpha"] == a]
            per.append({"position": i, "alpha": a, "fraction_of_residual_norm": a / rn,
                        "median_kl": float(np.median(kl)) if kl else None,
                        "coherent": float(np.mean([bool(r.get("coherent")) for r in st])) if st else None,
                        "steered": float(np.mean([bool(r.get("steered")) for r in st])) if st else None})
        out[arm] = per
        if arm == "concept":
            meta = {"residual_norm": rn, "concept_vector_norm": cfg.get("vector_norm_median"),
                    "vector_over_residual": (cfg.get("vector_norm_median") or np.nan) / rn,
                    "kl_targets": cfg.get("kl_targets")}
    return {"meta": meta, "arms": out}


def main():
    rep = {name: session(d, pre, q) for name, (d, pre, q, _) in SESSIONS.items()}
    # Transfer: concept-arm KL and coherence at the same fractions of the residual norm.
    rep["transfer_at_same_fraction"] = {
        name: [{"fraction": round(p["fraction_of_residual_norm"], 3), "median_kl": p["median_kl"],
                "coherent": p["coherent"]} for p in rep[name]["arms"]["concept"]]
        for name, (_, _, _, grid) in SESSIONS.items() if grid == "alpha-frac"}
    # Real vs random on the KL grid: the ratio at each dose.
    kl = rep["Gemma-3-4B (KL grid)"]["arms"]
    rep["real_vs_random_on_kl_grid"] = [
        {"position": i, "target_nats": t,
         **{arm: kl[arm][i]["median_kl"] for arm in ARMS},
         "concept_over_random": kl["concept"][i]["median_kl"] / kl["random"][i]["median_kl"]}
        for i, t in zip((1, 2, 3), rep["Gemma-3-4B (KL grid)"]["meta"]["kl_targets"] or [None] * 3)]
    print(json.dumps({k: v for k, v in rep.items()
                      if k in ("transfer_at_same_fraction", "real_vs_random_on_kl_grid")}, indent=1))
    print({n: {k: round(v, 3) if isinstance(v, float) else v for k, v in rep[n]["meta"].items()}
           for n in SESSIONS})
    json.dump(rep, open(ROOT / "results/p2d_dose_transfer.json", "w"), indent=1)


if __name__ == "__main__":
    main()
