"""P2-F: factual-NO P(YES) change against the introspective P(YES) change at the same arm and dose.

    python experiments/analyse_p2f.py      # writes results/p2f_analysis.json

Re-derives the numbers scored on 9-10 Oct (docs/LAB_NOTEBOOK.md, P2-F entries) into one file for
the Paper 2 figures. Changes are paired by concept against the same arm's strength 0. Qwen rows
store raw strengths; their fraction of the residual norm is the rank among the non-zero
strengths (0.25, 0.5, 1.0), as run.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SESSIONS = {  # model: [(arm, factual-NO file, introspective file, dose labels)]
    "Gemma-3-27B (released recipe)": [
        ("released", "results/p2f_gemma27/p2f_gemma27_forced_macar-release_factualno_fromtrial.jsonl",
         "results/s1_gemma27/s1_gemma27_s1m_forced_macar-release_fromtrial.jsonl", ["strength 4", "strength 8"])],
    "Qwen2.5-7B": [
        (arm, f"results/p2f_qwen7b/p2f_qwen7b_{arm}_forced{tag}_factualno.jsonl",
         f"results/s2_qwen7b/s2_qwen7b_{arm}_forced{tag}.jsonl", ["0.25", "0.5", "1.0"])
        for arm, tag in (("concept", ""), ("tail", ""), ("random", "_random"))],
}


def by_dose(path, framing):
    R = [json.loads(l) for l in open(ROOT / path, encoding="utf-8") if l.strip()]
    R = [r for r in R if r["framing"] == framing and r.get("control", "none") in ("none", "random")]
    out = {}
    for r in R:
        out.setdefault(r["alpha"], {})[r["concept"]] = (r["p_yes"], r.get("kl", 0.0))
    return out


def change(d, a):
    c = sorted(set(d[0.0]) & set(d[a]))
    delta = np.array([d[a][k][0] - d[0.0][k][0] for k in c])
    return float(delta.mean()), float(np.median([d[a][k][1] for k in c])), len(c)


def main():
    rep = {}
    for model, arms in SESSIONS.items():
        for arm, fno, intro, labels in arms:
            F, I = by_dose(fno, "factual_no"), by_dose(intro, "introspective")
            fa, ia = sorted(a for a in F if a), sorted(a for a in I if a)
            rows = []
            for lab, a_f, a_i in zip(labels[-len(fa):], fa, ia[-len(fa):]):
                f_mean, f_kl, n = change(F, a_f)
                i_mean, i_kl, _ = change(I, a_i)
                rows.append({"dose": lab, "factual_no_change": f_mean, "introspective_change": i_mean,
                             "ratio": f_mean / i_mean if i_mean else None, "median_kl": f_kl,
                             "concepts": n})
            rep[f"{model} / {arm}"] = rows
            for r in rows:
                print(f"{model:30s} {arm:9s} {r['dose']:10s} factual-NO {r['factual_no_change']:+.4f} "
                      f"introspective {r['introspective_change']:+.3f} ratio {r['ratio']:.3f} "
                      f"KL {r['median_kl']:.4g}", flush=True)
    json.dump(rep, open(ROOT / "results/p2f_analysis.json", "w"), indent=1)


if __name__ == "__main__":
    main()
