"""S-2: do the standard vector health checks detect dead vectors?
(docs/preregistration-s2-instrument-audit.md, with Amendment 1)

    python experiments/analyse_s2.py --dir <unzipped session> --model qwen3b
    python experiments/analyse_s2.py --dir ... --model qwen3b --out results/s2_qwen3b_analysis.json

Reference label, as filed: a real-arm vector is LIVE if its steer-stage trial is steered at
the gate dose and not at alpha 0; otherwise DEAD. The gate is the second non-zero alpha of the
grid: alpha-frac 0.5 on the original grid, the 0.5-nat KL target under Amendment 3. The three real arms (concept token,
template tail, sentence mean) are pooled per model (up to 90 vectors). A model with fewer
than 5 live or 5 dead vectors is reported but not scored.

Primary: per health statistic, AUC at telling live from dead, with a stratified bootstrap
95% CI. Orientation is fixed in advance so that a higher value looks healthier:
  norm                       raw vector norm
  distinctness               minus max |cos| to another vector of the same arm
  stability                  split-half or second-template agreement
  probe                      APERTURE's held-out probe accuracy
  logit steering             APERTURE's logit check (delta log P(concept))
  P(YES) shift               paired t of P(YES) at the gate dose vs 0 across the two
                             forced-stage framings (df = 1), per vector
AUC = P(statistic of a live vector > statistic of a dead vector), ties counted half.
Prediction: norm, distinctness and P(YES) shift near 0.5 (CI includes 0.5); logit steering
>= 0.8. Stability and probe have no stated prediction.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ARMS = {"concept token": ("concept_steer{q}", "concept_forced{q}"),
        "template tail": ("tail_steer{q}", "tail_forced{q}"),
        "sentence mean": ("sentence_steer_aperture{q}", "sentence_forced_aperture{q}")}
GATE = 2
STATS = ["norm", "distinctness", "stability", "probe", "logit steering", "P(YES) shift"]


def auc(score, live):
    score, live = np.asarray(score, float), np.asarray(live, bool)
    m, n = live.sum(), (~live).sum()
    r = rankdata(score)
    return float((r[live].sum() - m * (m + 1) / 2) / (m * n))


def boot_ci(score, live, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    score, live = np.asarray(score, float), np.asarray(live, bool)
    li, di = np.where(live)[0], np.where(~live)[0]
    out = [auc(score[i], live[i]) for i in
           (np.concatenate([rng.choice(li, len(li)), rng.choice(di, len(di))]) for _ in range(n_boot))]
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def rows(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def vectors(d, model, quant="none"):
    recs = []
    q = "" if quant == "4bit" else f"_{quant}"     # the script leaves 4bit out of file names
    for arm, (steer, forced) in ARMS.items():
        steer, forced = steer.format(q=q), forced.format(q=q)
        cfg = json.load(open(d / f"s2_{model}_{steer}.config.json"))
        pos = {al: i for i, al in enumerate(sorted(cfg["alphas"]))}
        st = rows(d / f"s2_{model}_{steer}.jsonl")
        fo = rows(d / f"s2_{model}_{forced}.jsonl")
        for c in cfg["concepts"]:
            s = {pos[r["alpha"]]: r["steered"] for r in st if r["concept"] == c}
            live = bool(s.get(GATE)) and not bool(s.get(0))
            diffs = []
            for fr in sorted({r["framing"] for r in fo}):
                p = {pos[r["alpha"]]: r["p_yes"] for r in fo if r["concept"] == c and r["framing"] == fr}
                diffs.append(p[GATE] - p[0])
            diffs = np.array(diffs)
            sd = diffs.std(ddof=1)
            t = diffs.mean() / (sd / np.sqrt(len(diffs))) if sd > 0 else np.sign(diffs.mean()) * np.inf
            h = cfg["health"][c]
            recs.append({"arm": arm, "concept": c, "live": live,
                         "norm": h["norm"], "distinctness": -h["max_cos_other"],
                         "stability": h["stability"], "probe": h["probe"],
                         "logit steering": h["steer_logit_delta"], "P(YES) shift": float(t)})
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--quant", default="none", choices=("none", "4bit", "8bit"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    recs = vectors(Path(a.dir), a.model, a.quant)
    live = np.array([r["live"] for r in recs])
    rep = {"model": a.model, "vectors": len(recs), "live": int(live.sum()), "dead": int((~live).sum()),
           "live_by_arm": {arm: [sum(r["live"] for r in recs if r["arm"] == arm),
                                 sum(1 for r in recs if r["arm"] == arm)] for arm in ARMS}}
    if live.sum() < 5 or (~live).sum() < 5:
        rep["scored"] = False
        rep["note"] = "fewer than 5 live or 5 dead vectors: reported, not scored (as filed)"
    else:
        rep["scored"] = True
        rep["auc"] = {}
        for s in STATS:
            x = np.array([r[s] for r in recs], float)
            x = np.where(np.isinf(x), np.sign(x) * 1e9, x)
            ci = boot_ci(x, live)
            rep["auc"][s] = {"auc": auc(x, live), "ci95": ci}
        p = rep["auc"]
        rep["predictions"] = {
            "norm near 0.5": p["norm"]["ci95"][0] <= 0.5 <= p["norm"]["ci95"][1],
            "distinctness near 0.5": p["distinctness"]["ci95"][0] <= 0.5 <= p["distinctness"]["ci95"][1],
            "P(YES) shift near 0.5": p["P(YES) shift"]["ci95"][0] <= 0.5 <= p["P(YES) shift"]["ci95"][1],
            "logit steering >= 0.8": p["logit steering"]["auc"] >= 0.8}
    rep["per_vector"] = recs
    print(json.dumps({k: v for k, v in rep.items() if k != "per_vector"}, indent=1))
    if a.out:
        json.dump(rep, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
