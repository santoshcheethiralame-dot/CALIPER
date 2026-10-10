"""Paper 2, C4 (exploratory, first seen 9 Oct): live rate against each real vector's own KL.

    python experiments/analyse_p2_live_kl.py     # writes results/p2_live_vs_kl.json

Every real-recipe vector (concept token, template tail, sentence mean) at every non-zero dose of
its session's steer stage: steered there and not at dose 0 (the S-2 live rule, applied per dose),
against the next-token KL that dose produced. If a vector's own KL were a transferable dose unit,
the live-rate curves of different models would coincide. Controls (random, impact, shuffled,
span) are excluded. S-2 Gemma-3-4B on the fraction grid is excluded: it failed its manipulation
check.
"""
import glob
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SESSIONS = {"Qwen2.5-3B": "results/s2_qwen3b/s2_qwen3b_*_steer*.jsonl",
            "Qwen2.5-7B": "results/s2_qwen7b/s2_qwen7b_*_steer*.jsonl",
            "Gemma-3-4B (KL grid)": "results/s2_gemma4b_kl/s2_gemma4b_kl_*_steer*.jsonl",
            "Gemma-3-12B (4-bit)": "results/s1_gemma12/s1_gemma12_4bit_*_steer*.jsonl",
            "Gemma-3-27B (4-bit)": "results/s1_gemma27/s1_gemma27_4bit_*_steer*.jsonl",
            "Gemma-3-27B, released recipe (S-1M)": "results/s1_gemma27/s1_gemma27_s1m_steer_*.jsonl"}
REAL = ("_concept_steer", "_tail_steer", "_sentence_steer", "_s1m_steer")
EDGES = np.array([1e-5, 1e-3, 1e-2, 1e-1, 1, 10, 1e3])


def points(pattern):
    out = []
    for f in sorted(glob.glob(str(ROOT / pattern))):
        if not any(t in os.path.basename(f) for t in REAL):
            continue
        R = [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
        base = {r["concept"]: bool(r["steered"]) for r in R if r["alpha"] == 0}
        for r in R:
            if r["alpha"] and r["concept"] in base:
                out.append((float(r["kl"]), bool(r["steered"]) and not base[r["concept"]]))
    return out


def main():
    rep = {}
    for name, pat in SESSIONS.items():
        P = points(pat)
        if not P:
            print(name, "no files", flush=True)
            continue
        kl, live = np.array([p[0] for p in P]), np.array([p[1] for p in P])
        bins = []
        for lo, hi in zip(EDGES[:-1], EDGES[1:]):
            m = (kl >= lo) & (kl < hi)
            if m.sum() >= 10:  # smaller bins are too noisy to plot
                bins.append({"kl_lo": float(lo), "kl_hi": float(hi), "n": int(m.sum()),
                             "median_kl": float(np.median(kl[m])), "live_rate": float(live[m].mean())})
        rep[name] = {"points": len(P), "live": int(live.sum()), "bins": bins}
        print(name, len(P), [(round(b["median_kl"], 3), round(b["live_rate"], 2), b["n"]) for b in bins],
              flush=True)
    json.dump(rep, open(ROOT / "results/p2_live_vs_kl.json", "w"), indent=1)


if __name__ == "__main__":
    main()
