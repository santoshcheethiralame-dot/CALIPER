"""Round-2 roadmap A6: every completed run in one table, and a sensitivity pool with X-1a.

    python experiments/analyse_every_run.py     # writes results/every_run.json

Adds the runs the round-1 table left out (X-1a/b, B-17b/c/d, B-2d, T-SAE) to the round-1 arm list
and scores them the same way. The sensitivity pools replace the pooled GPT-Neo L10 arm with X-1a, or
add X-1a as a seventh arm; X-1a has the same settings on new units.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT))
import analyse_review_round1 as r1  # noqa: E402

NEW = {
    "GPT-Neo-125m L10, new units (X-1a)": ("results/x1a_gptneo_l10.jsonl", "corrected", "CPU", "fp32"),
    "GPT-Neo-125m L10, new units, corpus seed 1 (X-1b)": ("results/x1b_gptneo_l10.jsonl", "corrected",
                                                          "CPU", "fp32"),
    "GPT-Neo-125m L6, coordinate dropped (B-17b)": ("results/b17b_gptneo125m_l6_drop.jsonl", "corrected",
                                                    "CPU", "fp32"),
    "GPT-2 L3 (B-17c)": ("results/b17c_gpt2_l3.jsonl", "corrected", "CPU", "fp32"),
    "GPT-2 L3, coordinate dropped (B-17d)": ("results/b17d_gpt2_l3_drop.jsonl", "corrected", "CPU", "fp32"),
    "Pythia-160m L6, float32 (B-2d)": ("results/b2d_pythia160m_fp32.jsonl", "corrected", "CPU", "fp32"),
    "GPT-2 L6, SAE latents (T-SAE)": ("results/tsae_main.jsonl", "corrected", "CPU", "fp32"),
}


def pool(arms, table):
    d = [table[a]["diff"] for a in arms]
    se = [table[a]["delong_se"] for a in arms]
    return {"arms": arms, **r1.hksj(d, se)}


def main():
    r1.ARMS.update(NEW)
    table = r1.r3_all_arms()
    matched = r1.r1_route_matched()
    for name in table:
        table[name]["route_matched"] = matched[name]
    swap = [a if a != "GPT-Neo-125m L10 (B-8b)" else "GPT-Neo-125m L10, new units (X-1a)" for a in r1.POOLED]
    rep = {"arms": table,
           "pool_filed": pool(r1.POOLED, table),
           "pool_x1a_for_b8b": pool(swap, table),
           "pool_x1a_added": pool(r1.POOLED + ["GPT-Neo-125m L10, new units (X-1a)"], table)}
    for name, d in table.items():
        if name in NEW:
            print(name, d["failures"], d.get("auc_restart"), d.get("auc_r2"), d.get("diff"),
                  d.get("boot_ci95"), flush=True)
    for k in ("pool_filed", "pool_x1a_for_b8b", "pool_x1a_added"):
        print(k, rep[k]["pooled"], rep[k]["hksj_ci95"], flush=True)
    json.dump(rep, open(ROOT / "results/every_run.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
