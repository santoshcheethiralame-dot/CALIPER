"""F-1b unit selection (docs/preregistration-f1b-identifiable.md), run once before any F-1b fit.

    python experiments/f1b_units.py      # writes results/f1b_units.json and results/f1b_units_<group>.txt

Population A: every unit whose archived fit is converged-wrong on the identifiable label (1/gamma
removed, held-out R2 > 0.99), in the GPT-2 and GPT-Neo runs with saved directions. Population B:
30 units drawn with numpy default_rng(0) from those runs' identifiable under-fitted units.
Population C: 10 units drawn the same way from B-8b's passing units (harm). A unit found in
several runs takes its settings from the first run in SOURCES order. Pythia is left out: its arm
was loaded in float16 at the time.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCES = [  # a4_spectrum arm, group, e01_gate settings beyond the defaults (8,000 tokens, 1,600 steps)
    ("GPT-Neo-125m L10 (B-8b)", "neo10_c0", "--model EleutherAI/gpt-neo-125M --layer 10 --restarts 2"),
    ("GPT-Neo-125m L10, new units (X-1a)", "neo10_c0", "--model EleutherAI/gpt-neo-125M --layer 10 --restarts 2"),
    ("GPT-Neo-125m L10, new units (X-1b)", "neo10_c1",
     "--model EleutherAI/gpt-neo-125M --layer 10 --restarts 2 --corpus-seed 1"),
    ("GPT-2 L6 re-fit, fit seed 1 (B-15a)", "gpt2_b15a", "--layer 6 --restarts 5 --fit-seed 1"),
    ("GPT-2 L6 re-fit, corpus seed 1 (B-15b)", "gpt2_b15b", "--layer 6 --restarts 2 --corpus-seed 1"),
    ("GPT-2 L6 re-fit, sequence split (B-15c)", "gpt2_b15c", "--layer 6 --restarts 2 --sequence-split"),
    ("GPT-Neo-125m L6 (B-17)", "neo6", "--model EleutherAI/gpt-neo-125M --layer 6 --restarts 2"),
]
MODEL_OF = {"neo10": "neo-L10", "gpt2": "gpt2-L6", "neo6": "neo-L6"}


def main():
    spec = json.load(open(ROOT / "results/a4_spectrum.json"))
    assigned, pools = {}, {"A": [], "B": [], "C": []}
    for arm, group, _ in SOURCES:
        sub = MODEL_OF[group.split("_")[0]]
        for u in spec[arm]["units"]:
            key = (sub, u["unit"])
            wrong = u["ident"] < 0.95
            pop = "A" if wrong and u["r2"] > 0.99 else "B" if wrong else ("C" if "B-8b" in arm else None)
            if pop and key not in assigned:
                assigned[key] = group
                pools[pop].append((group, u["unit"]))
    rng = np.random.default_rng(0)
    chosen = {"A": sorted(pools["A"]),
              "B": sorted(map(tuple, np.array(pools["B"], dtype=object)[rng.choice(len(pools["B"]), 30, replace=False)])),
              "C": sorted(map(tuple, np.array(pools["C"], dtype=object)[rng.choice(len(pools["C"]), 10, replace=False)]))}
    groups = {}
    for pop, items in chosen.items():
        for g, u in items:
            groups.setdefault(g, {"flags": next(f for _, gg, f in SOURCES if gg == g), "units": {}})
            groups[g]["units"][int(u)] = pop
    for g, d in groups.items():
        (ROOT / f"results/f1b_units_{g}.txt").write_text(",".join(str(u) for u in sorted(d["units"])))
    out = {"populations": {p: len(v) for p, v in chosen.items()}, "pool_sizes": {p: len(v) for p, v in pools.items()},
           "groups": groups}
    json.dump(out, open(ROOT / "results/f1b_units.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "groups"}),
          {g: (len(d["units"]), sum(p == "A" for p in d["units"].values())) for g, d in groups.items()})


if __name__ == "__main__":
    main()
