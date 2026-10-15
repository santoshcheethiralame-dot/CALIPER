"""E-1 unit draw (docs/preregistration-e1-fair-comparison.md), run once before any E-1 fit.

    python experiments/e1_units.py      # writes results/e1_units.json

200 GPT-2 layer-6 and 100 GPT-Neo-125m layer-10 MLP units never fitted in any earlier run, drawn
as D-1 drew its units: from the 3,072 units minus every unit id in any results/**/*.jsonl or
data/**/*.jsonl row (rows do not record their model, so ids from other models are excluded too,
which only shrinks the pool). GPT-2 with default_rng(20261015), then GPT-Neo with
default_rng(20261016) from the same pool minus the GPT-2 draw, so no id is shared.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
D_MLP = 3072


def main():
    out = ROOT / "results/e1_units.json"
    if out.exists():
        raise SystemExit(f"{out} exists; the draw is made once")
    seen = set()
    for f in sorted(list((ROOT / "results").rglob("*.jsonl")) + list((ROOT / "data").rglob("*.jsonl"))):
        for line in open(f, encoding="utf-8"):
            if line.strip():
                k = str(json.loads(line).get("_key"))
                if k.isdigit():
                    seen.add(int(k))
    pool = np.array(sorted(set(range(D_MLP)) - seen))
    gpt2 = np.random.default_rng(20261015).choice(pool, size=200, replace=False)
    rest = np.array(sorted(set(pool.tolist()) - set(gpt2.tolist())))
    neo = np.random.default_rng(20261016).choice(rest, size=100, replace=False)
    json.dump({"gpt2_l6": sorted(int(u) for u in gpt2), "gptneo_l10": sorted(int(u) for u in neo),
               "excluded": len(seen), "pool": len(pool)}, open(out, "w"), indent=1)
    print(f"pool {len(pool)} (excluded {len(seen)}); drew 200 GPT-2 and 100 GPT-Neo units")


if __name__ == "__main__":
    main()
