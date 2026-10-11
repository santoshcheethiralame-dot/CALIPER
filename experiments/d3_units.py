"""D-3 unit draw (docs/preregistration-d3-token-scaling.md), run once before any D-3 fit.

    python experiments/d3_units.py      # writes results/d3_units.txt and results/d3_units.json

150 GPT-2 layer-6 MLP units never fitted before and not in D-1's draw: numpy default_rng(20261012)
over the layer's 3,072 units minus every unit id in any results/**/*.jsonl or data/**/*.jsonl row
(as d1_units.py) and minus results/d1_units.txt.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
N, D_MLP, SEED = 150, 3072, 20261012


def main():
    seen = set()
    for f in sorted(list((ROOT / "results").rglob("*.jsonl")) + list((ROOT / "data").rglob("*.jsonl"))):
        keys = [str(json.loads(l).get("_key")) for l in open(f, encoding="utf-8") if l.strip()]
        seen |= {int(k) for k in keys if k.isdigit()}
    d1 = {int(u) for u in (ROOT / "results/d1_units.txt").read_text().split(",")}
    pool = np.array(sorted(set(range(D_MLP)) - seen - d1))
    units = np.random.default_rng(SEED).choice(pool, size=N, replace=False)
    (ROOT / "results/d3_units.txt").write_text(",".join(str(int(u)) for u in units))
    json.dump({"seed": SEED, "units": [int(u) for u in units], "excluded_earlier": len(seen),
               "excluded_d1": len(d1), "pool": len(pool)}, open(ROOT / "results/d3_units.json", "w"), indent=1)
    print(f"excluded {len(seen)} earlier ids and {len(d1)} D-1 ids; pool {len(pool)}; drew {N}")


if __name__ == "__main__":
    main()
