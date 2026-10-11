"""D-1 unit draw (docs/preregistration-d1-fresh-units.md), run once before any D-1 fit.

    python experiments/d1_units.py      # writes results/d1_units.txt and results/d1_units.json

200 GPT-2 layer-6 MLP units never fitted in any earlier run: drawn with numpy default_rng(20261011)
from the layer's 3,072 units minus every unit id that appears in any results/**/*.jsonl or
data/**/*.jsonl row. Rows do not record their model, so ids from other models and layers are
excluded too; that only shrinks the pool.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
N, D_MLP, SEED = 200, 3072, 20261011


def main():
    seen, files = set(), []
    for f in sorted(list((ROOT / "results").rglob("*.jsonl")) + list((ROOT / "data").rglob("*.jsonl"))):
        keys = [str(json.loads(l).get("_key")) for l in open(f, encoding="utf-8") if l.strip()]
        ids = {int(k) for k in keys if k.isdigit()}  # other files key rows by config, not unit
        if ids:
            seen |= ids
            files.append(f.relative_to(ROOT).as_posix())
    pool = np.array(sorted(set(range(D_MLP)) - seen))
    units = np.random.default_rng(SEED).choice(pool, size=N, replace=False)
    (ROOT / "results/d1_units.txt").write_text(",".join(str(int(u)) for u in units))
    json.dump({"seed": SEED, "units": [int(u) for u in units], "excluded": len(seen),
               "pool": len(pool), "files_scanned": files},
              open(ROOT / "results/d1_units.json", "w"), indent=1)
    print(f"excluded {len(seen)} ids from {len(files)} files; pool {len(pool)}; drew {N}")


if __name__ == "__main__":
    main()
