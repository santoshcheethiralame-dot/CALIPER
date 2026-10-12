"""G-0 unit draw (docs/preregistration-g0-gain-law.md), run once before any G-0 fit.

    python experiments/g0_units.py      # writes results/g0_units.txt and results/g0_units.json

50 GPT-2 layer-0 MLP units: numpy default_rng(20261013) over the 3,072 units minus every unit id in
any earlier result file and minus D-1's and D-3's draws (no GPT-2 layer-0 unit has been fitted
before; the exclusion only shrinks the pool, as in d1_units.py).
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
N, D_MLP, SEED = 50, 3072, 20261013


def main():
    seen = set()
    for f in sorted(list((ROOT / "results").rglob("*.jsonl")) + list((ROOT / "data").rglob("*.jsonl"))):
        keys = [str(json.loads(l).get("_key")) for l in open(f, encoding="utf-8") if l.strip()]
        seen |= {int(k) for k in keys if k.isdigit()}
    for f in ("d1_units.txt", "d3_units.txt"):
        seen |= {int(u) for u in (ROOT / "results" / f).read_text().split(",")}
    pool = np.array(sorted(set(range(D_MLP)) - seen))
    units = np.random.default_rng(SEED).choice(pool, size=N, replace=False)
    (ROOT / "results/g0_units.txt").write_text(",".join(str(int(u)) for u in units))
    json.dump({"seed": SEED, "units": [int(u) for u in units], "excluded": len(seen), "pool": len(pool)},
              open(ROOT / "results/g0_units.json", "w"), indent=1)
    print(f"excluded {len(seen)}; pool {len(pool)}; drew {N}")


if __name__ == "__main__":
    main()
