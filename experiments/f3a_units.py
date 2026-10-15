"""F-3a units: 20 OPT-125m MLP units, the same indices at every layer (docs/preregistration-f3a-opt-standardisation.md).

    python experiments/f3a_units.py      # writes results/f3a_units.json
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
units = sorted(int(u) for u in np.random.default_rng(20261018).choice(3072, 20, replace=False))
json.dump({"model": "facebook/opt-125m", "seed": 20261018, "units": units}, open(ROOT / "results/f3a_units.json", "w"))
print(units)
