"""Is the depth effect in B-7 about layers, or about specific units?

B-7 swept GPT-2 layers 2, 6 and 10 over the SAME 50 units - a paired design, because the
unit draw depends only on the seed and d_model, both of which are fixed across layers. The
three pass rates (0.48 / 0.86 / 0.90) are therefore comparable within unit, not just
between samples, and that permits a question the aggregate rates cannot answer:

    are the units that fail at L2 the SAME units that fail at L6 and L10?

Two readings of the depth effect are consistent with the aggregate numbers and mean
opposite things.

  SHARED CORE - the same units are unrecoverable at every depth, and shallow layers
                 additionally break some units that deep layers handle. Prediction: a
                 non-trivial set fails at all three layers.

  DEPTH-BROKEN - the units that fail at L2 mostly SUCCEED at L6 and L10. Prediction: the
                 three failure sets overlap little, and most L2 failures recover with
                 depth.

The distinction decides the write-up. "Shallow layers are noisier" and "these particular
units only work at some depths" are different claims about the instrument, and only one of
them is a depth effect in the usual sense.

A second question needs a different comparison. Depth varies the layer; restart count
varies the estimator's own randomness while holding the layer fixed. B-7 layer 6 at 5
restarts and B-1b layer 6 at 2 restarts share all 50 units, so they isolate the second.
That one says how much of any single pass rate is the unit and how much is the draw, which
bounds how finely the depth numbers can be read.

Reads the jsonl files and reports both. No model fitting: arithmetic over rows on disk.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
LAYERS = ["02", "06", "10"]
BAR = 0.95
GAIN_MAX = 0.01


def load(name: str) -> dict[int, dict]:
    rows = {}
    with (OUT / name).open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows[int(row["_key"])] = row
    return rows


def passed(row: dict) -> bool:
    """Same rule as e01_gate.py: alignment over the bar AND no k2 gain."""
    return row["align_selected"] > BAR and row["k2_gain"] < GAIN_MAX


def median(values: list[float]) -> float:
    s = sorted(values)
    return s[len(s) // 2] if len(s) % 2 else (s[len(s) // 2 - 1] + s[len(s) // 2]) / 2


def depth_structure(data: dict[str, dict[int, dict]], units: list[int]) -> None:
    n = len(units)
    flags = {layer: {u: passed(data[layer][u]) for u in units} for layer in LAYERS}

    print(f"\npaired units: {n} (identical draw across L2/L6/L10)\n")
    print(f"{'pattern':<16}{'count':>6}   L2 L6 L10")
    counts: dict[tuple[bool, bool, bool], int] = {}
    for u in units:
        pat = tuple(flags[layer][u] for layer in LAYERS)
        counts[pat] = counts.get(pat, 0) + 1
    for pat in sorted(counts, key=lambda p: -counts[p]):
        cells = " ".join("  P " if p else "  . " for p in pat)
        print(f"{str(pat):<16}{counts[pat]:>6}   {cells}")

    always_fail = [u for u in units if not any(flags[layer][u] for layer in LAYERS)]
    l2_fail = [u for u in units if not flags["02"][u]]
    l2_only = [u for u in units if not flags["02"][u] and flags["06"][u] and flags["10"][u]]
    broke = [u for u in units if flags["02"][u] and not flags["10"][u]]

    print("\ndepth reading")
    print(f"  fail at all three layers (shared core) : {len(always_fail)}/{n}")
    print(f"  fail at L2                            : {len(l2_fail)}/{n}")
    print(f"  ...of those, pass at BOTH L6 and L10  : {len(l2_only)}/{n}")
    print(f"  pass at L2 but FAIL at L10            : {len(broke)}/{n}")
    print(f"  pass everywhere                       : "
          f"{sum(1 for u in units if all(flags[l][u] for l in LAYERS))}/{n}")

    if l2_only:
        print("\n  the L2-only failures, median selected alignment")
        for layer in LAYERS:
            print(f"    L{layer}: {median([data[layer][u]['align_selected'] for u in l2_only]):.4f}")
    if broke:
        print("\n  the units that pass at L2 and fail at L10, median selected alignment")
        for layer in LAYERS:
            print(f"    L{layer}: {median([data[layer][u]['align_selected'] for u in broke]):.4f}")


def restart_stability() -> None:
    b7 = load("b7_layer06_gpt2.jsonl")
    b1b = load("b1b_primary_gpt2.jsonl")
    units = sorted(set(b7) & set(b1b))
    if not units:
        print("\nno overlap between B-7 L6 and B-1b; skipping restart check")
        return

    both_pass = [u for u in units if passed(b7[u]) and passed(b1b[u])]
    both_fail = [u for u in units if not passed(b7[u]) and not passed(b1b[u])]
    only_r5 = [u for u in units if not passed(b7[u]) and passed(b1b[u])]
    only_r2 = [u for u in units if passed(b7[u]) and not passed(b1b[u])]
    agree = len(both_pass) + len(both_fail)

    print(f"\nrestart stability at fixed layer 6 (B-7 at 5 restarts vs B-1b at 2)")
    print(f"  shared units            : {len(units)}")
    print(f"  verdict agreement       : {agree}/{len(units)} "
          f"({100 * agree / len(units):.1f}%)")
    print(f"  both PASS               : {len(both_pass)}")
    print(f"  both FAIL (stable core) : {len(both_fail)}")
    print(f"  fail only at 5 restarts : {len(only_r5)}")
    print(f"  fail only at 2 restarts : {len(only_r2)}")


def main() -> None:
    data = {layer: load(f"b7_layer{layer}_gpt2.jsonl") for layer in LAYERS}
    units = sorted(set.intersection(*(set(d) for d in data.values())))
    depth_structure(data, units)
    restart_stability()


if __name__ == "__main__":
    main()