"""The T-SAE operating-point rule (Amendment 1, completed by Amendment 2), on failure counts only.

    python experiments/tsae_operating_point.py results/tsae0_a.jsonl results/tsae0_b.jsonl

Prints "tokens steps n min_events" for the main run. Reads align_selected and nothing else:
the reliability signals under test are never looked at here.
"""
import json
import sys

BAR = 0.95
A = (8000, 1600, 100)    # tokens, steps, min firing events in the fitted stimulus
B = (16000, 3200, 200)


def failures(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    if len(rows) != 16:
        sys.exit(f"{path}: {len(rows)} rows, the pilot has 16")
    return sum(r["align_selected"] < BAR for r in rows)


def choose(fa, fb):
    """Failure counts out of 16 -> (budget, n). 20-80% is 4-12 of 16."""
    if 4 <= fa <= 12:
        return A, 200
    if 4 <= fb <= 12:
        return B, 200
    if fa > 12 and fb > 12:
        return B, 100
    if fa < 4:
        return A, 200
    return B, 200    # A > 12 and B < 4: the larger budget is the one that fits


if __name__ == "__main__":
    fa, fb = failures(sys.argv[1]), failures(sys.argv[2])
    (tokens, steps, events), n = choose(fa, fb)
    print(f"{tokens} {steps} {n} {events}  # failures A={fa}/16 B={fb}/16")
