"""S-3: score APERTURE's F1 once with its frozen scorer, plus the declared c05 drift check.

    C:/Users/carbo/projects/mirror/.venv/Scripts/python.exe experiments/score_s3_f1.py

Runs aperture.f1_score.score_directory unchanged (mirror b5bb2fb; 2,000 draws and seed 0, as
the prereg's analysis section fixes) on results/s3_f1/, the 24 files from the 7 Oct session.
All 12 configs enter the pool; none is excluded. The old-stack c05 neutral is in
results/s3_f1/oldstack/ and is not scored. The drift check (kaggle/NEXT_SESSION_S3_F1.md,
decided before the data) compares its chosen option with the new file's, trial by trial.
Not pre-registered, labelled as such: the pooled difference again without the configs over
the prereg's 25% unparseable flag, which the filed pool keeps.
Writes results/s3_f1_score.json.
"""
import json
import os

import numpy as np
from aperture.f1_score import pooled_difference, score_directory
from aperture.forced_choice import concept_abstractness, concept_frequencies
from aperture.concepts import load_bank

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
F1 = os.path.join(ROOT, "results", "s3_f1")
BANK = os.path.join(ROOT, "..", "mirror", "data", "concepts", "dev_bank.yaml")


def rows(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def drift():
    old = rows(os.path.join(F1, "oldstack", "f1_c05_neutral.jsonl"))
    new = rows(os.path.join(F1, "f1_c05_neutral.jsonl"))
    key = lambda r: (r["order_index"], r["concept"])
    o, n = {key(r): r for r in old}, {key(r): r for r in new}
    assert set(o) == set(n), "old and new c05 neutral cover different trials"
    changed = [k for k in o if o[k]["chosen"] != n[k]["chosen"]]
    return {"trials": len(o), "chosen_changed": len(changed),
            "reports_changed": sum(o[k]["report"] != n[k]["report"] for k in o)}


def pooled_without_flagged(result, flag=0.25, n_boot=2000, seed=0):
    flagged = [r["tag"] for r in result["rows"]
               if max(r["unparsed_neutral"], r["unparsed_introspective"]) > flag]
    groups = {t: a for t, a in result["groups"].items() if t not in flagged}
    names = json.load(open(os.path.join(F1, "f1_c00_neutral.config.json")))["names"]
    bank = load_bank(BANK)
    pooled = pooled_difference(groups, names, concept_frequencies(names),
                               concept_abstractness(bank, names), n_boot,
                               np.random.default_rng(seed + len(groups)))
    return {"excluded": flagged, "pooled": pooled}


def plain(x):
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    if isinstance(x, np.generic):
        return x.item()
    return x


def main():
    result = score_directory(F1, BANK)
    result["sensitivity_without_flagged"] = pooled_without_flagged(result)
    result.pop("groups")
    result["c05_drift"] = drift()
    json.dump(plain(result), open(os.path.join(ROOT, "results", "s3_f1_score.json"), "w"),
              indent=2)
    print(result["table"])
    print("predictions", plain(result["predictions"]))
    print("pooled", plain(result["pooled"]))
    print("reference", plain(result["reference"]))
    print("c05 drift", result["c05_drift"])
    print("sensitivity", plain(result["sensitivity_without_flagged"]))


if __name__ == "__main__":
    main()
