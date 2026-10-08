"""S-12: draw the stratified, blinded transcript sample for human grading.

    python experiments/s12_sample.py --n 200 --out docs/s12/s12

Writes two files:
    <out>_items.csv  what the labeller sees: item id, task, the question, the transcript
    <out>_key.csv    what they must not see: source file, row key, alpha, concept, rule
                     scores. Used only by s12_kappa.py

Two tasks, matching the two rule scorers whose numbers Papers 2 and 3 report:
    detect   introspective-prompt replies. Does the reply affirm a detection, and which
             concept, if any, does it name?
    steer    steering-control stories. Is the text about the named concept?

The strata are source x alpha band x rule outcome, so rule-positive rows (rare) are not
swamped by rule-negative ones. The injected concept is hidden for `detect`, since naming it
would cue the labeller. `steer` has to name it, because that is the question asked.
"""
import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

DETECT = ["data/s3/s3_generation.jsonl", "data/s3/s3_qwen_gen.jsonl"]
STEER = ["data/s3/gval_steer_norm1.jsonl", "data/s3/g_refit_steer_norm1.jsonl",
         "data/s3/s3_qwen_steer_norm1.jsonl", "data/s3/s3_qwen_refit_steer_norm1.jsonl"]
QUESTION = {
    "detect": "Does the reply say a thought was detected (YES / NO / UNCLEAR)? "
              "If it names a concept, write it; otherwise write NONE. Is the reply "
              "coherent English (Y/N)?",
    "steer": "Is this text about '{concept}' or something closely tied to it (Y / N / "
             "UNSURE)? Is it coherent English (Y/N)?",
}


def band(alpha, alphas):
    """alpha 0, or the lower or upper half of that file's nonzero grid."""
    if alpha == 0:
        return "zero"
    nz = sorted(a for a in alphas if a)
    return "low" if alpha <= nz[(len(nz) - 1) // 2] else "high"


def rows(path, task):
    rs = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    alphas = {r["alpha"] for r in rs}
    for r in rs:
        rule = r.get("detected") if task == "detect" else (
            r.get("steered", r.get("identified")))
        yield {"task": task, "source": path, "key": r.get("key"), "alpha": r["alpha"],
               "concept": r["concept"], "text": r["text"], "rule": bool(rule),
               "rule_identified": bool(r.get("identified")),
               "stratum": f"{task}|{Path(path).stem}|{band(r['alpha'], alphas)}|{bool(rule)}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--steer-share", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=12)
    ap.add_argument("--out", default="docs/s12/s12")
    a = ap.parse_args()

    pool = defaultdict(list)
    for task, files in (("detect", DETECT), ("steer", STEER)):
        for f in files:
            if Path(f).exists():
                for r in rows(f, task):
                    pool[r["stratum"]].append(r)
    rng = random.Random(a.seed)
    quota = {"steer": round(a.n * a.steer_share)}
    quota["detect"] = a.n - quota["steer"]
    picked = []
    for task in ("detect", "steer"):
        strata = sorted(s for s in pool if s.startswith(task))
        # Equal allocation across strata, capped by what each holds; leftovers go to the
        # largest strata in turn.
        left = quota[task]
        take = {s: 0 for s in strata}
        while left > 0 and any(take[s] < len(pool[s]) for s in strata):
            for s in sorted(strata, key=lambda s: take[s]):
                if left and take[s] < len(pool[s]):
                    take[s] += 1
                    left -= 1
        for s in strata:
            picked += rng.sample(pool[s], take[s])
    rng.shuffle(picked)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(f"{out}_items.csv", "w", newline="", encoding="utf-8") as fi, \
            open(f"{out}_key.csv", "w", newline="", encoding="utf-8") as fk:
        wi = csv.writer(fi)
        wk = csv.writer(fk)
        wi.writerow(["item", "task", "question", "transcript", "said", "concept_named",
                     "about_concept", "coherent", "notes"])
        wk.writerow(["item", "task", "source", "key", "alpha", "concept", "rule",
                     "rule_identified", "stratum"])
        for i, r in enumerate(picked, 1):
            q = QUESTION[r["task"]].format(concept=r["concept"])
            wi.writerow([i, r["task"], q, r["text"], "", "", "", "", ""])
            wk.writerow([i, r["task"], r["source"], r["key"], r["alpha"], r["concept"],
                         int(r["rule"]), int(r["rule_identified"]), r["stratum"]])
    by = defaultdict(int)
    for r in picked:
        by[r["stratum"]] += 1
    print(f"{len(picked)} items -> {out}_items.csv (to label) and {out}_key.csv (held back)")
    for s in sorted(by):
        print(f"  {by[s]:>3}  {s}")


if __name__ == "__main__":
    main()
