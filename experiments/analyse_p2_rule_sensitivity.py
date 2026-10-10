"""Paper 2, outline section 8: sensitivity of the live label to the steering rule.

    PYTHONPATH=. python experiments/analyse_p2_rule_sensitivity.py   # results/p2_rule_sensitivity.json

The filed rule (kaggle_s3_positive_control.steered) counts a text as steered if the concept word or
any listed associate occurs anywhere as a substring, so "sea" matches "search" and "star" matches
"start". A vector is live if its steer-stage text at the gate dose (grid position 2) is steered and
its text at dose 0 is not. Two stricter rules are scored beside it:
  prefix    associates must start a word (stems such as "volcan" or "arachn" still match)
  literal   the concept word itself, as a whole word, singular or plural; no associates
Per session and rule: live vectors per recipe, labels changed against the filed rule, and each
health check's AUC with a stratified bootstrap interval. The checks' values do not depend on the
rule, only the labels do. The filed rows also give the Gemma-3-12B intervals the S-1 analysis did
not store.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import analyse_p2l as p2l  # noqa: E402
import analyse_p2m as p2m  # noqa: E402
import analyse_s1 as s1  # noqa: E402
from kaggle_s3_positive_control import ASSOCIATES, steered as filed_rule  # noqa: E402

GATE = 2
CHECKS = ["norm", "distinctness", "stability", "probe", "logit steering", "P(YES) shift",
          "logit-lens accessibility"]
S2 = [("Qwen2.5-3B", "qwen3b", "_none", "Qwen/Qwen2.5-3B-Instruct"),
      ("Qwen2.5-7B", "qwen7b", "", "Qwen/Qwen2.5-7B-Instruct"),
      ("Gemma-3-4B, KL grid", "gemma4b_kl", "_none", "google/gemma-3-4b-it")]
S1 = [("Gemma-3-12B, 4-bit", "gemma12", "google/gemma-3-12b-it"),
      ("Gemma-3-27B, 4-bit", "gemma27", "google/gemma-3-27b-it")]


def prefix_rule(text, concept):
    t = (text or "").lower()
    return bool(re.search(r"\b" + re.escape(concept.lower()), t)) or any(
        re.search(r"\b" + re.escape(k), t) for k in ASSOCIATES.get(concept, []))


def literal_rule(text, concept):
    return bool(re.search(r"\b" + re.escape(concept.lower()) + r"(s|es)?\b", (text or "").lower()))


RULES = {"filed": filed_rule, "prefix": prefix_rule, "literal": literal_rule}


def live_labels(rows, alphas):
    pos = {a: i for i, a in enumerate(sorted(alphas))}
    by = {}
    for r in rows:
        if r.get("framing", "steer") == "steer":
            by.setdefault(r["concept"], {})[pos[r["alpha"]]] = r.get("text", "")
    return {name: {c: fn(t[GATE], c) and not fn(t[0], c) for c, t in by.items() if GATE in t and 0 in t}
            for name, fn in RULES.items()}


def lap(vectors, repo):
    U, gamma, tok = p2l.weights(repo)
    out = {}
    for key, v in vectors.items():
        t = tok(" " + key[1], add_special_tokens=False)["input_ids"][0]
        logits = U @ (gamma * (v / np.linalg.norm(v)))
        out[key] = float((logits[t] - logits.mean()) / logits.std())
    return out


def session_s2(name, q, repo):
    d = ROOT / f"results/s2_{name}"
    per = {(r["arm"], r["concept"]): r for r in json.load(open(ROOT / f"results/s2_{name}_analysis.json"))["per_vector"]}
    labels, vectors = {}, {}
    for arm, tag in p2l.S2_ARMS.items():
        base = d / f"s2_{name}_{tag}{q}"
        cfg = json.load(open(str(base) + ".config.json"))
        rows = [json.loads(l) for l in open(str(base) + ".jsonl", encoding="utf-8") if l.strip()]
        for rule, lab in live_labels(rows, cfg["alphas"]).items():
            labels.setdefault(rule, {}).update({(arm, c): x for c, x in lab.items()})
        z = np.load(str(base) + ".vectors.npz", allow_pickle=True)
        vectors.update({(arm, str(c)): v for c, v in zip(z["names"], z["vectors"])})
    stats = {k: {c: per[k][c] for c in CHECKS[:-1]} for k in per}
    for k, z in lap(vectors, repo).items():
        stats.setdefault(k, {})["logit-lens accessibility"] = z
    return labels, stats


def session_s1(name, repo):
    cell = s1.load_cell(ROOT / f"results/s1_{name}", name, "4bit")
    labels, stats, vectors = {}, {}, {}
    for arm in s1.ARMS:
        cfg, rows = cell[arm]["cfg"], cell[arm]["rows"]
        for rule, lab in live_labels(rows, cfg["alphas"]).items():
            labels.setdefault(rule, {}).update({(arm, c): x for c, x in lab.items()})
        for c, h in cfg.get("health", {}).items():
            stats[(arm, c)] = {"norm": h.get("norm"), "stability": h.get("stability"), "probe": h.get("probe"),
                               "logit steering": h.get("steer_logit_delta"),
                               "distinctness": -h["max_cos_other"] if h.get("max_cos_other") is not None else None}
        z = np.load(sorted((ROOT / f"results/s1_{name}").glob(f"s1_{name}_4bit_{arm}_*.vectors.npz"))[0],
                    allow_pickle=True)
        vectors.update({(arm, str(c)): v for c, v in zip(z["names"], z["vectors"])})
    for k, z in lap(vectors, repo).items():
        stats.setdefault(k, {})["logit-lens accessibility"] = z
    return labels, stats


def summarise(labels, stats):
    out = {}
    filed = labels["filed"]
    for rule, lab in labels.items():
        keys = sorted(lab)
        L = np.array([lab[k] for k in keys])
        res = {"live": int(L.sum()), "vectors": len(keys),
               "live_by_arm": {a: int(sum(lab[k] for k in keys if k[0] == a)) for a in sorted({k[0] for k in keys})},
               "labels_changed_vs_filed": int(sum(lab[k] != filed[k] for k in keys)), "auc": {}}
        for chk in CHECKS:
            ks = [k for k in keys if stats.get(k, {}).get(chk) is not None]
            x = np.array([stats[k][chk] for k in ks], float)
            y = np.array([lab[k] for k in ks])
            if len(ks) and 5 <= y.sum() <= len(y) - 5:
                res["auc"][chk] = {"auc": p2m.auc(x, y), "ci95": p2m.boot(lambda i: p2m.auc(x[i], y[i]), len(x), strata=y)}
        out[rule] = res
    return out


def main():
    rep = {}
    for label, name, q, repo in S2:
        rep[f"S-2 {label}"] = summarise(*session_s2(name, q, repo))
        print(label, {r: (v["live"], v["labels_changed_vs_filed"]) for r, v in rep[f"S-2 {label}"].items()}, flush=True)
    for label, name, repo in S1:
        rep[f"S-1 {label}"] = summarise(*session_s1(name, repo))
        print(label, {r: (v["live"], v["labels_changed_vs_filed"]) for r, v in rep[f"S-1 {label}"].items()}, flush=True)
    json.dump(rep, open(ROOT / "results/p2_rule_sensitivity.json", "w"), indent=1)


if __name__ == "__main__":
    main()
