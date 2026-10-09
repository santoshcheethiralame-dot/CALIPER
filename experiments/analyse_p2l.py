"""P2-L: does logit-lens accessibility tell live vectors from dead ones?
(docs/preregistration-p2l-logit-lens-health.md)

    python experiments/analyse_p2l.py                    # every session, once

Per real-arm vector v (unit, as injected): logits = W_U (gamma * v) and
z(v) = (logits[t_c] - mean) / sd, t_c the first token of " {concept}". Weights are read with
analyse_p2m's safetensors reader (unembedding and final norm only) and cached under
~/.cache/caliper_unembed. Live labels exactly as each study filed them: S-2 from
analyse_s2's output, S-1 from its 0.5-nat gate (analyse_s1.passes).
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import analyse_p2m as p2m  # noqa: E402
import analyse_s1 as s1  # noqa: E402

CACHE = Path.home() / ".cache" / "caliper_unembed"
S2_ARMS = {"concept token": "concept_steer", "template tail": "tail_steer",
           "sentence mean": "sentence_steer_aperture"}
SESSIONS = [  # name, hub repo, kind, location, scored
    ("S-2 Qwen2.5-3B", "Qwen/Qwen2.5-3B-Instruct", "s2", ("qwen3b", "results/s2_qwen3b", "_none"), True),
    ("S-2 Qwen2.5-7B", "Qwen/Qwen2.5-7B-Instruct", "s2", ("qwen7b", "results/s2_qwen7b", ""), True),
    ("S-2 Gemma-3-4B, KL grid", "google/gemma-3-4b-it", "s2",
     ("gemma4b_kl", "results/s2_gemma4b_kl", "_none"), True),
    ("S-1 Gemma-3-12B, 4-bit", "google/gemma-3-12b-it", "s1", ("gemma12", "results/s1_gemma12"), True),
    ("S-1 Gemma-3-27B, 4-bit", "google/gemma-3-27b-it", "s1", ("gemma27", "results/s1_gemma27"), True),
    ("S-2 Gemma-3-4B, alpha-frac (failed manipulation check)", "google/gemma-3-4b-it", "s2",
     ("gemma4b", "results/s2_gemma4b", "_none"), False),
]


def weights(repo):
    """gamma * unembedding rows, as float32 (vocab, d), plus the tokenizer directory."""
    from transformers import AutoTokenizer
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / (repo.replace("/", "__") + ".npz")
    src = p2m.Hub(repo)
    if not f.exists():
        cfg = json.loads(src.text("config.json"))
        gemma = "gemma" in json.dumps(cfg.get("model_type", "")) + json.dumps(cfg.get("architectures", ""))
        t = p2m.tensors(src, p2m.pick)
        u, g = [t[k] for k in p2m.pick(list(t))]
        np.savez(f, u=u.astype(np.float16), gamma=((1.0 + g) if gemma else g).astype(np.float32))
    z = np.load(f)
    return z["u"].astype(np.float32), z["gamma"], AutoTokenizer.from_pretrained(src.tokenizer_dir())


def vectors_s2(name, d, q):
    per = {(r["arm"], r["concept"]): r for r in
           json.load(open(ROOT / f"results/s2_{name}_analysis.json"))["per_vector"]}
    out = []
    for arm, tag in S2_ARMS.items():
        z = np.load(ROOT / d / f"s2_{name}_{tag}{q}.vectors.npz", allow_pickle=True)
        for c, v in zip(z["names"], z["vectors"]):
            r = per[(arm, str(c))]
            out.append((arm, str(c), v, r["live"], r["logit steering"]))
    return out


def vectors_s1(name, d):
    cell = s1.load_cell(ROOT / d, name, "4bit")
    out = []
    for arm in s1.ARMS:
        live = s1.passes(cell, arm, 2)
        health = cell[arm]["cfg"].get("health", {})
        z = np.load(sorted((ROOT / d).glob(f"s1_{name}_4bit_{arm}_*.vectors.npz"))[0], allow_pickle=True)
        for c, v in zip(z["names"], z["vectors"]):
            out.append((arm, str(c), v, live[str(c)], health.get(str(c), {}).get("steer_logit_delta")))
    return out


def score(rows, U, gamma, tok):
    A, L, arm_of, steer = [], [], [], []
    for arm, c, v, live, sl in rows:
        t = tok(" " + c, add_special_tokens=False)["input_ids"][0]
        logits = U @ (gamma * (v / np.linalg.norm(v)))
        A.append(float((logits[t] - logits.mean()) / logits.std()))
        L.append(bool(live))
        arm_of.append(arm)
        steer.append(np.nan if sl is None else float(sl))
    A, L, arm_of, steer = np.array(A), np.array(L), np.array(arm_of), np.array(steer)
    rep = {"vectors": len(A), "live": int(L.sum()),
           "scored": bool(5 <= L.sum() <= len(L) - 5)}
    if rep["scored"]:
        rep["auc"] = p2m.auc(A, L)
        rep["ci95"] = p2m.boot(lambda i: p2m.auc(A[i], L[i]), len(A), strata=L)
        ok = ~np.isnan(steer)
        if ok.sum() > 10 and 0 < L[ok].sum() < ok.sum():
            rep["auc_logit_steering"] = p2m.auc(steer[ok], L[ok])
            rep["diff_vs_logit_steering_ci95"] = p2m.boot(
                lambda i: p2m.auc(A[ok][i], L[ok][i]) - p2m.auc(steer[ok][i], L[ok][i]),
                int(ok.sum()), strata=L[ok])
        rep["within_arm_auc"] = {a: p2m.auc(A[arm_of == a], L[arm_of == a])
                                 for a in sorted(set(arm_of))
                                 if 0 < L[arm_of == a].sum() < (arm_of == a).sum()}
    rep["median_z_live_dead"] = [float(np.median(A[L])) if L.any() else None,
                                 float(np.median(A[~L])) if (~L).any() else None]
    return rep


def main():
    out, cache = {}, {}
    for name, repo, kind, loc, scored in SESSIONS:
        if repo not in cache:
            print(f"weights {repo} ...", flush=True)
            cache[repo] = weights(repo)
        rows = vectors_s2(*loc) if kind == "s2" else vectors_s1(*loc)
        rep = score(rows, *cache[repo])
        rep["counts_for_primary"] = scored and rep["scored"]
        out[name] = rep
        print(name, json.dumps({k: v for k, v in rep.items() if k != "within_arm_auc"}), flush=True)
    primary = [v for v in out.values() if v["counts_for_primary"]]
    out["primary"] = {"scored_sessions": len(primary),
                      "ci_above_half": sum(v["ci95"][0] > 0.5 for v in primary),
                      "holds": sum(v["ci95"][0] > 0.5 for v in primary) >= 3}
    print(json.dumps(out["primary"]))
    json.dump(out, open(ROOT / "results/p2l_logit_lens_health.json", "w"), indent=1)


if __name__ == "__main__":
    main()
