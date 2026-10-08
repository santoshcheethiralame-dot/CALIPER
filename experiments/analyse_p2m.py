"""P2-M: is the inverted P(YES) check output steering? (docs/preregistration-p2m-output-steering.md)

    python experiments/analyse_p2m.py --model qwen3b --hub Qwen/Qwen2.5-3B-Instruct
    python experiments/analyse_p2m.py --model qwen7b --hub Qwen/Qwen2.5-7B-Instruct --quant 4bit
    python experiments/analyse_p2m.py --model gemma4b_kl --weights /kaggle/input/.../gemma-3-4b-it

Per real-arm vector v (unit, as injected), the logit-lens answer score
    a(v) = (mean_{t in YES} u_t - mean_{t in NO} u_t) . (gamma * v)
with u the unembedding rows, YES / NO the script's first-token id sets, gamma the final norm's
gain (Gemma: 1 + weight). Only the unembedding and the final norm are read: from a local
safetensors directory with --weights, or with --hub by HTTP range requests for just those two
tensors (no full shard is downloaded).

Primary: Spearman rho(a, P(YES)-shift t) per model, bootstrap 95% CI over vectors.
Secondary: AUC of a at live vs dead (live > dead), the P(YES)-shift AUC after removing a from
the ranks, and both by arm.
"""
import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
ARMS = {"concept token": "concept_steer", "template tail": "tail_steer",
        "sentence mean": "sentence_steer_aperture"}
DT = {"BF16": "bf16", "F16": np.float16, "F32": np.float32}


def _decode(raw, dtype, shape):
    if DT[dtype] == "bf16":
        u = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
        return u.view(np.float32).reshape(shape)
    return np.frombuffer(raw, dtype=DT[dtype]).astype(np.float32).reshape(shape)


class Local:
    def __init__(self, d):
        self.d = Path(d)

    def text(self, name):
        return (self.d / name).read_text(encoding="utf-8")

    def exists(self, name):
        return (self.d / name).exists()

    def read(self, name, lo, hi):
        with open(self.d / name, "rb") as f:
            f.seek(lo)
            return f.read(hi - lo)

    def tokenizer_dir(self):
        return str(self.d)


class Hub:
    def __init__(self, repo):
        import os
        import requests
        self.repo, self.s = repo, requests.Session()
        tok = os.environ.get("HF_TOKEN")
        if tok:
            self.s.headers["Authorization"] = f"Bearer {tok}"

    def url(self, name):
        return f"https://huggingface.co/{self.repo}/resolve/main/{name}"

    def text(self, name):
        r = self.s.get(self.url(name), timeout=60)
        r.raise_for_status()
        return r.text

    def exists(self, name):
        return self.s.head(self.url(name), allow_redirects=True, timeout=60).status_code == 200

    def read(self, name, lo, hi):
        r = self.s.get(self.url(name), headers={"Range": f"bytes={lo}-{hi - 1}"}, timeout=600)
        r.raise_for_status()
        assert len(r.content) == hi - lo, "range request returned the wrong length"
        return r.content

    def tokenizer_dir(self):
        from huggingface_hub import snapshot_download
        return snapshot_download(self.repo, allow_patterns=["tokenizer*", "*.json", "*.model",
                                                            "*.txt"],
                                 ignore_patterns=["*.safetensors*"])


def tensors(src, wanted):
    """Read the named tensors from a safetensors checkpoint, header first, then only their
    bytes."""
    if src.exists("model.safetensors.index.json"):
        wmap = json.loads(src.text("model.safetensors.index.json"))["weight_map"]
    else:
        wmap = None
    names = list(wmap) if wmap else None
    if names is None:
        n = struct.unpack("<Q", src.read("model.safetensors", 0, 8))[0]
        names = [k for k in json.loads(src.read("model.safetensors", 8, 8 + n)) if k != "__metadata__"]
    out = {}
    for key in wanted(names):
        f = wmap[key] if wmap else "model.safetensors"
        n = struct.unpack("<Q", src.read(f, 0, 8))[0]
        meta = json.loads(src.read(f, 8, 8 + n))[key]
        lo, hi = meta["data_offsets"]
        out[key] = _decode(src.read(f, 8 + n + lo, 8 + n + hi), meta["dtype"], meta["shape"])
    return out


def pick(names):
    unemb = [k for k in names if k.endswith("lm_head.weight")] or \
            [k for k in names if k.endswith("embed_tokens.weight")]
    norm = [k for k in names if k.endswith("norm.weight") and "layers." not in k
            and "vision" not in k and "multi_modal" not in k]
    assert unemb and len(norm) >= 1, (unemb, norm)
    return [unemb[0], sorted(norm, key=len)[0]]


def answer_direction(src):
    from transformers import AutoTokenizer
    from kaggle_s3_positive_control import yes_no_ids
    cfg = json.loads(src.text("config.json"))
    gemma = "gemma" in json.dumps(cfg.get("model_type", "")) + json.dumps(cfg.get("architectures", ""))
    t = tensors(src, pick)
    u, g = [t[k] for k in pick(list(t))]
    gamma = (1.0 + g) if gemma else g
    ids = yes_no_ids(AutoTokenizer.from_pretrained(src.tokenizer_dir()))
    return gamma * (u[ids["yes"]].mean(0) - u[ids["no"]].mean(0)), ids


def auc(score, live):
    score, live = np.asarray(score, float), np.asarray(live, bool)
    m, n = live.sum(), (~live).sum()
    r = rankdata(score)
    return float((r[live].sum() - m * (m + 1) / 2) / (m * n))


def boot(fn, n, n_boot=2000, seed=0, strata=None):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        if strata is None:
            i = rng.integers(0, n, n)
        else:
            li, di = np.where(strata)[0], np.where(~strata)[0]
            i = np.concatenate([rng.choice(li, len(li)), rng.choice(di, len(di))])
        out.append(fn(i))
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def score(model, d, q, direction):
    per = {(r["arm"], r["concept"]): r for r in
           json.load(open(ROOT / f"results/s2_{model}_analysis.json"))["per_vector"]}
    A, T, live, arm_of = [], [], [], []
    for arm, tag in ARMS.items():
        z = np.load(ROOT / d / f"s2_{model}_{tag}{q}.vectors.npz", allow_pickle=True)
        for name, v in zip(z["names"], z["vectors"]):
            r = per[(arm, str(name))]
            A.append(float(direction @ (v / np.linalg.norm(v))))
            t = r["P(YES) shift"]
            T.append(np.sign(t) * 1e9 if np.isinf(t) else t)
            live.append(r["live"])
            arm_of.append(arm)
    A, T, live, arm_of = np.array(A), np.array(T), np.array(live), np.array(arm_of)
    rho = float(spearmanr(A, T)[0])
    rA, rT = rankdata(A), rankdata(T)
    resid = rT - np.polyval(np.polyfit(rA, rT, 1), rA)
    rep = {"vectors": len(A), "live": int(live.sum()),
           "primary": {"spearman_rho": rho,
                       "ci95": boot(lambda i: spearmanr(A[i], T[i])[0], len(A)),
                       "holds": None},
           "auc_answer_score_live_gt_dead": {"auc": auc(A, live),
                                             "ci95": boot(lambda i: auc(A[i], live[i]), len(A),
                                                          strata=live)},
           "auc_pyes_shift": auc(T, live),
           "auc_pyes_shift_after_removing_answer_score": auc(resid, live),
           "by_arm": {a: {"spearman_rho": float(spearmanr(A[arm_of == a], T[arm_of == a])[0]),
                          "live": int(live[arm_of == a].sum())} for a in ARMS},
           "per_vector": [{"arm": a, "answer_score": float(x), "pyes_t": float(t), "live": bool(l)}
                          for a, x, t, l in zip(arm_of, A, T, live)]}
    rep["primary"]["holds"] = rho > 0 and rep["primary"]["ci95"][0] > 0
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="S-2 run name, e.g. qwen3b")
    ap.add_argument("--quant", default="none", choices=("none", "4bit", "8bit"))
    ap.add_argument("--dir", default=None, help="default results/s2_<model>")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--weights", help="local model directory with safetensors")
    src.add_argument("--hub", help="Hugging Face repo id; reads two tensors by range request")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    direction, ids = answer_direction(Local(a.weights) if a.weights else Hub(a.hub))
    q = "" if a.quant == "4bit" else f"_{a.quant}"
    rep = {"model": a.model, "yes_ids": ids["yes"], "no_ids": ids["no"],
           **score(a.model, a.dir or f"results/s2_{a.model}", q, direction)}
    print(json.dumps({k: v for k, v in rep.items() if k != "per_vector"}, indent=1))
    json.dump(rep, open(ROOT / (a.out or f"results/p2m_{a.model}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
