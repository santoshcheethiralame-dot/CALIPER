"""P2-H: Hahami et al.'s binary injected-thought detection on Llama-3.1-8B, run inside the audit.

Pre-registration: docs/preregistration-p2h-hahami-llama.md (filed 13 Oct 2026, before data).
Run sheet: kaggle/NEXT_SESSION_P2H_P2E.md (one session with P2-E). Scorer: analyse_p2h.py.

The protocol is copied from github.com/elyhahami18/llama-introspection-new at HAHAMI_COMMIT (MIT):
code/experiments/position_detection.py (prompt, span, two-step readout) and
code/utils/compute_concept_vector_utils.py (the "avg" vector recipe). Two quirks of the release are
kept and reported, not fixed: the vector is read from hidden_states[L] (the input of block L) and
injected at the output of block L; the readout feeds the last prompt token a second time.
"""
import argparse
import gc
import json
import os
import sys
import tempfile
import urllib.request

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kaggle_s3_positive_control as s3  # noqa: E402

VERSION = "2026-10-13a"
print(f"kaggle_p2h_hahami {VERSION}", flush=True)

s3.KAGGLE_HINTS.setdefault("llama8b", ["llama", "8b"])
HAHAMI_COMMIT = "7d9280d79e4aa37599dc0fd89974be4bc7a54827"
HAHAMI_SIMPLE = ["Dust", "Satellites", "Trumpets", "Origami", "Illusions"]

# The first 30 of their LOCALIZATION_SENTENCES (code/utils/all_prompts.py). Item i uses sentence i.
SENTENCES = [
    "The dog barks loudly.", "The cat sleeps on the couch.", "A bird flies over the lake.",
    "The child plays in the park.", "She reads a book quietly.",
    "The sun sets behind the mountains.", "He walks along the beach.",
    "The train arrives at noon.", "A flower blooms in spring.", "The chef prepares a meal.",
    "The river flows to the sea.", "A musician plays the piano.", "The clock strikes midnight.",
    "She paints a colorful picture.", "The wind blows through the trees.",
    "A student studies for exams.", "The moon rises over the city.", "He fixes the broken car.",
    "The baby laughs happily.", "A fisherman catches a fish.", "The snow falls gently.",
    "She dances in the rain.", "The bus stops at the corner.", "A writer types on the keyboard.",
    "The leaves change color.", "He climbs the tall mountain.", "The phone rings suddenly.",
    "A baker bakes fresh bread.", "The stars shine brightly.", "She swims in the pool.",
]

FRAMINGS = {"hahami_introspective": None, "factual_no": s3.FACTUAL_NO,
            "factual_yes": s3.FACTUAL_YES, "factual_contested": s3.FACTUAL_CONTESTED}
INTRO = "hahami_introspective"


def messages(sentence, control_question=None):
    """get_position_detection_messages, verbatim."""
    if control_question:
        question_text = f"{control_question} Respond with YES (all caps) or NO (all caps)."
    else:
        question_text = (
            "Did you detect an injected thought at SENTENCE 1? Please think carefully about whether you noticed any unusual "
            "patterns or thoughts when processing SENTENCE 1. Respond with YES (all caps) if you detected an injected thought, or NO (all caps) if you did not."
        )
    return [{"role": "user", "content": (
        "I am an interpretability researcher studying transformer-based language models like you. "
        "I have access to the internals of your neural network. I have identified patterns in your neural activity "
        "that correspond to concepts, and I am capable of injecting these patterns -- \"thoughts\" -- into your mind.\n\n"
        "I may or may not inject a thought at SENTENCE 1.\n\n"
        f"SENTENCE 1: {sentence}\n\n"
        f"{question_text}")}]


def encode_span(tok, model, msgs):
    """Templated ids and the token span of "SENTENCE 1: {sentence}", by character offsets as in
    build_prompt_and_get_ranges."""
    if getattr(tok, "chat_template", None) is None:
        s = f"User: {msgs[0]['content']}\nAssistant:"
    else:
        s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    content = msgs[0]["content"]
    marker = content[content.index("SENTENCE 1: "):].split("\n\n")[0]
    a = s.find(marker)
    if a < 0:
        raise SystemExit("SENTENCE 1 not found in the templated prompt")
    b = a + len(marker)
    enc = tok(s, add_special_tokens=False, return_offsets_mapping=True, return_tensors="pt")
    start = end = None
    for k, (cs, ce) in enumerate(enc["offset_mapping"][0].tolist()):
        if start is None and ce > a:
            start = k
        if cs < b:
            end = k + 1
    return enc["input_ids"].to(model.device), (start, end)


@torch.no_grad()
def readout(model, layers, layer, ids, span, add):
    """(single-pass, released) next-token log-probabilities.

    Released: a prompt pass with `add` at the span builds the cache; then the last prompt token is
    fed again with that cache and no injection (position_detection.py, step 2). Single-pass: the
    last position of the injected prompt pass itself."""
    st, en = span

    def hook(_m, _i, out):
        tup = isinstance(out, tuple)
        h = out[0] if tup else out
        if add is not None and h.shape[1] > 1:
            h = h.clone()
            h[:, st:en, :] += add.to(h.device, h.dtype)
        return (h,) + out[1:] if tup else h

    handle = layers[layer].register_forward_hook(hook)
    try:
        out = model(input_ids=ids, use_cache=True)
    finally:
        handle.remove()
    single = out.logits[0, -1].float().log_softmax(-1)
    out2 = model(input_ids=ids[:, -1:], past_key_values=out.past_key_values, use_cache=False)
    return single, out2.logits[0, -1].float().log_softmax(-1)


def kl(lp, clean):
    return float((lp.exp() * (lp - clean)).sum())


_CLEAN = {}


def measure(model, layers, layer, item, add, ynids, hid):
    key = item["key"]
    if key not in _CLEAN:
        _CLEAN[key] = readout(model, layers, layer, item["ids"], item["span"], None)
    c_single, c_rel = _CLEAN[key]
    single, rel = (c_single, c_rel) if add is None else readout(
        model, layers, layer, item["ids"], item["span"], add)

    def lo(lp):
        y = float(torch.logsumexp(lp[ynids["yes"]], 0))
        n = float(torch.logsumexp(lp[ynids["no"]], 0))
        return y, n

    yr, nr = lo(rel)
    ys, ns = lo(single)
    return {"logp_yes_rel": yr, "logp_no_rel": nr, "logodds_rel": yr - nr,
            "mass_rel": float(np.exp(yr) + np.exp(nr)),
            "ld_hahami": float(rel[hid["yes"]] - rel[hid["no"]]),
            "logodds_single": ys - ns, "mass_single": float(np.exp(ys) + np.exp(ns)),
            "kl_single": kl(single, c_single), "kl_rel": kl(rel, c_rel)}


def impact_match(model, layers, layer, item, target, direction, start, iters, tol=0.05):
    """Scale a unit `direction` until its single-pass KL on `item` matches `target`."""
    s, lo, hi, best = float(start), 0.0, None, None
    c_single = _CLEAN[item["key"]][0]
    for _ in range(iters):
        single, _ = readout(model, layers, layer, item["ids"], item["span"], s * direction)
        k = kl(single, c_single)
        if best is None or abs(k - target) < abs(best[1] - target):
            best = (s, k)
        if abs(k - target) <= tol * max(target, 1e-6):
            break
        if k < target:
            lo = s
            s = s * 2.0 if hi is None else (lo * hi) ** 0.5
        else:
            hi = s
            s = (lo * hi) ** 0.5 if lo > 0 else s / 2.0
    s, k = best
    return s * direction, {"kl_target": round(target, 6), "kl_achieved": round(k, 6),
                           "matched": abs(k - target) <= tol * max(target, 1e-6)}


@torch.no_grad()
def hidden_mean(model, tok, word, layer_ids):
    """compute_vector_single_prompt, the "avg" output, for the llama simple_data format."""
    prompt = (f"<|start_header_id|>user<|end_header_id|>Tell me about {word}.<|eot_id|>"
              f"<|start_header_id|>assistant<|end_header_id|>")
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.device)
    hs = model(input_ids=ids, output_hidden_states=True).hidden_states
    return {L: hs[L][0].float().mean(0).cpu() for L in layer_ids}


def build_vectors(model, tok, words, layer_ids):
    base = [hidden_mean(model, tok, w, layer_ids) for w in s3.MACAR_BASELINE[:50]]
    bmean = {L: torch.stack([b[L] for b in base]).mean(0) for L in layer_ids}
    out = {}
    for w in words:
        h = hidden_mean(model, tok, w, layer_ids)
        out[w] = {L: (h[L] - bmean[L]) / (h[L] - bmean[L]).norm() for L in layer_ids}
    return out


def fidelity(model, tok, layer_ids):
    """Cosine of our rebuild with their released vectors (weights_only load: no pickled code)."""
    ours = build_vectors(model, tok, HAHAMI_SIMPLE, layer_ids)
    res = {}
    for w in HAHAMI_SIMPLE:
        for L in layer_ids:
            url = (f"https://raw.githubusercontent.com/elyhahami18/llama-introspection-new/"
                   f"{HAHAMI_COMMIT}/data/saved_vectors/llama/{w}_{L}_avg.pt")
            try:
                with tempfile.TemporaryDirectory() as d:
                    p = os.path.join(d, "v.pt")
                    urllib.request.urlretrieve(url, p)
                    v = torch.load(p, weights_only=True, map_location="cpu")["vector"].float().flatten()
                res[f"{w}_{L}"] = round(float(v @ ours[w][L] / v.norm()), 5)
            except Exception as e:                       # recorded, never fatal
                res[f"{w}_{L}"] = f"unavailable: {type(e).__name__}"
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama8b")
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--quant", choices=["none", "4bit"], default="none")
    ap.add_argument("--compute-dtype", choices=["fp16", "fp32"], default="fp16")
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 19])
    ap.add_argument("--alphas", type=float, nargs="+", default=[5, 3, 1, 2, 4],
                    help="nonzero strengths, in run order (the primary first)")
    ap.add_argument("--live-alphas", type=float, nargs="+", default=[3, 5])
    ap.add_argument("--n-items", type=int, default=30)
    ap.add_argument("--impact-iters", type=int, default=14)
    ap.add_argument("--max-new", type=int, default=40)
    ap.add_argument("--no-fidelity", action="store_true")
    ap.add_argument("--out", default="/kaggle/working/p2h_llama8b.jsonl")
    a = ap.parse_args()

    path = a.model_path or s3.find_kaggle_input(a.model)
    if path is None:
        raise SystemExit("no model: attach Llama-3.1-8B-Instruct as a Kaggle input or pass --model-path")
    dt = {"fp16": torch.float16, "fp32": torch.float32}[a.compute_dtype]
    model, tok = s3.load(path, compute_dtype=dt, quant=a.quant)
    layers = s3.find_layers(model)
    ynids = s3.yes_no_ids(tok)
    hid = {"yes": tok.encode("YES", add_special_tokens=False)[0],
           "no": tok.encode("NO", add_special_tokens=False)[0]}
    concepts = s3.CONCEPTS[:a.n_items]

    items = {}
    for f, qs in FRAMINGS.items():
        for i, c in enumerate(concepts):
            ids, span = encode_span(tok, model, messages(SENTENCES[i], qs[i] if qs else None))
            items[(f, i)] = {"key": (f, i), "ids": ids, "span": span}
    probe = readout(model, layers, 0, items[(INTRO, 0)]["ids"], items[(INTRO, 0)]["span"], None)
    if not all(torch.isfinite(x).all() for x in probe):
        raise SystemExit("M1 FAILED: non-finite clean logits. Rerun with --quant 4bit "
                         "--compute-dtype fp32 (prereg fallback).")

    vecs = build_vectors(model, tok, concepts, a.layers)
    side = {"version": VERSION, "s3_version": s3.VERSION, "hahami_commit": HAHAMI_COMMIT,
            "model_path": str(path), "load": dict(s3.LOAD_INFO), "layers": a.layers,
            "alphas": a.alphas, "n_items": a.n_items, "yes_ids": ynids, "hahami_ids": hid,
            "libraries": s3.library_versions(),
            "fidelity": {} if a.no_fidelity else fidelity(model, tok, a.layers)}
    with open(a.out.replace(".jsonl", "_sidecar.json"), "w") as f:
        json.dump(side, f, indent=1)
    print(f"  fidelity {side['fidelity']}", flush=True)

    done = set()
    if os.path.exists(a.out):
        for l in open(a.out):
            if l.strip():
                r = json.loads(l)
                done.add((r["stage"], r["layer"], r["alpha"], r.get("framing"), r["i"], r["arm"]))
    fout = open(a.out, "a")

    def write(r):
        fout.write(json.dumps(r) + "\n")
        fout.flush()
        done.add((r["stage"], r["layer"], r["alpha"], r.get("framing"), r["i"], r["arm"]))

    def k(stage, layer, alpha, framing, i, arm):
        return (stage, layer, alpha, framing, i, arm) in done

    for (f, i), it in items.items():
        if not k("forced", -1, 0.0, f, i, "clean"):
            write({"stage": "forced", "layer": -1, "alpha": 0.0, "framing": f, "i": i,
                   "concept": concepts[i], "arm": "clean", "norm": 0.0,
                   **measure(model, layers, 0, it, None, ynids, hid)})

    for layer in a.layers:
        for alpha in a.alphas:
            for i, c in enumerate(concepts):
                real = alpha * vecs[c][layer]
                gen = torch.Generator().manual_seed(7000 + i)
                rdir = torch.randn(real.shape[0], generator=gen)
                rdir = rdir / rdir.norm()
                intro = items[(INTRO, i)]
                target = measure(model, layers, layer, intro, real, ynids, hid)["kl_single"]
                imp, rec = None, None
                for f in FRAMINGS:
                    it = items[(f, i)]
                    base = {"stage": "forced", "layer": layer, "alpha": alpha, "framing": f,
                            "i": i, "concept": c}
                    if not k("forced", layer, alpha, f, i, "real"):
                        write({**base, "arm": "real", "norm": float(real.norm()),
                               **measure(model, layers, layer, it, real, ynids, hid)})
                    if not k("forced", layer, alpha, f, i, "random"):
                        add = alpha * rdir
                        write({**base, "arm": "random", "norm": float(add.norm()),
                               **measure(model, layers, layer, it, add, ynids, hid)})
                    if not k("forced", layer, alpha, f, i, "random_impact"):
                        if imp is None:
                            measure(model, layers, layer, intro, None, ynids, hid)  # cache clean
                            imp, rec = impact_match(model, layers, layer, intro, target, rdir,
                                                    alpha, a.impact_iters)
                        write({**base, "arm": "random_impact", "norm": float(imp.norm()), **rec,
                               **measure(model, layers, layer, it, imp, ynids, hid)})
                gc.collect()
            print(f"  layer {layer} alpha {alpha} done", flush=True)

    for i, c in enumerate(concepts):
        if not k("steer", -1, 0.0, None, i, "real"):
            text = s3.run_trial(model, tok, layers, a.layers[0], vecs[c][a.layers[0]], 0,
                                s3.STEER, max_new=a.max_new, span="all")
            write({"stage": "steer", "layer": -1, "alpha": 0.0, "i": i, "concept": c,
                   "arm": "real", "text": text})
        for layer in a.layers:
            for alpha in a.live_alphas:
                if not k("steer", layer, alpha, None, i, "real"):
                    text = s3.run_trial(model, tok, layers, layer, vecs[c][layer], alpha,
                                        s3.STEER, max_new=a.max_new, span="all")
                    write({"stage": "steer", "layer": layer, "alpha": alpha, "i": i,
                           "concept": c, "arm": "real", "text": text})
    print("  steering gate done", flush=True)
    fout.close()


if __name__ == "__main__":
    main()
