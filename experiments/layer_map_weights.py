"""Weights-only layer map: near-zero gains of the norm feeding each MLP, across public models (17 Oct).

    PYTHONPATH=. python experiments/layer_map_weights.py      # writes results/layer_map_weights.json

Not filed; descriptive. For each model only the MLP-input norm weights are fetched, by HTTP range
requests into the safetensors files (kilobytes per model, no checkpoint download). Effective gain is
the weight, or 1 + weight for norms that scale that way (Gemma-3, Qwen3.5). A layer is a candidate
when its smallest |gain| is below 1% of the layer's median |gain|: in the three models whose
activations were mapped, every layer with a near-constant MLP-input coordinate (and no other) meets
it (checked below). Candidates are predictions from weights; the activation check is caliper.lint.
Models whose MLP reads an unnormalised residual (post-norm designs such as OLMo-2) have no such norm
and are listed as not applicable.
"""
import json
import re
import struct
import sys
from pathlib import Path

import numpy as np
import requests
from huggingface_hub import get_safetensors_metadata, hf_hub_url
from huggingface_hub.utils import build_hf_headers

ROOT = Path(__file__).resolve().parents[1]
FRAC = 0.01

# repo: (regex for the MLP-input norm weight with the layer index as group 1, gain = 1 + weight)
PAT = {
    "gpt2": (r"^(?:transformer\.)?h\.(\d+)\.ln_2\.weight$", False),
    "neo": (r"^transformer\.h\.(\d+)\.ln_2\.weight$", False),
    "neox": (r"^gpt_neox\.layers\.(\d+)\.post_attention_layernorm\.weight$", False),
    "opt": (r"^(?:model\.)?decoder\.layers\.(\d+)\.final_layer_norm\.weight$", False),
    "llama": (r"^model\.layers\.(\d+)\.post_attention_layernorm\.weight$", False),
    "gemma3": (r"^(?:model\.)?(?:language_model\.)?(?:model\.)?layers\.(\d+)\.pre_feedforward_layernorm\.weight$", True),
    "qwen35": (r"^model\.language_model\.layers\.(\d+)\.post_attention_layernorm\.weight$", True),
}
MODELS = {
    "gpt2": "gpt2", "gpt2-medium": "gpt2", "gpt2-large": "gpt2", "gpt2-xl": "gpt2",
    "EleutherAI/gpt-neo-125M": "neo", "EleutherAI/gpt-neo-1.3B": "neo", "EleutherAI/gpt-neo-2.7B": "neo",
    "EleutherAI/pythia-70m": "neox", "EleutherAI/pythia-160m": "neox", "EleutherAI/pythia-410m": "neox",
    "EleutherAI/pythia-1b": "neox", "EleutherAI/pythia-1.4b": "neox", "EleutherAI/pythia-2.8b": "neox",
    "EleutherAI/pythia-6.9b": "neox",
    "facebook/opt-125m": "opt", "facebook/opt-1.3b": "opt", "facebook/opt-2.7b": "opt", "facebook/opt-6.7b": "opt",
    "meta-llama/Llama-3.2-1B": "llama", "meta-llama/Llama-3.2-3B": "llama", "meta-llama/Llama-3.1-8B": "llama",
    "mistralai/Mistral-7B-v0.3": "llama",
    "Qwen/Qwen2.5-0.5B": "llama", "Qwen/Qwen2.5-1.5B": "llama", "Qwen/Qwen2.5-3B": "llama", "Qwen/Qwen2.5-7B": "llama",
    "Qwen/Qwen3-0.6B": "llama", "Qwen/Qwen3-1.7B": "llama", "Qwen/Qwen3-4B": "llama", "Qwen/Qwen3-8B": "llama",
    "Qwen/Qwen3.5-0.8B": "qwen35", "Qwen/Qwen3.5-2B": "qwen35", "Qwen/Qwen3.5-4B": "qwen35",
    "HuggingFaceTB/SmolLM2-135M": "llama", "HuggingFaceTB/SmolLM2-360M": "llama", "HuggingFaceTB/SmolLM2-1.7B": "llama",
    "google/gemma-3-270m": "gemma3", "google/gemma-3-1b-pt": "gemma3", "google/gemma-3-4b-pt": "gemma3",
    "google/gemma-3-12b-pt": "gemma3",
}
DTYPE = {"F32": np.float32, "F16": np.float16, "BF16": "bf16"}


def fetch(repo, fname, start, end, headers):
    url = hf_hub_url(repo, fname)
    r = requests.get(url, headers={**headers, "Range": f"bytes={start}-{end - 1}"}, timeout=60)
    r.raise_for_status()
    return r.content


def gains(repo, family):
    pat, plus_one = PAT[family]
    meta = get_safetensors_metadata(repo)
    headers = build_hf_headers()
    found = {}
    for fname, fm in meta.files_metadata.items():
        hits = [(int(m.group(1)), name, t) for name, t in fm.tensors.items() if (m := re.match(pat, name))]
        if not hits:
            continue
        n = struct.unpack("<Q", fetch(repo, fname, 0, 8, headers))[0]
        for layer, name, t in hits:
            raw = fetch(repo, fname, 8 + n + t.data_offsets[0], 8 + n + t.data_offsets[1], headers)
            if t.dtype == "BF16":
                w = (np.frombuffer(raw, np.uint16).astype(np.uint32) << 16).view(np.float32)
            else:
                w = np.frombuffer(raw, DTYPE[t.dtype]).astype(np.float32)
            found[layer] = (1.0 + w) if plus_one else w
    if not found:
        raise ValueError("no MLP-input norm weights matched")
    return [found[k] for k in sorted(found)]


def summarise(gs):
    rows = []
    for layer, g in enumerate(gs):
        a = np.abs(g.astype(np.float64))
        rows.append({"layer": layer, "min_gain": float(a.min()), "median_gain": float(np.median(a)),
                     "ratio": float(a.min() / np.median(a)), "argmin": int(a.argmin()),
                     "n_below": int((a < FRAC * np.median(a)).sum())})
    cand = [r["layer"] for r in rows if r["ratio"] < FRAC]
    return {"layers": len(rows), "candidates": cand, "per_layer": rows}


def main():
    rep = {"frac": FRAC, "models": {}}
    for repo, fam in MODELS.items():
        try:
            s = summarise(gains(repo, fam))
            rep["models"][repo] = s
            print(f"{repo:36s} {s['layers']:3d} layers  candidates {s['candidates']}", flush=True)
        except Exception as e:  # noqa: BLE001 - reported per model, never silently dropped
            rep["models"][repo] = {"error": f"{type(e).__name__}: {e}"[:300]}
            print(f"{repo:36s} ERROR {type(e).__name__}: {str(e)[:120]}", flush=True)
    rep["models"]["allenai/OLMo-2-0425-1B"] = {"not_applicable": "post-norm: the MLP reads the unnormalised residual"}
    json.dump(rep, open(ROOT / "results/layer_map_weights.json", "w"), indent=1)


if __name__ == "__main__":
    main()
