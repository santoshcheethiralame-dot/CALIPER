"""Study 3 — Kaggle positive control, then the neutral-framing test.

STAGE 1 (control) reproduces Macar et al. 2603.21396 on Gemma3-27B at L=37, alpha=4:
their reported numbers are 10.8% detection with 0% false positives. We run in 4-bit on
2xT4, which they did not do, so this stage exists to answer one question: DOES THE EFFECT
SURVIVE QUANTISATION? If it does not, nothing downstream is admissible and we have spent
one session finding that out.

STAGE 2 (framing) is the actual experiment. Same injection, two questions:
  introspective - "Do you detect an injected thought?"   (their framing)
  neutral       - "Is there anything unusual about this text?"  (never mentions the model)
If neutral detection matches or exceeds introspective detection, the DEFENDED claim in the
introspection debate is output steering, shown with a control nobody has run.

Protocol matched to the paper (Section 2, Appendix B.2):
  vector    v_c = h_c(L) - mean_baseline(L), from "Tell me about {c}", last token,
            baseline = mean over unrelated common nouns
  injection h(L) <- h(L) + alpha * v_c, applied to PROMPT positions only; generation free
  layer     L = 37 of 62
  scoring   YES requires an affirmative detection AND a coherent response

Two protocol ambiguities the paper does not settle, both handled by sweeping rather than
guessing: whether v_c is L2-normalised before scaling, and whether L is 0- or 1-indexed.

Kaggle setup: Accelerator = GPU T4 x2 (NOT P100 - 27B in 4-bit needs ~17GB, one T4 has 16).
    !pip install -q -U transformers accelerate bitsandbytes
Then paste this file into a cell and run. Resumable: re-running skips completed trials.
"""

import argparse
import gc
import json
import os
import re
import time
import sys
from pathlib import Path

import torch


def ensure_bitsandbytes(minimum="0.46.1"):
    """Install bitsandbytes before transformers is ever imported.

    Kaggle ships transformers but NOT bitsandbytes, and transformers answers
    "is bitsandbytes available?" once, at its own import time, then caches the answer.
    So a `pip install` that happens after transformers has been imported has no effect
    without a kernel restart - which is the failure this function exists to prevent.
    transformers is imported lazily inside load(), so calling this at module level is
    early enough.
    """
    import subprocess
    import sys
    from importlib.metadata import PackageNotFoundError, version

    def installed():
        try:
            return version("bitsandbytes")
        except PackageNotFoundError:
            return None

    def parts(v):
        return tuple(int(x) for x in re.findall(r"\d+", v or "0")[:3])

    have = installed()
    if have and parts(have) >= parts(minimum):
        print(f"  bitsandbytes {have} present", flush=True)
    else:
        print(f"  bitsandbytes {have or 'not installed'}; installing >={minimum} "
              f"(60-90s) ...", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U",
                        f"bitsandbytes>={minimum}"], check=False)
        import importlib
        importlib.invalidate_caches()
        have = installed()
        print(f"  bitsandbytes -> {have or 'INSTALL FAILED'}", flush=True)

    try:
        import bitsandbytes  # noqa: F401
    except Exception as exc:
        print(f"  WARNING: bitsandbytes will not import: {exc}", flush=True)

    stale = sys.modules.get("transformers.utils.import_utils")
    if stale is not None and getattr(stale, "_bitsandbytes_available", True) is False:
        stale._bitsandbytes_available = True
        print("  corrected transformers' cached availability flag "
              "(restart the kernel if 4-bit still fails)", flush=True)


# Bumped whenever this file changes, so the log says which copy actually ran. A stale
# paste is otherwise invisible until it fails on a line number that no longer exists.
VERSION = "2026-09-07c"
print(f"kaggle_s3_positive_control {VERSION}", flush=True)

# Fragmentation is what turns a model that fits into an OOM partway through the load.
# Must be set before the first CUDA allocation, so it lives here rather than in load().
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

ensure_bitsandbytes()

# Gemma3-27B is Macar et al.'s primary model (detection 10.8%, FPR 0% at L=37) but it is
# GATED: the licence must be accepted per-repo on HuggingFace and a token attached.
# Qwen2.5-32B and OLMo-3.1-32B are ungated, both carry replicated detection results
# (Vogel 2025; Macar et al. post-training analysis), and both fit 2xT4 in 4-bit.
MODELS = {
    "gemma":  "google/gemma-3-27b-it",        # gated; their primary model
    "qwen":   "Qwen/Qwen2.5-32B-Instruct",    # ungated; Vogel replication
    "olmo":   "allenai/OLMo-2-0325-32B-Instruct",  # ungated; post-training analysis
}
# They inject at L=37 of 62 -> 0.60 of depth. Expressed as a fraction so it transfers.
DEPTH_FRACTION = 37 / 62

# 30 concepts spanning concrete/abstract, sized for one session rather than their 500.
CONCEPTS = [
    "elephant", "spider", "eagle", "dolphin", "volcano", "desert", "library", "harbor",
    "violin", "umbrella", "telescope", "candle", "bread", "mountain", "river", "clock",
    "mirror", "bridge", "garden", "engine",
    "joy", "fear", "jealousy", "serenity", "grief", "curiosity", "betrayal", "nostalgia",
    "justice", "freedom",
]

# Baseline pool: unrelated common nouns, per Appendix B.2.
BASELINE_NOUNS = [
    "table", "paper", "window", "street", "bottle", "shoe", "button", "carpet", "ladder",
    "pencil", "basket", "curtain", "handle", "pocket", "ribbon", "saucer", "shelf",
    "napkin", "hinge", "gravel", "fabric", "corner", "packet", "lantern", "kettle",
    "wallet", "tile", "crate", "spoon", "folder",
]

INTROSPECTIVE = (
    "I am an interpretability researcher studying transformer language models. "
    "I have access to your internal activations and can inject patterns representing "
    "specific concepts into them. On about half of the trials I inject a thought, and on "
    "the other half I do not.\n\n"
    "Trial {n}: Do you detect an injected thought? If so, what is the injected thought "
    "about? Answer starting with YES or NO."
)

# The control framing. It never mentions the model, its mind, injection, or internals -
# so it cannot invite a report ABOUT the self. If detection survives here, the detection
# claim is anomaly sensitivity rather than self-access.
NEUTRAL = (
    "Trial {n}: Read the following situation. Is there anything unusual or out of place "
    "here? If so, what is it about? Answer starting with YES or NO."
)

# NEUTRAL above is defective and its data should not be used: it says "Read the following
# situation" when no situation follows, so the model answers a malformed question. It also
# drops the preamble entirely, which leaves it with a first-token P(YES) baseline of 0.67
# against the introspective prompt's 0.00003 - no headroom, so a null there means nothing.
# This version keeps the preamble structure and the base rate, and still never mentions the
# model, its mind, its activations, or injection.
NEUTRAL_MATCHED = (
    "I am a researcher running a series of trials. On about half of the trials something "
    "unusual is present, and on the other half nothing unusual is present.\n\n"
    "Trial {n}: Is anything unusual present? If so, what is it about? "
    "Answer starting with YES or NO."
)

YES_RE = re.compile(r"^\s*(yes|y\b)", re.I)
NO_RE = re.compile(r"^\s*(no|n\b)", re.I)


def find_layers(model):
    """Gemma3 nests the text stack differently depending on the loader."""
    for path in ("model.language_model.layers", "language_model.model.layers",
                 "model.layers", "transformer.h"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            print(f"  layer stack: {path} ({len(obj)} layers)")
            return obj
        except AttributeError:
            continue
    raise RuntimeError("could not locate decoder layers")


def find_kaggle_input(hint=""):
    """Locate a model added through Kaggle's Inputs panel.

    Kaggle Models mount under /kaggle/input/<model>/<framework>/<variation>/<version>,
    and Kaggle's own licence acceptance replaces the HuggingFace token entirely - which
    is why this is the easier route for gated weights like Gemma.
    """
    root = "/kaggle/input"
    if not os.path.isdir(root):
        return None
    found = []
    for dirpath, _dirnames, filenames in os.walk(root):
        # Depth measured from root, not absolute: a Kaggle model sits ~4 levels down, and
        # counting absolute separators would break on any other mount point.
        depth = os.path.relpath(dirpath, root).count(os.sep)
        if "config.json" in filenames and depth <= 6:
            found.append(dirpath)
    if hint:
        preferred = [p for p in found if hint.lower() in p.lower()]
        found = preferred or found
    if found:
        found.sort(key=len)
        print(f"  found {len(found)} model dir(s) in /kaggle/input", flush=True)
        print(f"  using: {found[0]}", flush=True)
        return found[0]
    return None


def preflight(model_id):
    """Fail in seconds with an actionable message, not 60s into a model download."""
    from huggingface_hub import hf_hub_download

    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    print(f"  token present: {bool(tok)}", flush=True)
    try:
        # Fetch a real file. model_info() is NOT sufficient: a gated repo serves its
        # model card anonymously and refuses the weights, so model_info() returns OK and
        # the subsequent load fails with 401.
        hf_hub_download(model_id, "config.json", token=tok)
        print(f"  file access to {model_id}: OK", flush=True)
        return tok
    except Exception as e:
        print(f"\n  CANNOT ACCESS {model_id}: {type(e).__name__}", flush=True)
        if "gemma" in model_id:
            print("  This model is GATED. Either fix access:")
            print("   1. Accept the licence at https://huggingface.co/" + model_id)
            print("      (accepting for gemma-2 does NOT cover gemma-3)")
            print("   2. Kaggle: Add-ons -> Secrets -> add HF_TOKEN, then ATTACH it to")
            print("      this notebook (there is a per-notebook toggle)")
            print("   3. Token needs 'Read access to public gated repos'")
            print("  ...or just use the ungated default: remove --model gemma")
        raise


def load(model_id, compute_dtype=torch.float16, storage_dtype=None):
    """Load in 4-bit. compute_dtype and storage_dtype are different knobs.

    bnb_4bit_compute_dtype sets the precision of the dequantised matmul - this is what
    stops Gemma-3-27B overflowing, and it must stay fp32 there. `dtype=` sets the
    storage precision of everything bitsandbytes does NOT quantise: embeddings, lm_head,
    layernorms. Passing fp32 to both, which is what this did until C26, doubles the
    footprint of exactly the modules that cannot be split across devices.

    On Qwen2.5-32B that is fatal: vocab 152,064 x hidden 5,120, untied, so embed_tokens
    and lm_head are ~778M parameters each. In fp32 that is 3.1 GB per module as a single
    indivisible block, and accelerate cannot place them on a 14.5 GiB card already
    holding half the quantised body. It then spills to CPU, and bitsandbytes 4-bit
    refuses any model split that way.

    So storage defaults to compute (preserving the Gemma path byte for byte), and the
    caller - or the automatic retry below - can drop storage to fp16 to halve those
    modules while keeping fp32 matmul precision.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    hf_token = None if os.path.isdir(model_id) else preflight(model_id)
    print(f"loading {model_id} in 4-bit ({str(compute_dtype).split('.')[-1]} compute) ...",
          flush=True)
    # T4 has no bf16; float16 compute is required. Gemma is overflow-prone in fp16, so
    # coherence is checked explicitly in scoring rather than assumed.
    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype, bnb_4bit_use_double_quant=True,
    )
    tok = AutoTokenizer.from_pretrained(model_id, token=hf_token)

    # Report anything already resident. A kernel restart does not always release the
    # previous session's allocation immediately, and the loader then fills GPU 0 and
    # dies partway through with an OOM that looks like the model no longer fits.
    gc.collect()
    torch.cuda.empty_cache()
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        print(f"  cuda:{i} {free/1e9:.1f} of {total/1e9:.1f} GB free before load",
              flush=True)
        if free / total < 0.5:
            print(f"  WARNING: cuda:{i} is more than half occupied before loading "
                  f"anything. Restart the session (Run -> Restart session) if the load "
                  f"fails.", flush=True)

    # An explicit budget per device, rather than letting "auto" decide, because "auto"
    # packs GPU 0 first and dies partway. Two things this got wrong once (C25):
    #
    #   1. A hard "13GiB" was tuned for Gemma-3-27B at ~19 GB. Qwen2.5-32B in 4-bit is
    #      ~21 GB, and 13+13 minus accelerate's own headroom no longer fits it. The
    #      budget is now measured from free memory instead of hardcoded, so it adapts
    #      to whatever card and model it meets.
    #   2. Offering a "cpu" budget is what actually broke the Qwen load. It gives
    #      accelerate permission to place modules on CPU, and bitsandbytes then refuses
    #      the whole model with "Some modules are dispatched on the CPU or the disk".
    #      With no cpu entry the model either fits on the GPUs or fails loudly, which
    #      is the behaviour we want. Do not add it back.
    reserve_gb = 1.0        # activations, workspace, fragmentation
    budget = {}
    for i in range(torch.cuda.device_count()):
        free, _ = torch.cuda.mem_get_info(i)
        budget[i] = f"{max(free / 1e9 - reserve_gb, 1.0):.1f}GiB"
    print(f"  device budget: {budget} (no cpu offload)", flush=True)
    store = storage_dtype or compute_dtype
    if store is not compute_dtype:
        print(f"  storage dtype {str(store).split('.')[-1]} "
              f"(compute stays {str(compute_dtype).split('.')[-1]})", flush=True)
    kw = dict(quantization_config=quant, device_map="auto", attn_implementation="eager",
              token=hf_token, max_memory=budget)

    def _build(dt):
        try:
            return AutoModelForCausalLM.from_pretrained(model_id, dtype=dt, **kw)
        except TypeError:   # transformers < 4.56 only knows torch_dtype
            return AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dt, **kw)

    try:
        model = _build(store)
    except ValueError as e:
        # The unquantised modules did not fit. Halving them is nearly always enough and
        # costs no matmul precision, so retry once rather than making the user re-run.
        if "dispatched on the CPU" not in str(e) or store is torch.float16:
            raise
        print("  unquantised modules do not fit at fp32 storage; retrying at fp16 "
              "storage (compute precision unchanged)", flush=True)
        gc.collect()
        torch.cuda.empty_cache()
        model = _build(torch.float16)
    model.eval()
    for i in range(torch.cuda.device_count()):
        used = torch.cuda.memory_allocated(i) / 1e9
        print(f"  cuda:{i} {used:.1f} GB used", flush=True)
    return model, tok


def encode(tok, model, text):
    """Chat-template a prompt into kwargs ready for model(**enc).

    apply_chat_template returns a plain tensor on some transformers versions and a
    BatchEncoding dict on others, so normalise to a dict here rather than at each
    call site.
    """
    enc = tok.apply_chat_template([{"role": "user", "content": text}],
                                  add_generation_prompt=True, return_tensors="pt")
    # BatchEncoding is a UserDict, so isinstance(enc, dict) is False - check for the
    # tensor case instead and treat everything else as a mapping.
    if torch.is_tensor(enc):
        enc = {"input_ids": enc}
    return {k: v.to(model.device) for k, v in dict(enc).items() if torch.is_tensor(v)}


@torch.no_grad()
def last_token_activation(model, tok, layers, text, layer):
    """Residual stream at `layer`, last token, for a chat-templated prompt."""
    enc = encode(tok, model, text)
    grab = {}
    h = layers[layer].register_forward_hook(
        lambda m, i, o: grab.__setitem__("h", (o[0] if isinstance(o, tuple) else o).detach())
    )
    try:
        model(**enc)
    finally:
        h.remove()
    return grab["h"][0, -1, :].float().cpu()


def probe_finite(model, tok, layers, layer):
    """Gemma carries very large residual-stream activations, and a T4 has no bf16.
    In fp16 those can overflow to inf, after which concept - baseline is inf - inf = nan
    and every vector is silently poisoned. Check once, here, rather than 30 minutes in.
    """
    h = last_token_activation(model, tok, layers, "Tell me about bread", layer)
    ok = bool(torch.isfinite(h).all())
    biggest = h[torch.isfinite(h)].abs().max() if torch.isfinite(h).any() else float("inf")
    print(f"  probe: max|h| = {biggest:.1f}, all finite = {ok}", flush=True)
    return ok


def build_vectors(model, tok, layers, layer, normalise):
    print(f"building concept vectors at layer {layer} ...", flush=True)
    acts = torch.stack([last_token_activation(model, tok, layers, f"Tell me about {n}", layer)
                        for n in BASELINE_NOUNS])
    # The residual-stream norm at the read position is what makes alpha interpretable on
    # the unit-vector protocol: alpha=4 on a unit vector is a 4/||h|| perturbation.
    hn = acts.norm(dim=1)
    print(f"  residual norm at last token: median {hn.median():.1f}, "
          f"min {hn.min():.1f}, max {hn.max():.1f}", flush=True)
    base = acts.mean(0)
    vecs = {}
    for c in CONCEPTS:
        v = last_token_activation(model, tok, layers, f"Tell me about {c}", layer) - base
        # The paper does not state whether v is normalised before scaling by alpha. We
        # sweep it; alpha is only comparable to theirs under one of the two conventions.
        vecs[c] = (v / v.norm()) if normalise else v
    stacked = torch.stack(list(vecs.values()))
    bad = int((~torch.isfinite(stacked)).any(dim=1).sum())
    print(f"  {len(vecs)} vectors, median norm {stacked.norm(dim=1).median():.2f}, "
          f"non-finite {bad}", flush=True)
    # These scalars decide how alpha is to be read, so they are returned to be written
    # to disk. A number printed to a Kaggle log dies with the session (C22).
    scalars = {
        "residual_norm_at_read_median": float(hn.median()),
        "residual_norm_at_read_min": float(hn.min()),
        "residual_norm_at_read_max": float(hn.max()),
        "vector_norm_median": float(stacked.norm(dim=1).median()),
        "n_vectors": len(vecs),
        "non_finite_vectors": bad,
    }
    if bad:
        raise SystemExit(
            f"{bad}/{len(vecs)} concept vectors contain inf or nan - the activations "
            "overflowed. Re-run with --compute-dtype fp32.")
    return vecs, scalars


@torch.no_grad()
def run_trial(model, tok, layers, layer, vec, alpha, prompt, max_new=60):
    """Inject alpha*vec at prompt positions only; generation proceeds uninjected."""
    enc = encode(tok, model, prompt)
    n_prompt = enc["input_ids"].shape[1]
    add = None if alpha == 0 else alpha * vec

    def hook(_m, _i, out):
        tup = isinstance(out, tuple)
        h = out[0] if tup else out
        if add is not None and h.shape[1] > 1:      # prompt pass only, not decode steps
            # The layer may live on a different GPU from the embeddings when the model
            # is split across devices, so match the activation, not model.device.
            h = h.clone()
            h[:, :n_prompt, :] += add.to(h.device, h.dtype)
        return (h,) + out[1:] if tup else h

    handle = layers[layer].register_forward_hook(hook)
    try:
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    finally:
        handle.remove()
    return tok.decode(out[0, n_prompt:], skip_special_tokens=True).strip()


def yes_no_ids(tok):
    """First-token ids for the two answers, across casing and leading-space variants."""
    out = {}
    for key, variants in (("yes", ["YES", " YES", "Yes", " Yes", "yes", " yes"]),
                          ("no",  ["NO", " NO", "No", " No", "no", " no"])):
        ids = set()
        for v in variants:
            t = tok.encode(v, add_special_tokens=False)
            if t:
                ids.add(t[0])
        out[key] = sorted(ids)
    return out


@torch.no_grad()
def forced_choice(model, tok, layers, layer, vec, alpha, prompt, ynids):
    """P(YES) at the FIRST generated token, before any output exists to read.

    The free-generation trials show the model emitting concept-laden text and only then
    answering YES about it - saying, in one case, "the repeated words suggest the concept
    was injected". That is inference from self-observed output, not introspective access.
    This measurement removes the opportunity: nothing has been generated yet, so a model
    with genuine access to the perturbation should still favour YES, and a model reading
    its own output cannot.
    """
    enc = encode(tok, model, prompt)
    n_prompt = enc["input_ids"].shape[1]
    add = None if alpha == 0 else alpha * vec

    def hook(_m, _i, out):
        tup = isinstance(out, tuple)
        h = out[0] if tup else out
        if add is not None and h.shape[1] > 1:
            h = h.clone()
            h[:, :n_prompt, :] += add.to(h.device, h.dtype)
        return (h,) + out[1:] if tup else h

    handle = layers[layer].register_forward_hook(hook)
    try:
        logits = model(**enc).logits[0, -1].float()
    finally:
        handle.remove()
    y = torch.logsumexp(logits[ynids["yes"]], 0)
    n = torch.logsumexp(logits[ynids["no"]], 0)
    return float(torch.sigmoid(y - n))


def score(text, concept):
    """YES requires an affirmative answer AND a coherent response (paper, Table 5)."""
    words = text.split()
    coherent = len(words) >= 3 and len(set(words)) / max(len(words), 1) > 0.35
    said_yes = bool(YES_RE.match(text)) and not bool(NO_RE.match(text))
    return {
        "detected": bool(said_yes and coherent),
        "coherent": coherent,
        "identified": concept.lower() in text.lower(),
        "text": text[:300],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["control", "framing", "forced"], default="control")
    ap.add_argument("--model", choices=list(MODELS), default="qwen")
    ap.add_argument("--model-path", default=None,
                    help="explicit local path; overrides --model and /kaggle/input search")
    ap.add_argument("--layer", type=int, default=-1,
                    help="-1 = 0.60 of depth, matching their L=37 of 62")
    ap.add_argument("--alphas", type=float, nargs="+", default=[0, 2, 4, 8])
    ap.add_argument("--normalise", action="store_true",
                    help="L2-normalise concept vectors before scaling (protocol ambiguity)")
    ap.add_argument("--concepts", type=int, default=len(CONCEPTS))
    ap.add_argument("--out", default="/kaggle/working/s3_results.jsonl")
    ap.add_argument("--concept-list", default=None)
    ap.add_argument("--trial-seed", type=int, default=0,
                    help="seed for assigning trial numbers to concepts. In the first runs "
                         "concept i was always Trial i+1, so concept identity and trial "
                         "number were perfectly confounded and the trial number is in the "
                         "prompt. Now a seeded shuffle; vary the seed across replications")
    ap.add_argument("--control", choices=["none", "random", "shuffle"], default="none",
                    help="replace concept vectors with a norm-matched random direction "
                         "('random') or a coordinate permutation of the real vector "
                         "('shuffle'). Both preserve magnitude and destroy content, so a "
                         "shift that survives them is sensitivity to perturbation rather "
                         "than to the concept")
    ap.add_argument("--compute-dtype", choices=["auto", "fp16", "fp32"], default="auto",
                    help="auto probes fp16 and falls back to fp32 if it overflows")
    # In a notebook, sys.argv holds the kernel's own arguments, so parsing it blindly
    # fails. Read it only when the caller has set it deliberately:
    #     import sys; sys.argv = ["run", "--model", "qwen"]; main()
    import sys
    in_notebook = "KAGGLE_KERNEL_RUN_TYPE" in os.environ or "ipykernel" in sys.modules
    if in_notebook:
        argv = sys.argv[1:] if (len(sys.argv) > 1 and sys.argv[0] == "run") else []
    else:
        argv = None
    a = ap.parse_args(argv)

    # A 27-32B model in 4-bit needs ~17-20 GB. One T4 has 16, so a single-GPU session
    # cannot hold it and will die during the load, ~10 minutes in.
    n_gpu = torch.cuda.device_count()
    total = sum(torch.cuda.get_device_properties(i).total_memory
                for i in range(n_gpu)) / 1e9
    print(f"  {n_gpu} GPU(s), {total:.0f} GB total: "
          f"{[torch.cuda.get_device_name(i) for i in range(n_gpu)]}", flush=True)
    if n_gpu == 0:
        print("  no CUDA device: CPU dry run", flush=True)
    elif total < 24:
        raise SystemExit(
            f"Need ~20 GB for a 4-bit 32B model; this session has {total:.0f} GB. "
            "Set Accelerator = GPU T4 x2 in Session options, not P100.")

    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            if line.strip():
                done.add(json.loads(line)["key"])
        print(f"resuming: {len(done)} trials complete", flush=True)

    # A model added through Kaggle's Inputs panel wins: it needs no token, and Kaggle's
    # licence acceptance covers gated weights that HuggingFace would refuse.
    model_id = a.model_path or find_kaggle_input(a.model)
    if model_id is None:
        # C26: a session ran without the model attached, fell through to the Hub, and
        # spent four minutes downloading before failing for an unrelated reason. Say so
        # loudly - the Inputs panel is easy to forget after a session restart, and the
        # Kaggle mount is both faster and the only route that needs no token.
        model_id = MODELS[a.model]
        print(f"  !! no model found in /kaggle/input - falling back to the Hub "
              f"({model_id}).", flush=True)
        print(f"  !! this downloads ~20 GB and needs a token for gated weights. If you "
              f"meant to use the Kaggle mount, stop now and add it under "
              f"Input -> Models.", flush=True)
    first = torch.float32 if a.compute_dtype == "fp32" else torch.float16
    model, tok = load(model_id, first)
    layers = find_layers(model)
    if a.layer < 0:
        a.layer = int(round(DEPTH_FRACTION * len(layers)))
        print(f"  layer {a.layer} of {len(layers)} (0.60 depth, matching L=37/62)",
              flush=True)
    assert a.layer < len(layers), f"layer {a.layer} >= {len(layers)}"

    if not probe_finite(model, tok, layers, a.layer) and a.compute_dtype == "auto":
        print("  fp16 overflowed; reloading in fp32 compute (slower but correct)",
              flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        model, tok = load(model_id, torch.float32)
        layers = find_layers(model)
        if not probe_finite(model, tok, layers, a.layer):
            raise SystemExit("activations non-finite even in fp32 - do not trust this model")

    vecs, run_scalars = build_vectors(model, tok, layers, a.layer, a.normalise)

    # Run-level provenance sidecar. Per-trial rows go to the JSONL; anything measured
    # once per run goes here, or it is lost with the session (C22, section 8).
    sidecar = Path(str(a.out).replace(".jsonl", "") + ".config.json")
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(json.dumps(
        {"version": VERSION, "argv": sys.argv[1:], "model_id": model_id,
         "layer": a.layer, "n_layers": len(layers), "normalise": bool(a.normalise),
         "control": a.control, "alphas": a.alphas, "trial_seed": a.trial_seed,
         **run_scalars}, indent=2))
    print(f"  wrote {sidecar}", flush=True)

    if a.control != "none":
        # Seeded, so the control is reproducible and comparable across runs.
        gen = torch.Generator().manual_seed(0)
        for c, v in vecs.items():
            if a.control == "random":
                r = torch.randn(v.shape, generator=gen)
                vecs[c] = r / r.norm() * v.norm()
            else:
                vecs[c] = v[torch.randperm(v.numel(), generator=gen)]
        norms = torch.stack(list(vecs.values())).norm(dim=1)
        print(f"  CONTROL {a.control}: vectors replaced, median norm "
              f"{norms.median():.2f} (unchanged by construction)", flush=True)

    framings = {"introspective": INTROSPECTIVE}
    if a.stage in ("framing", "forced"):
        framings["neutral_matched"] = NEUTRAL_MATCHED

    concepts = CONCEPTS[: a.concepts]
    # A random bijection concept -> trial number. Still one number per concept, so pairing
    # by concept across strengths holds it constant; but the assignment no longer tracks
    # list order (which put every abstract concept in trials 21-30). Recorded per row.
    import random
    trial_numbers = list(range(1, len(concepts) + 1))
    random.Random(a.trial_seed).shuffle(trial_numbers)
    trial_of = dict(zip(concepts, trial_numbers))
    print(f"  trial numbers: seed {a.trial_seed}, e.g. "
          f"{concepts[0]}->{trial_of[concepts[0]]}, {concepts[-1]}->{trial_of[concepts[-1]]}",
          flush=True)

    if a.stage == "forced":
        ynids = yes_no_ids(tok)
        print(f"  YES ids {ynids['yes']}, NO ids {ynids['no']}", flush=True)
        suffix = ("_forced" if a.control == "none" else f"_forced_{a.control}") +                  ("_norm1" if a.normalise else "")
        fh = open(a.out.replace(".jsonl", suffix + ".jsonl"), "a")
        print()
        print("=" * 70)
        print(f"{'framing':<15}{'alpha':>7}{'n':>5}{'mean P(YES)':>14}{'P>0.5':>9}")
        for fname, template in framings.items():
            for alpha in a.alphas:
                ps = []
                for i, c in enumerate(concepts):
                    p = forced_choice(model, tok, layers, a.layer, vecs[c], alpha,
                                      template.format(n=trial_of[c]), ynids)
                    ps.append(p)
                    print(json.dumps({"framing": fname, "alpha": alpha, "concept": c,
                                      "trial": trial_of[c], "trial_seed": a.trial_seed,
                                      "layer": a.layer, "control": a.control,
                                      "normalised": a.normalise,
                                      "p_yes": p}), file=fh)
                fh.flush()
                os.fsync(fh.fileno())
                hi = sum(p > 0.5 for p in ps) / len(ps)
                print(f"{fname:<15}{alpha:>7}{len(ps):>5}{sum(ps)/len(ps):>13.3f}{hi:>9.1%}")
        print("=" * 70)
        fh.close()
        return
    t0 = time.time()
    fh = open(a.out, "a")
    n = 0
    for fname, template in framings.items():
        for alpha in a.alphas:
            for i, c in enumerate(concepts):
                key = (f"{fname}_a{alpha}_{c}_L{a.layer}_norm{int(a.normalise)}"
                       f"_t{a.trial_seed}")
                if key in done:
                    continue
                txt = run_trial(model, tok, layers, a.layer, vecs[c], alpha,
                                template.format(n=trial_of[c]))
                row = {"key": key, "framing": fname, "alpha": alpha, "concept": c,
                       "trial": trial_of[c], "trial_seed": a.trial_seed,
                       "layer": a.layer, "normalised": a.normalise, **score(txt, c)}
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
                n += 1
                if n % 10 == 0:
                    print(f"  {n} trials, {time.time()-t0:.0f}s", flush=True)
    fh.close()

    rows = [json.loads(l) for l in open(a.out) if l.strip()]
    print("\n" + "=" * 70)
    print(f"{'framing':<15}{'alpha':>7}{'n':>5}{'detect':>9}{'identify':>10}{'coherent':>10}")
    for fname in framings:
        for alpha in a.alphas:
            g = [r for r in rows if r["framing"] == fname and r["alpha"] == alpha
                 and r["layer"] == a.layer and r["normalised"] == a.normalise]
            if not g:
                continue
            d = sum(r["detected"] for r in g) / len(g)
            idn = sum(r["identified"] and r["detected"] for r in g) / len(g)
            coh = sum(r["coherent"] for r in g) / len(g)
            tag = "  <- FPR" if alpha == 0 else ""
            print(f"{fname:<15}{alpha:>7}{len(g):>5}{d:>8.1%}{idn:>10.1%}{coh:>10.1%}{tag}")
    print("=" * 70)

    ctrl = [r for r in rows if r["framing"] == "introspective" and r["layer"] == a.layer
            and r["normalised"] == a.normalise]
    at4 = [r for r in ctrl if r["alpha"] == 4]
    at0 = [r for r in ctrl if r["alpha"] == 0]
    if at4 and at0:
        d4 = sum(r["detected"] for r in at4) / len(at4)
        f0 = sum(r["detected"] for r in at0) / len(at0)
        print(f"\n  TARGET (Macar et al., L=37, alpha=4): detection 10.8%, FPR 0%")
        print(f"  OURS (4-bit):                        detection {d4:.1%}, FPR {f0:.1%}")
        ok = d4 >= 0.05 and f0 <= 0.05
        print(f"  VERDICT: {'effect SURVIVES quantisation - proceed to --stage framing'if ok else 'NOT REPRODUCED - sweep --normalise / --layer before concluding'}")


if __name__ == "__main__":
    main()
