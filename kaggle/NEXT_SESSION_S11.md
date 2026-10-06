# Kaggle session: S-11, re-running APERTURE's naturalistic arm (R9)

Prereg: `docs/preregistration-s11-naturalistic.md`. It uses APERTURE's own code, installed
from GitHub at commit `b5bb2fb`, exactly as the S-3 (F1) notebook does. About 30 minutes on
one T4.

## Before you start

- **Model:** Gemma 2, `google/gemma-2-2b-it` (Transformers).
- **Secrets:** `HF_TOKEN` attached.
- **Session options:** GPU T4 (x1 is enough), Internet On.

Every block below is one notebook **code cell**, pasted as is. A `%%bash` first line
makes the cell run as shell. Without it, Kaggle runs the lines as Python and fails
with `SyntaxError`.

## Cell 1

```
%%bash
pip install -q -U bitsandbytes accelerate transformers wordfreq
pip install -q --no-cache-dir --force-reinstall --no-deps https://github.com/santoshcheethiralame-dot/APERTURE/archive/b5bb2fb4554de04ebe80184e156b151dde9f9662.zip
```

## Cell 2: data files from the same commit

```
%%bash
rm -rf /kaggle/working/aperture
git clone -q https://github.com/santoshcheethiralame-dot/APERTURE.git /kaggle/working/aperture
cd /kaggle/working/aperture && git checkout -q b5bb2fb && ls data/concepts
```

## Cell 3: R9, as designed, then a second extraction seed

```python
import glob, json, os, torch, importlib.metadata as md
os.chdir("/kaggle/working/aperture")
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from aperture.concepts import load_bank
from aperture.naturalistic import load_contexts, collect_naturalistic_hf
import aperture.hf_model as hfm

path = [p for p in glob.glob("/kaggle/input/**/config.json", recursive=True)
        if "gemma-2" in p and "2b-it" in p]
assert len(path) == 1, path
mdir = os.path.dirname(path[0])
tok = AutoTokenizer.from_pretrained(mdir)
model = AutoModelForCausalLM.from_pretrained(
    mdir, quantization_config=BitsAndBytesConfig(load_in_8bit=True), device_map="auto")
model.eval()

DISTRACTOR = "\n\nQ: What is the capital of France?\nA: Paris.\n\n"
REPORT = "Q: Setting that aside, in one word, what concept was most on your mind just now?\nA:"
bank = load_bank("data/concepts/dev_bank.yaml")
contexts = load_contexts("data/concepts/contexts.yaml")
versions = {p: md.version(p) for p in ("torch", "transformers", "bitsandbytes", "accelerate")}
json.dump({"commit": "b5bb2fb", "layer": 13, "n_pairs": 12, "versions": versions},
          open("/kaggle/working/s11_config.json", "w"), indent=2)

# Primary: R9 unchanged (extraction seed 0, the default).
res = collect_naturalistic_hf(model, tok, bank, contexts, DISTRACTOR, REPORT, layer=13,
                              n_pairs=12, max_new_tokens=12,
                              out="/kaggle/working/s11_primary.jsonl")
k = sum(r["predicted"] == r["concept"] for r in res["records"])
print(f"PRIMARY identifiability {k}/16 (criterion: >= 4)")

# Secondary: pairs seed 1. extract_hf takes a seed; collect_naturalistic_hf does not pass
# one, so wrap it for this call only.
orig = hfm.extract_hf
import aperture.naturalistic as nat
nat.extract_hf = lambda *a, **kw: orig(*a, **{**kw, "seed": 1})
res1 = collect_naturalistic_hf(model, tok, bank, contexts, DISTRACTOR, REPORT, layer=13,
                               n_pairs=12, max_new_tokens=12,
                               out="/kaggle/working/s11_seed1.jsonl")
nat.extract_hf = orig
print("SEED 1 identifiability", sum(r["predicted"] == r["concept"] for r in res1["records"]), "/16")
```

## Cell 4: save the raw activations for the uncentred analysis

```python
from aperture.naturalistic import last_activation_hf
acts = {n: last_activation_hf(model, tok, p, 13).float().cpu() for n, p in contexts.items()}
dirs = {n: hfm.extract_hf(model, tok, bank, bank.get(n), 13, 12).direction.float().cpu()
        for n in contexts}
torch.save({"acts": acts, "dirs": dirs}, "/kaggle/working/s11_activations.pt")
import shutil; shutil.make_archive("/kaggle/working/s11", "zip", "/kaggle/working",
                                   base_dir=".")
print("DOWNLOAD /kaggle/working/s11.zip")
```

Download `s11.zip` and tell me. I score it once against the prereg and register it as S-11 /
A-R9b in the notebook.
