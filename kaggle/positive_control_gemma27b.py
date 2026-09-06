# %% [markdown]
# # Positive control — concept-injection detection, Gemma3-27B-it, 4-bit, 2×T4
#
# Reproduces Macar et al. (arXiv:2603.21396) layer-37, α=4 detection under Kaggle
# constraints. Pre-registered in `docs/preregistration-positive-control-gemma27b.md`.
#
# **Setup on Kaggle:** Accelerator = **GPU T4 x2**. Add a secret named `HF_TOKEN`
# (Add-ons → Secrets) with a Hugging Face token that has accepted the Gemma licence.
# Turn **Persistence** on for `/kaggle/working` so the run resumes after a 12h cut.

# %%
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "bitsandbytes>=0.43", "accelerate", "transformers>=4.50"], check=False)

import os, json, re, time, math, gc
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

try:
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
except Exception as e:
    print("no Kaggle secret found; set HF_TOKEN in the environment yourself:", e)

MODEL   = "google/gemma-3-27b-it"
LAYER   = 37          # residual after block 37, as in the paper (62 layers total)
ALPHA   = 4.0
N_NEW   = 60
OUT     = "/kaggle/working/positive_control.jsonl"
print("torch", torch.__version__, "| GPUs:", torch.cuda.device_count(),
      [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])

# %%
# ---- load 4-bit across both T4s ---------------------------------------------------
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_use_double_quant=True,
                         bnb_4bit_compute_dtype=torch.float16)
tok = AutoTokenizer.from_pretrained(MODEL)
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    MODEL, quantization_config=bnb, device_map="auto",
    max_memory={0: "14GiB", 1: "14GiB", "cpu": "24GiB"}, torch_dtype=torch.float16)
model.eval()
print(f"loaded in {time.time()-t0:.0f}s")
for i in range(torch.cuda.device_count()):
    print(f"  gpu{i}: {torch.cuda.memory_allocated(i)/1e9:.1f} GB allocated")

def find_layers(m):
    """Locate the decoder ModuleList regardless of the wrapper class."""
    for name, mod in m.named_modules():
        if name.endswith("layers") and isinstance(mod, torch.nn.ModuleList) and len(mod) > 30:
            return mod, name
    raise RuntimeError("could not find decoder layers")
layers, layers_name = find_layers(model)
print(f"decoder layers at `{layers_name}`: {len(layers)}  -> injecting after block {LAYER}")

# %%
# ---- residual capture and injection hooks ----------------------------------------
_cap = {}
def _capture_hook(_m, _i, out):
    h = out[0] if isinstance(out, tuple) else out
    _cap["h"] = h.detach()
    return out

_inj = {"v": None, "alpha": 0.0}
def _inject_hook(_m, _i, out):
    h = out[0] if isinstance(out, tuple) else out
    # Inject on the prompt (prefill) only: decode steps have seq_len == 1.
    if _inj["v"] is not None and _inj["alpha"] != 0.0 and h.shape[1] > 1:
        h = h + _inj["alpha"] * _inj["v"].to(h.device, h.dtype)
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
    return out

def chat(user_text):
    return tok.apply_chat_template([