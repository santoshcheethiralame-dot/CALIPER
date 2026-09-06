# %% [markdown]
# # Positive control — introspective detection in Gemma3-27B under 4-bit
#
# Reproduces the headline setting of Macar, Yang, Wang, Wallich, Ameisen & Lindsey
# (arXiv:2603.21396): **10.8% detection, 0% false positives, layer 37, α = 4.**
#
# Pre-registration: `docs/preregistration-positive-control-gemma3.md`. Criteria are fixed there.
#
# **Setup (once):** Accelerator → *GPU T4 x2*. Internet → *On*. Add a Kaggle secret named
# `HF_TOKEN` (accept the Gemma licence on Hugging Face first). Then Run All.
#
# Resumable: every trial is appended to `/kaggle/working/positive_control.jsonl` as it
# completes. If the session dies, Run All again and it continues from where it stopped.

# %%
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U",
                "bitsandbytes>=0.45", "accelerate>=1.0", "transformers>=4.50"], check=True)

# %%
import os, re, json, time, math, random
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

try:
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
except Exception:
    assert os.environ.get("HF_TOKEN"), "set HF_TOKEN (Kaggle secret or env var)"

MODEL   = "google/gemma-3-27b-it"
LAYER   = 37                  # residual stream after decoder block 37 (0-indexed)
ALPHAS  = [0.0, 4.0]          # control, headline
SEEDS   = [0, 1, 2]
MAX_NEW = 60
OUT     = "/kaggle/working/positive_control.jsonl" if os.path.isdir("/kaggle/working") \
          else "positive_control.jsonl"

torch.backends.cuda.matmul.allow_tf32 = False
print("GPUs:", torch.cuda.device_count(),
      [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])

# %% [markdown]
# ## Load the model in 4-bit across both T4s

# %%
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_use_double_quant=True,
                         bnb_4bit_compute_dtype=torch.float16)   # T4 has no bf16

tok = AutoTokenizer.from_pretrained(MODEL)
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    MODEL, quantization_config=bnb, device_map="auto",
    max_memory={0: "13GiB", 1: "13GiB", "cpu": "24GiB"},
    torch_dtype=torch.float16, low_cpu_mem_usage=True)
model.eval()
print(f"loaded in {time.time()-t0:.0f}s")
for i in range(torch.cuda.device_count()):
    print(f"  gpu{i}: {torch.cuda.memory_allocated(i)/1e9:.1f} GB allocated")


def find_layers(m):
    for path in ("model.language_model.layers", "model.layers",
                 "language_model.model.layers", "language_model.layers"):
        obj = m
        try:
            for p in path.split("."):
                obj = getattr(obj, p)
            return obj
        except AttributeError:
            continue
    raise RuntimeError("decoder layers not found on this model class")

layers = find_layers(model)
print(f"{len(layers)} decoder layers; injecting after block {LAYER}")
assert LAYER < len(layers)


def chat(user_text):
    return tok.apply_chat_template([{"role": "user", "content": user_text}],
                                   tokenize=False, add_generation_prompt=True)

def _hidden(o):
    return o[0] if isinstance(o, tuple) else o

# %% [markdown]
# ## Sanity: fp16 forward must be finite

# %%
@torch.no_grad()
def last_token_resid(text):
    ids = tok(chat(text), return_tensors="pt").to(model.device)
    cap = {}
    h = layers[LAYER].register_forward_hook(
        lambda m, i, o: cap.__setitem__("h", _hidden(o)[0, -1].float().cpu()))
    try:
        model(**ids)
    finally:
        h.remove()
    return cap["h"]

probe = last_token_resid("Tell me about bread")
assert torch.isfinite(probe).all(), \
    "NaN/Inf in fp16 forward - set bnb_4bit_compute_dtype=torch.float32 and reload (