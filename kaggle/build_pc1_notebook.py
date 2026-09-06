"""Emit the PC1 positive-control notebook as .ipynb.

Run:  python build_pc1_notebook.py  ->  pc1_gemma27b_detection.ipynb
Then upload to Kaggle, set Accelerator = GPU T4 x2, add secret HF_TOKEN.
"""
import json
from pathlib import Path

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": s.strip("\n")})


md("""
# PC1 — Positive control: introspective detection on Gemma3-27B in 4-bit

**Pre-registered:** `docs/preregistration-pc1-gemma27b-detection.md` (CALIPER repo).
**Target to reproduce:** Macar et al. 2603.21396 — 10.8% detection, 0% FPR at L=37, α=4.

**Before running**
1. Accelerator → **GPU T4 x2** (this does *not* fit a single P100).
2. Add-ons → Secrets → `HF_TOKEN` with a Hugging Face token that has accepted the Gemma licence.
3. Internet → on (for the model download).

Resumable: results append to `/kaggle/working/pc1_results.jsonl`; re-running skips finished trials.
""")

code("""
!pip -q install -U "transformers>=4.50" "accelerate>=0.30" "bitsandbytes>=0.43" scipy
""")

code("""
import os, json, re, time, math, random
import numpy as np, torch
from pathlib import Path
from kaggle_secrets import UserSecretsClient

os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")

MODEL    = "google/gemma-3-27b-it"
LAYER    = 37            # residual output of block 37 (0-indexed), 62 layers total
ALPHAS   = [0.0, 4.0]    # 0 = matched control (FPR); 4 = published operating point
N_TRIALS = 6             # per concept per alpha
MAX_NEW  = 60
TEMP     = 1.0
OUT      = Path("/kaggle/working/pc1_results.jsonl")
SEED     = 0

torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)
print("GPUs:", torch.cuda.device_count(),
      [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
""")

md("""
## Concepts (40, fixed before the run) and baseline nouns (100)
Declared deviation from the source: a 40-concept subset, not 500. Mix of concrete, abstract and
affective items, Lindsey-style.
""")

code("""
CONCEPTS = [
  # concrete
  "elephant","volcano","telescope","violin","umbrella","lighthouse","cactus","submarine",
  "waterfall","glacier","cathedral","tornado","pineapple","saxophone","hedgehog","compass",
  # abstract
  "justice","infinity","democracy","gravity","nostalgia","irony","entropy","freedom",
  "silence","chaos","symmetry","destiny",
  # affective
  "joy","fear","jealousy","serenity","grief","curiosity","shame","awe",
  # misc / Lindsey-style
  "bread","aquarium","dust","ALL CAPS shouting",
]
assert len(CONCEPTS) == 40

BASELINE_NOUNS = [
  "table","river","window","book","mountain","chair","garden","cloud","bottle","road",
  "letter","kitchen","forest","bridge","pencil","market","island","blanket","engine","valley",
  "ladder","harbor","pillow","meadow","candle","tunnel","basket","orchard","mirror","fence",
  "hammer","desert","carpet","lantern","canyon","saddle","ribbon","furnace","anchor","pasture",
  "wagon","ceiling","puzzle","shovel","cliff","teapot","statue","corridor","helmet","stream",
  "curtain","barrel","planet","bakery","needle","canal","trumpet","cabinet","pebble","tower",
  "jacket","village","spoon","cellar","kite","grove","wallet","hallway","comet","stable",
  "pitcher","attic","rope","pond","banner","tractor","meadow","chimney","locket","dune",
  "scarf","plaza","goblet","hive","pier","satchel","lagoon","gutter","flute","quarry",
  "cradle","summit","apron","lodge","tulip","vault","marsh","beacon","sled","thicket",
]
BASELINE_NOUNS = list(dict.fromkeys(BASELINE_NOUNS))[:100]
assert len(BASELINE_NOUNS) == 100
""")

md("""
## Load Gemma3-27B in 4-bit across both T4s
T4 has no bf16; compute in fp16. First generation is checked for NaNs/degeneracy — Gemma has a
history of fp16 overflow. If that check fails, switch `COMPUTE_DTYPE` to `torch.float32` (slower).
""")

code("""
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

COMPUTE_DTYPE = torch.float16
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_use_double_quant=True,
                         bnb_4bit_compute_dtype=COMPUTE_DTYPE)
tok = AutoTokenizer.from_pretrained(MODEL)
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    MODEL, quantization_config=bnb, device_map="auto",
    torch_dtype=COMPUTE_DTYPE, low_cpu_mem_usage=True)
model.eval()
print(f"loaded in {time.time()-t0:.0f}s")
for i in range(torch.cuda.device_count()):
    print(f"  GPU{i}: {torch.cuda.memory_allocated(i)/1e9:.1f} GB allocated")
""")

code("""
def find_decoder_layers(m):
    \"\"\"Gemma3 is multimodal; the text decoder's layer list moves between versions.\"\"\"
    best = None
    for name, mod in m.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and len(mod) >= 40:
            if best is None or len(mod) > len(best[1]):
                best = (name, mod)
    assert best is not None, "could not find decoder layer list"
    print(f"decoder layers at `{best[0]}` ({len(best[1])} layers)")
    return best[1]

LAYERS = find_decoder_layers(model)
assert len(LAYERS) == 62, f"expected 62 layers, found {len(LAYERS)}"
TARGET = LAYERS[LAYER]
""")

md("""
## Concept vectors: `v_c = h_c − mean(h_baseline)` at the last token of "Tell me about {c}"
""")

code("""
_cap = {}