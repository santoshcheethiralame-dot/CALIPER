"""Run kaggle_s3_positive_control.py unchanged on a CPU, with fp32 weights at a lower peak memory.

    python experiments/cpu_fp32_lowmem.py <the script's usual arguments>

On the 16 GB laptop a direct fp32 load of Qwen3.5-2B commits 15.7 GB and the run pages to disk.
Here an unquantised fp32 load is done as a bf16 load cast to fp32, and every tensor the
checkpoint stores in fp32 (Qwen3.5's A_log and linear-attention norms) is then copied back from
the checkpoint, so the weights equal those of a direct fp32 load. Anything else is passed through.
"""
import gc
import glob
import os
import runpy
import sys

import torch
import transformers
from safetensors import safe_open

_orig = transformers.AutoModelForCausalLM.from_pretrained


def _restore_fp32(model, path):
    params = dict(model.named_parameters())
    n = 0
    for f in glob.glob(os.path.join(path, "*.safetensors")):
        with safe_open(f, "pt") as s:
            for k in s.keys():
                if s.get_slice(k).get_dtype() != "F32":
                    continue
                name = k.replace("model.language_model.", "model.", 1)
                if name not in params:
                    continue
                with torch.no_grad():
                    params[name].copy_(s.get_tensor(k))
                n += 1
    return n


def from_pretrained(path, *a, dtype=None, torch_dtype=None, **kw):
    dt = dtype or torch_dtype
    if dt is not torch.float32 or "quantization_config" in kw or not os.path.isdir(str(path)):
        return _orig(path, *a, dtype=dt, **kw)
    model = _orig(path, *a, dtype=torch.bfloat16, **kw).float()
    n = _restore_fp32(model, str(path))
    gc.collect()
    print(f"  cpu_fp32_lowmem: bf16 load cast to fp32, {n} fp32 checkpoint tensors restored", flush=True)
    return model


transformers.AutoModelForCausalLM.from_pretrained = staticmethod(from_pretrained)

if __name__ == "__main__":
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kaggle_s3_positive_control.py")
    sys.argv = [script] + sys.argv[1:]
    runpy.run_path(script, run_name="__main__")
