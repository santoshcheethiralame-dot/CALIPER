"""B-0 - is the device a confound?

Before any B-series run moves to a GPU, one question has to be answered: does fitting on
CUDA give the same answer as fitting on CPU?

It is not a pedantic question here. The whole study is about WHICH UNITS FAIL. A failing
unit is one where the optimiser lands in the wrong basin, which means failing units sit
near basin boundaries by construction - exactly the population where a tiny difference in
floating-point reduction order can flip the outcome. GPU and CPU do not agree bitwise.
So a run comparing GPT-2 on CPU against Pythia on GPU could report a difference in
failure rate that is a difference in hardware.

C31 is the precedent. A vector that carried no content passed every health check anyone
reports, and only a positive control caught it. The device question gets the same
treatment: measure it, do not assume it.

This script fits the SAME units with the SAME seeds on both devices and compares.

Usage (Kaggle, GPU on):
    python experiments/kaggle_device_equivalence.py --neurons 16 --restarts 2

Pre-registered criterion, filed before the run:
    PASS  - no unit changes pass/fail side, and max |d alignment| < 0.01.
            Device is not a confound. B-series runs may mix devices freely.
    FAIL  - any unit flips side, or max |d alignment| >= 0.01.
            Every set of runs that gets compared must share one device, and the paper
            reports the device beside every number.
"""
import argparse
import json
import time

import numpy as np
import torch

from caliper.activations import collect, load_model, sample_corpus
from caliper.batched import fit_batch
from caliper.estimator import fit_cascade, subspace_alignment

PASS_BAR = 0.95
TOL = 0.01

ap = argparse.ArgumentParser()
ap.add_argument("--neurons", type=int, default=16)
ap.add_argument("--tokens", type=int, default=8000)
ap.add_argument("--steps", type=int, default=1600)
# 2, not 5, and deliberately: this run certifies the configuration the ladder and the
# primary arms actually use (Addendum 1), and its CPU arm is the expensive half. At 5
# restarts on Kaggle's 4-core CPU the CPU arm alone runs about 90 minutes while the GPU
# sits idle - quota spent proving something about a config nothing else runs.
ap.add_argument("--restarts", type=int, default=2)
ap.add_argument("--layer", type=int, default=6)
ap.add_argument("--model", default="gpt2")
ap.add_argument("--d-mlp", type=int, default=3072)
ap.add_argument("--out", default="/kaggle/working/device_equivalence.json")
a = ap.parse_args()

if not torch.cuda.is_available():
    raise SystemExit("no CUDA visible - this run is meaningless without both devices")

t0 = time.time()
model, tok = load_model(a.model)

# The SAME units the gate drew, so the comparison covers known passes and known failures
# rather than an easy sample. Same rng, same seed, same call as e01_gate.py.
rng = np.random.default_rng(0)
neurons = rng.choice(a.d_mlp, size=100, replace=False)[: a.neurons]
p = collect(model, tok, sample_corpus(n_docs=300, seed=0), layer=a.layer,
            neurons=neurons, max_tokens=a.tokens, seed=0)
print(f"  stimulus {p.stimulus.shape}  ({time.time() - t0:.0f}s)", flush=True)

results = {}
for device in ("cpu", "cuda"):
    t = time.time()
    Y = p.response
    d1 = fit_batch(p.stimulus, Y, k=1, n_restarts=a.restarts, steps=a.steps,
                   seed=0, device=device)
    rows = []
    for i in range(len(p.neurons)):
        wu = p.weights[:, i] / np.linalg.norm(p.weights[:, i])
        c = fit_cascade(p.stimulus, p.response[:, i], k=1,
                        n_restarts=a.restarts, steps=a.steps, seed=0)
        ad = abs(subspace_alignment(d1[i].subspace, wu[:, None]))
        ac = abs(subspace_alignment(c.subspace, wu[:, None]))
        sel = ac if c.test_r2 > d1[i].test_r2 else ad
        rows.append({"neuron": int(p.neurons[i]), "align_selected": round(float(sel), 6),
                     "stability": round(float(d1[i].stability), 6),
                     "r2_k1": round(float(max(d1[i].test_r2, c.test_r2)), 6)})
    el = time.time() - t
    results[device] = {"rows": rows, "seconds": round(el, 1),
                       "seconds_per_neuron": round(el / len(rows), 2)}
    print(f"  {device:<5} {el:6.0f}s  ({el / len(rows):.1f}s per neuron)", flush=True)

cpu, gpu = results["cpu"]["rows"], results["cuda"]["rows"]
d_align = [abs(x["align_selected"] - y["align_selected"]) for x, y in zip(cpu, gpu)]
d_stab = [abs(x["stability"] - y["stability"]) for x, y in zip(cpu, gpu)]
flips = [(x["neuron"], x["align_selected"], y["align_selected"])
         for x, y in zip(cpu, gpu)
         if (x["align_selected"] >= PASS_BAR) != (y["align_selected"] >= PASS_BAR)]

speedup = results["cpu"]["seconds"] / max(results["cuda"]["seconds"], 1e-9)
verdict = "PASS" if (not flips and max(d_align) < TOL) else "FAIL"

print(f"\n{'=' * 66}")
print(f"  units compared.................. {len(cpu)}")
print(f"  max |d alignment|.............. {max(d_align):.6f}   (tolerance {TOL})")
print(f"  median |d alignment|........... {np.median(d_align):.6f}")
print(f"  max |d restart agreement|...... {max(d_stab):.6f}")
print(f"  units flipping pass/fail....... {len(flips)}")
for n, c_, g_ in flips:
    print(f"      neuron {n}: cpu {c_:.4f} -> cuda {g_:.4f}")
print(f"  GPU speedup.................... {speedup:.1f}x")
print(f"  VERDICT........................ {verdict}")
print("=" * 66)
print("  PASS -> B-series runs may mix devices. FAIL -> every compared set shares one,")
print("  and the device is reported beside every number.")

json.dump({"verdict": verdict, "n": len(cpu), "tolerance": TOL,
           "max_abs_d_alignment": max(d_align),
           "median_abs_d_alignment": float(np.median(d_align)),
           "max_abs_d_stability": max(d_stab),
           "n_flips": len(flips), "flips": flips, "speedup": round(speedup, 2),
           "seconds": {k: v["seconds"] for k, v in results.items()},
           "config": vars(a), "results": results},
          open(a.out, "w"), indent=2)
print(f"  wrote {a.out}")
