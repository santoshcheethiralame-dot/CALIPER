"""S1 multi-seed confirmatory run, per docs/preregistration-s1-multiseed.md.

Frozen protocol. Do not edit to make a result nicer; the pre-registration is filed and
dated, and any change here after data exists invalidates the run.

  units      the same 100 as C13, read from e01_gate.jsonl
  fits       5 seeds (0-4), n_restarts=1, steps=800, at k=1 AND k=2
  selection  highest test_r2. Ground truth is never consulted in selection.
  pass       alignment > 0.95 AND k2_gain < 0.01, identical to C13
  k2_gain    (best k=2 test_r2) - (selected k=1 test_r2), per the 7 Sep clarification

Appends one JSON line per unit and skips units already present, so a killed session
resumes rather than restarting.
"""
import json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from caliper.activations import collect, load_model, sample_corpus
from caliper.estimator import fit, subspace_alignment

LAYER, TOKENS, STEPS, SEEDS, KS = 6, 20_000, 800, [0, 1, 2, 3, 4], (1, 2)
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = "/kaggle/working/s1_multiseed.jsonl" if os.path.isdir("/kaggle/working") \
      else os.path.join(HERE, "s1_multiseed.jsonl")

dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {dev}", flush=True)
if dev == "cuda":
    print(f"  {torch.cuda.get_device_name(0)}", flush=True)

gate = {int(r["_key"]): r for r in
        (json.loads(l) for l in open(os.path.join(HERE, "e01_gate.jsonl"), encoding="utf-8")
         if l.strip())}
ids = sorted(gate)
print(f"{len(ids)} units from the C13 gate", flush=True)

done = set()
if os.path.exists(OUT):
    done = {json.loads(l)["neuron"] for l in open(OUT, encoding="utf-8") if l.strip()}
    print(f"resuming: {len(done)} already complete", flush=True)

model, tok = load_model("gpt2")
p = collect(model, tok, sample_corpus(n_docs=300, seed=0), layer=LAYER,
            neurons=np.array(ids), max_tokens=TOKENS, seed=0)
print(f"  stimulus {p.stimulus.shape}", flush=True)
del model
S = p.stimulus

fh = open(OUT, "a")
t0 = time.time()
for j, n in enumerate(p.neurons):
    n = int(n)
    if n in done:
        continue
    wu = p.weights[:, j] / np.linalg.norm(p.weights[:, j])
    best = {}
    for k in KS:
        cand = []
        for sd in SEEDS:
            f = fit(S, p.response[:, j], k=k, n_restarts=1, steps=STEPS, seed=sd, device=dev)
            cand.append((float(f.test_r2), f))
        r2, f = max(cand, key=lambda x: x[0])
        best[k] = {"test_r2": r2, "fit": f,
                   "all_r2": sorted(round(c[0], 6) for c in cand)}
    a_sel = abs(subspace_alignment(best[1]["fit"].subspace, wu[:, None]))
    k2g = best[2]["test_r2"] - best[1]["test_r2"]
    row = {"neuron": n, "align_multiseed": round(float(a_sel), 4),
           "k2_gain": round(float(k2g), 4),
           "test_r2_k1": round(best[1]["test_r2"], 6),
           "test_r2_k2": round(best[2]["test_r2"], 6),
           "seed_r2_k1": best[1]["all_r2"],
           "gate_align_selected": gate[n]["align_selected"],
           "gate_align_direct": gate[n]["align_direct"],
           "gate_k2_gain": gate[n]["k2_gain"],
           "steps": STEPS, "seeds": SEEDS, "device": dev}
    fh.write(json.dumps(row) + "\n"); fh.flush(); os.fsync(fh.fileno())
    el = time.time() - t0
    ndone = j + 1 - len([x for x in ids[:j + 1] if x in done])
    if ndone:
        print(f"  {j+1}/{len(ids)}  n{n} {a_sel:.4f} (gate {gate[n]['align_selected']:.4f})"
              f"  {el:.0f}s  eta {el/ndone*(len(ids)-j-1)/60:.0f}m", flush=True)
fh.close()

rows = [json.loads(l) for l in open(OUT, encoding="utf-8") if l.strip()]
def wilson(k, n, z=1.96):
    if not n: return (0.0, 0.0)
    ph = k / n; d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * ((ph * (1 - ph) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, c - h), min(1.0, c + h))

ms = sum(r["align_multiseed"] > 0.95 and r["k2_gain"] < 0.01 for r in rows)
gt = sum(r["gate_align_selected"] > 0.95 and r["gate_k2_gain"] < 0.01 for r in rows)
lo, hi = wilson(ms, len(rows))
print("\n" + "=" * 66)
print(f"PRIMARY  multi-seed pass rate {ms}/{len(rows)} = {ms/len(rows):.1%}")
print(f"         Wilson 95% [{lo:.4f}, {hi:.4f}]")
print(f"         criterion: lower bound > 0.90  ->  {'PASS' if lo > 0.90 else 'FAIL'}")
print(f"REFERENCE  C13 single-fit pass rate on the same units {gt}/{len(rows)} = {gt/len(rows):.1%}")
d = [r["align_multiseed"] - r["gate_align_selected"] for r in rows]
print(f"SECONDARY  paired change median {np.median(d):+.4f}, "
      f"improved {sum(x > 0 for x in d)}/{len(d)}, worsened {sum(x < 0 for x in d)}/{len(d)}")
was = [r for r in rows if r["gate_align_selected"] > 0.95 and r["gate_k2_gain"] < 0.01]
hurt = [r for r in was if not (r["align_multiseed"] > 0.95 and r["k2_gain"] < 0.01)]
print(f"           units C13 passed that this loses: {len(hurt)}/{len(was)}")
print("=" * 66)
print(f"\nwrote {OUT}")
