"""E0.1 gate run at n = 100. Protocol frozen in docs/preregistration-e01-n100.md.

Dual protocol (direct + cascade, selected by held-out R^2) at the E0.3b operating point,
batched, resumable. Pass rate reported as a Wilson interval, because a bare fraction over
20 neurons cannot resolve a 0.95 threshold.
"""
import argparse, json, time
import numpy as np
from caliper.activations import collect, load_model, sample_corpus
from caliper.batched import fit_batch
from caliper.estimator import fit_cascade, subspace_alignment
from caliper.runtime import Checkpoint, pick_device


def wilson(k, n, z=1.96):
    """95% CI on a proportion. Normal approximation is invalid near 1.0."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return (max(0.0, c - h), min(1.0, c + h))


ap = argparse.ArgumentParser()
ap.add_argument("--neurons", type=int, default=100)
ap.add_argument("--tokens", type=int, default=8000)     # E0.3b operating point
ap.add_argument("--steps", type=int, default=1600)
ap.add_argument("--restarts", type=int, default=2)
ap.add_argument("--layer", type=int, default=6)
ap.add_argument("--batch", type=int, default=32)        # speedup saturates by 32
ap.add_argument("--device", default="auto")
ap.add_argument("--model", default="gpt2",
                help="HF id. gpt2 is C13; EleutherAI/pythia-160m is the "
                     "second-family replication, prereg docs/preregistration-e01-pythia.md")
ap.add_argument("--d-mlp", type=int, default=3072,
                help="units to draw from; 3072 for both gpt2 and pythia-160m")
ap.add_argument("--out", default="results/e01_gate.jsonl")
a = ap.parse_args()

t0 = time.time()
device = pick_device(a.device)
ck = Checkpoint(a.out)

model, tok = load_model(a.model)
rng = np.random.default_rng(0)
neurons = rng.choice(a.d_mlp, size=a.neurons, replace=False)
p = collect(model, tok, sample_corpus(n_docs=300, seed=0), layer=a.layer,
            neurons=neurons, max_tokens=a.tokens, seed=0)
print(f"  stimulus {p.stimulus.shape}  ({time.time()-t0:.0f}s)", flush=True)

alive = p.response.std(0) > 1e-4
print(f"  alive: {int(alive.sum())}/{len(alive)} (screened units are reported, not dropped)",
      flush=True)

todo = [i for i in range(len(p.neurons)) if alive[i] and not ck.done(int(p.neurons[i]))]
for start in range(0, len(todo), a.batch):
    idx = todo[start:start + a.batch]
    Y = p.response[:, idx]
    d1 = fit_batch(p.stimulus, Y, k=1, n_restarts=a.restarts, steps=a.steps,
                   seed=0, device=device)
    d2 = fit_batch(p.stimulus, Y, k=2, n_restarts=a.restarts, steps=a.steps,
                   seed=0, device=device)
    for j, i in enumerate(idx):
        n = int(p.neurons[i])
        wu = p.weights[:, i] / np.linalg.norm(p.weights[:, i])
        c = fit_cascade(p.stimulus, p.response[:, i], k=1,
                        n_restarts=a.restarts, steps=a.steps, seed=0)
        ad = abs(subspace_alignment(d1[j].subspace, wu[:, None]))
        ac = abs(subspace_alignment(c.subspace, wu[:, None]))
        use_cascade = c.test_r2 > d1[j].test_r2
        sel = ac if use_cascade else ad
        ck.record(n, {
            "align_direct": round(ad, 4), "align_cascade": round(ac, 4),
            "align_selected": round(sel, 4), "best_available": round(max(ad, ac), 4),
            "picked": "cascade" if use_cascade else "direct",
            "r2_k1": round(float(max(d1[j].test_r2, c.test_r2)), 6),
            "k2_gain": round(float(d2[j].test_r2 - max(d1[j].test_r2, c.test_r2)), 4),
            "disagreement": round(abs(ad - ac), 4),
        })
    el = time.time() - t0
    print(f"  {len(ck.rows())}/{int(alive.sum())} neurons  {el:.0f}s "
          f"({el/max(len(ck.rows()),1):.1f}s each)", flush=True)

rows = ck.rows()
sel = np.array([r["align_selected"] for r in rows])
gain = np.array([r["k2_gain"] for r in rows])
best = np.array([r["best_available"] for r in rows])
dis = np.array([r["disagreement"] for r in rows])
passed = (sel > 0.95) & (gain < 0.01)
lo, hi = wilson(int(passed.sum()), len(rows))

summary = {
    "n": len(rows), "n_passing": int(passed.sum()),
    "pass_rate": round(float(passed.mean()), 4),
    "wilson_95_lower": round(lo, 4), "wilson_95_upper": round(hi, 4),
    "median_alignment": round(float(np.median(sel)), 4),
    "min_alignment": round(float(sel.min()), 4),
    "median_k2_gain": round(float(np.median(gain)), 4),
    "median_regret": round(float(np.median(best - sel)), 4),
    "max_regret": round(float((best - sel).max()), 4),
    "frac_methods_disagree_gt_0.05": round(float((dis > 0.05).mean()), 4),
    "seconds_per_neuron": round((time.time() - t0) / max(len(rows), 1), 1),
    "VERDICT": "PASS" if lo > 0.90 else "FAIL",
}
json.dump(summary, open("results/e01_gate_summary.json", "w"), indent=2)
print("\n" + "=" * 66)
for k, v in summary.items():
    print(f"  {k:.<44} {v}")
print("=" * 66)
print("  Criterion (pre-registered): Wilson 95% lower bound > 0.90")
