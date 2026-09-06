"""E0.3 - required N for a target recovery, indexed by events and by K.

The specification promised a sample-complexity table; nobody has one for language
models. Two things measured this week reshape what it should contain:

  * N is not the binding constraint above a few thousand events (E0.3b), so the table
    must show WHERE it plateaus rather than assuming a monotone curve;
  * the criterion is the worst case, not the median - E0.1's median was 0.993 while 23%
    of units were silently wrong.

Reported per (K, N): median and 10th-percentile recovery, and the smallest N reaching a
0.95 tenth percentile. Synthetic units on the real residual stream, so the direction is
known while the stimulus statistics are genuine.
"""
import itertools, json, time
import numpy as np
from caliper.activations import collect, load_model, sample_corpus
from caliper.batched import fit_batch
from caliper.estimator import subspace_alignment
from caliper.runtime import Checkpoint, pick_device

N_GRID = [2000, 4000, 8000]
K_GRID = [2, 3]
N_UNITS = 24
COUPLING = __import__("os").environ.get("COUPLING", "additive")

device = pick_device("auto")
ck = Checkpoint("results/e03_required_n.jsonl")
t0 = time.time()

m, tok = load_model("gpt2")
p = collect(m, tok, sample_corpus(n_docs=300, seed=0), layer=6,
            neurons=np.array([0]), max_tokens=max(N_GRID), seed=0)
X = p.stimulus
D = X.shape[1]
rng = np.random.default_rng(0)
g = lambda x: 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
print(f"  stimulus {X.shape}  ({time.time()-t0:.0f}s)", flush=True)

print(f"\n{'K':>3} {'N':>7} {'events':>8} {'median':>8} {'p10':>8} {'sec':>7}")
for K, N in itertools.product(K_GRID, N_GRID):
    key = f"{COUPLING}_K{K}_N{N}"
    if ck.done(key):
        continue
    # Plant N_UNITS units of known dimensionality K, sparse-firing like real neurons.
    V = np.stack([np.linalg.qr(rng.standard_normal((D, K)))[0] for _ in range(N_UNITS)])
    Z = np.einsum("sd,ndk->snk", X[:N], V)
    lead = Z[:, :, 0]
    base = g(lead - np.quantile(lead, 0.90, axis=0))
    if COUPLING == "gated":
        # Multiplicative: dimensions beyond the first are only visible where the sparse
        # first dimension is open, so their effective sample is ~10% of N. This makes K>1
        # an order of magnitude harder than K=1 BY CONSTRUCTION, and the first version of
        # this experiment mistook that for an estimator limit.
        Y = base
        for j in range(1, K):
            Y = Y * np.tanh(Z[:, :, j])
    else:
        # Additive: every dimension is visible at every position - the fair comparison.
        Y = base + sum(np.tanh(Z[:, :, j]) for j in range(1, K))
    Y = Y.astype(np.float32)

    t1 = time.time()
    fits = fit_batch(X[:N], Y, k=K, n_restarts=2, steps=1600, seed=0, device=device)
    al = np.array([abs(subspace_alignment(fits[i].subspace, V[i])) for i in range(N_UNITS)])
    events = int((lead > np.quantile(lead, 0.90, axis=0)).sum() / N_UNITS)
    row = {"K": K, "N": N, "coupling": COUPLING, "events_per_unit": events,
           "median": round(float(np.median(al)), 4),
           "p10": round(float(np.percentile(al, 10)), 4),
           "min": round(float(al.min()), 4),
           "events_over_D": round(events / D, 3),
           "seconds": round(time.time() - t1, 1)}
    ck.record(key, row)
    print(f"{K:>3} {N:>7} {events:>8} {row['median']:>8.4f} {row['p10']:>8.4f} "
          f"{row['seconds']:>7.1f}", flush=True)

rows = sorted(ck.rows(), key=lambda r: (r["K"], r["N"]))
json.dump(rows, open("results/e03_required_n.json", "w"), indent=2)
print("\n  required N for a 0.95 tenth-percentile recovery:")
for K in K_GRID:
    ok = [r for r in rows if r["K"] == K and r["p10"] >= 0.95]
    if ok:
        b = min(ok, key=lambda r: r["N"])
        print(f"    K={K}: N={b['N']} ({b['events_per_unit']} events/unit, "
              f"events/D = {b['events_over_D']})")
    else:
        best = max((r for r in rows if r["K"] == K), key=lambda r: r["p10"])
        print(f"    K={K}: NOT REACHED on this grid (best p10 = {best['p10']:.3f} "
              f"at N={best['N']})")
