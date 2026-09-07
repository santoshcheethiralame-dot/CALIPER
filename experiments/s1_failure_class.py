"""S1-3 - what distinguishes the 23% of units the estimator fails on?

C7 looked at 8 neurons and found failures sitting at z_mean -1.3 to -1.9, deep in
GELU's non-monotone region, with softplus and identity surrogates recovering them. C6
found a rank transform helping on 3. Neither is a characterisation: this scores every
unit in the C13 gate run.

The neuron ids come from the gate output itself, so the sample matches by construction
rather than by reproducing a seed. Local, CPU, no GPU.
"""
import json
import numpy as np
from caliper.activations import collect, load_model, sample_corpus

LAYER, TOKENS, PASS = 6, 20_000, 0.95

rows = [json.loads(l) for l in open("results/e01_gate.jsonl", encoding="utf-8") if l.strip()]
ids = [int(r["_key"]) for r in rows]
align = {int(r["_key"]): r["align_selected"] for r in rows}
r2 = {int(r["_key"]): r["r2_k1"] for r in rows}
print(f"{len(ids)} units from the gate run, {sum(align[i] < PASS for i in ids)} failures")

print("collecting activations ...", flush=True)
model, tok = load_model("gpt2")
p = collect(model, tok, sample_corpus(n_docs=300, seed=0), layer=LAYER,
            neurons=np.array(ids), max_tokens=TOKENS, seed=0)
print(f"  stimulus {p.stimulus.shape}, response {p.response.shape}", flush=True)

S, W = p.stimulus, p.weights
stats = {}
for j, n in enumerate(p.neurons):
    z = S @ W[:, j]                      # pre-activation, the exact quantity fitted
    a = p.response[:, j]
    sd = z.std() + 1e-12
    pos = z > 0
    stats[int(n)] = {
        "z_mean": float(z.mean()),
        "z_std": float(sd),
        "z_mean_over_std": float(z.mean() / sd),      # how far below zero, in sigma
        "frac_active": float(pos.mean()),
        "n_events": int(pos.sum()),
        "resp_kurtosis": float(((a - a.mean()) ** 4).mean() / (a.var() ** 2 + 1e-12)),
        "resp_max_over_std": float((a.max() - a.mean()) / (a.std() + 1e-12)),
        "w_norm": float(np.linalg.norm(W[:, j])),
    }

def auc(scores, labels):
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return float("nan")
    return sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / (len(pos) * len(neg))

fail = [align[i] < PASS for i in ids]
print(f"\n=== does any activation statistic predict failure? ===")
print(f"  {'statistic':<20} {'AUC':>6} {'mean|fail':>12} {'mean|pass':>12}")
out = {}
for k in next(iter(stats.values())):
    v = [stats[i][k] for i in ids]
    a1 = auc(v, fail)
    # report the better orientation, since direction is not known in advance
    a = max(a1, 1 - a1)
    mf = np.mean([x for x, y in zip(v, fail) if y])
    mp = np.mean([x for x, y in zip(v, fail) if not y])
    out[k] = {"auc_raw": a1, "auc_oriented": a, "mean_fail": float(mf), "mean_pass": float(mp)}
    flag = "  <-- " if a >= 0.70 else ""
    print(f"  {k:<20} {a:>6.3f} {mf:>12.4f} {mp:>12.4f}{flag}")

print(f"\n  (for reference, C35: held-out R2 AUC 0.906, disagreement 0.802)")
print(f"\n=== C7's claim: do failures sit at z_mean -1.3 to -1.9? ===")
zf = sorted(stats[i]["z_mean"] for i in ids if align[i] < PASS)
zp = sorted(stats[i]["z_mean"] for i in ids if align[i] >= PASS)
print(f"  failures (n={len(zf)}): min {zf[0]:.2f}  median {zf[len(zf)//2]:.2f}  max {zf[-1]:.2f}")
print(f"  passes   (n={len(zp)}): min {zp[0]:.2f}  median {zp[len(zp)//2]:.2f}  max {zp[-1]:.2f}")
inband = sum(1 for z in zf if -1.9 <= z <= -1.3)
inband_p = sum(1 for z in zp if -1.9 <= z <= -1.3)
print(f"  in C7's [-1.9, -1.3] band: {inband}/{len(zf)} failures, {inband_p}/{len(zp)} passes")

json.dump({"pass_bar": PASS, "n": len(ids), "n_fail": int(sum(fail)),
           "predictors": out, "per_unit": stats,
           "align": {str(k): v for k, v in align.items()},
           "r2_k1": {str(k): v for k, v in r2.items()}},
          open("results/s1_failure_class.json", "w"), indent=2)
print("\nwrote results/s1_failure_class.json")
