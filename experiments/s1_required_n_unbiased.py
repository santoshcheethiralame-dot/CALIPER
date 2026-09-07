"""C37 without the selection biases, plus the control C37 was missing.

C37 required n_events >= 1600, which kept 8 of 23 gate-failures and excluded every
catastrophic one; its six failures were all near-misses at 0.86-0.95 against a 0.95 bar.
It also changed two things at once - less data AND a cheaper fit config - so "the
failures recovered on less data" was confounded with "the failures recovered under a
different optimiser budget".

This fixes both:

  * no event filter. Targets are capped per unit at whatever it has, so low-event units
    contribute at the levels they can reach and are reported with their own n.
  * units stratified into worst failures / marginal failures / passes by gate alignment,
    so the class that carries the 23% headline is actually represented.
  * a FULL-data arm at the same cheap config. Comparing gate -> full isolates the
    optimiser budget; comparing full -> subsampled isolates the amount of data. C37
    could not separate them.

Local, CPU.
"""
import json
import time
import numpy as np
from caliper.activations import collect, load_model, sample_corpus
from caliper.estimator import fit, subspace_alignment

LAYER, TOKENS = 6, 20_000
TARGETS = [50, 100, 200, 400, 800, 1600]
FIT = dict(k=1, n_restarts=2, steps=800, seed=0)
N_PER_GROUP = 6

rows = [json.loads(l) for l in open("results/e01_gate.jsonl", encoding="utf-8") if l.strip()]
gate = {int(r["_key"]): r for r in rows}
passed = {n: (r["align_selected"] > 0.95 and r["k2_gain"] < 0.01) for n, r in gate.items()}
stats = json.load(open("results/s1_failure_class.json"))["per_unit"]

fails = sorted((n for n in gate if not passed[n]), key=lambda n: gate[n]["align_selected"])
passes = sorted((n for n in gate if passed[n]), key=lambda n: -gate[n]["align_selected"])
groups = {"worst-fail": fails[:N_PER_GROUP],
          "marginal-fail": fails[-N_PER_GROUP:],
          "pass": passes[:N_PER_GROUP]}
sel = [n for g in groups.values() for n in g]
grp_of = {n: g for g, ns in groups.items() for n in ns}

for g, ns in groups.items():
    al = [gate[n]["align_selected"] for n in ns]
    ev = [stats[str(n)]["n_events"] for n in ns]
    print(f"{g:<14} n={len(ns)}  gate align {min(al):.3f}-{max(al):.3f}  "
          f"events {min(ev)}-{max(ev)}")

model, tok = load_model("gpt2")
p = collect(model, tok, sample_corpus(n_docs=300, seed=0), layer=LAYER,
            neurons=np.array(sel), max_tokens=TOKENS, seed=0)
S, W = p.stimulus, p.weights
print(f"  stimulus {S.shape}\n", flush=True)

rng = np.random.default_rng(0)
out, t0 = [], time.time()
for j, n in enumerate(p.neurons):
    n = int(n)
    z = S @ W[:, j]
    act, inact = np.flatnonzero(z > 0), np.flatnonzero(z <= 0)
    frac = len(act) / len(z)
    # every reachable target, plus the full-data control at this unit's own event count
    for E in [e for e in TARGETS if e <= len(act)] + ["full"]:
        if E == "full":
            idx = np.arange(len(z))
            e_used = len(act)
        else:
            n_in = min(int(round(E * (1 - frac) / frac)), len(inact))
            idx = np.concatenate([rng.choice(act, E, replace=False),
                                  rng.choice(inact, n_in, replace=False)])
            rng.shuffle(idx)
            e_used = E
        f = fit(S[idx], p.response[idx, j], **FIT)
        a = abs(subspace_alignment(f.subspace, W[:, j][:, None]))
        out.append({"neuron": n, "group": grp_of[n], "target": E, "events": e_used,
                    "n_rows": len(idx), "alignment": round(float(a), 4),
                    "gate_alignment": gate[n]["align_selected"]})
    print(f"  n{n} [{grp_of[n]}] {stats[str(n)]['n_events']} events  "
          f"{time.time()-t0:.0f}s", flush=True)

print(f"\n{'target':>8} | " + " | ".join(f"{g:>22}" for g in groups))
print(f"{'':>8} | " + " | ".join(f"{'median':>9} {'min':>7} {'n':>3}" for _ in groups))
print("-" * 84)
for E in TARGETS + ["full"]:
    line = f"{str(E):>8} |"
    for g in groups:
        v = [r["alignment"] for r in out if r["target"] == E and r["group"] == g]
        line += (f" {np.median(v):>9.4f} {min(v):>7.4f} {len(v):>3} |" if v
                 else f" {'-':>9} {'-':>7} {0:>3} |")
    print(line)

print("\n=== the decomposition C37 could not do ===")
print("  gate  -> full   isolates the optimiser budget (same data, cheaper config)")
print("  full  -> best   isolates resampling (same config, different subsets)\n")
print(f"  {'group':<14} {'gate':>8} {'full':>8} {'best sub':>9} {'gate->full':>11} {'full->best':>11}")
for g in groups:
    ns = groups[g]
    gt = np.median([gate[n]["align_selected"] for n in ns])
    fu = np.median([r["alignment"] for r in out if r["group"] == g and r["target"] == "full"])
    bs = np.median([max(r["alignment"] for r in out
                        if r["neuron"] == n and r["target"] != "full") for n in ns])
    print(f"  {g:<14} {gt:>8.4f} {fu:>8.4f} {bs:>9.4f} {fu-gt:>+11.4f} {bs-fu:>+11.4f}")

json.dump({"targets": TARGETS, "fit": FIT, "groups": {g: ns for g, ns in groups.items()},
           "rows": out}, open("results/s1_required_n_unbiased.json", "w"), indent=2)
print("\nwrote results/s1_required_n_unbiased.json")
