"""Does C14's ~200-event required-N hold on real units?

C14 measured sample complexity on synthetic planted-direction units and found K=1
saturating at roughly 200 informative events. C36 raised a tension with that: real
failures average 1,353 events and passes 4,097, both far above 200, so raw event count
is evidently not what binds on real neurons. Either the synthetic figure does not
transfer, or something other than event count is the constraint.

This subsamples the corpus per unit to a target number of informative events, preserving
each unit's natural active fraction, and fits at each level. Units are stratified by
whether they passed the C13 gate, because "do failures need more data?" is the question
that decides whether the failure class is a data problem or a landscape problem.

Local, CPU.
"""
import json
import time
import numpy as np
from caliper.activations import collect, load_model, sample_corpus
from caliper.estimator import fit, subspace_alignment

LAYER, TOKENS = 6, 20_000
TARGETS = [50, 100, 200, 400, 800, 1600]
N_PER_GROUP = 6
FIT = dict(k=1, n_restarts=2, steps=800, seed=0)     # E0.3b's cheap knee

rows = [json.loads(l) for l in open("results/e01_gate.jsonl", encoding="utf-8") if l.strip()]
passed = {int(r["_key"]): (r["align_selected"] > 0.95 and r["k2_gain"] < 0.01) for r in rows}
stats = json.load(open("results/s1_failure_class.json"))["per_unit"]

# Stratify, and prefer units with enough events to reach the top target.
ok = [i for i in passed if stats[str(i)]["n_events"] >= max(TARGETS)]
sel = ([i for i in ok if passed[i]][:N_PER_GROUP] +
       [i for i in ok if not passed[i]][:N_PER_GROUP])
print(f"{len(sel)} units: {sum(passed[i] for i in sel)} passed the gate, "
      f"{sum(not passed[i] for i in sel)} failed")
print(f"  targets (informative events): {TARGETS}")

model, tok = load_model("gpt2")
p = collect(model, tok, sample_corpus(n_docs=300, seed=0), layer=LAYER,
            neurons=np.array(sel), max_tokens=TOKENS, seed=0)
S, W = p.stimulus, p.weights
print(f"  stimulus {S.shape}", flush=True)

rng = np.random.default_rng(0)
out = []
t0 = time.time()
for j, n in enumerate(p.neurons):
    n = int(n)
    z = S @ W[:, j]
    act = np.flatnonzero(z > 0)
    inact = np.flatnonzero(z <= 0)
    frac = len(act) / len(z)
    for E in TARGETS:
        if E > len(act):
            continue
        # Preserve the unit's natural sparsity: E active plus the matching inactive count.
        n_in = min(int(round(E * (1 - frac) / frac)), len(inact))
        idx = np.concatenate([rng.choice(act, E, replace=False),
                              rng.choice(inact, n_in, replace=False)])
        rng.shuffle(idx)
        f = fit(S[idx], p.response[idx, j], **FIT)
        a = abs(subspace_alignment(f.subspace, W[:, j][:, None]))
        out.append({"neuron": n, "passed": passed[n], "events": E,
                    "n_rows": len(idx), "alignment": round(float(a), 4)})
    print(f"  n{n} ({'pass' if passed[n] else 'FAIL'}) done  {time.time()-t0:.0f}s", flush=True)

print(f"\n{'events':>8} | {'passed gate':>28} | {'failed gate':>28}")
print(f"{'':>8} | {'median':>9} {'min':>9} {'n':>7} | {'median':>9} {'min':>9} {'n':>7}")
print("-" * 74)
for E in TARGETS:
    line = f"{E:>8} |"
    for grp in (True, False):
        v = [r["alignment"] for r in out if r["events"] == E and r["passed"] == grp]
        line += (f" {np.median(v):>9.4f} {min(v):>9.4f} {len(v):>7} |" if v
                 else f" {'-':>9} {'-':>9} {0:>7} |")
    print(line)

json.dump({"targets": TARGETS, "fit": FIT, "rows": out},
          open("results/s1_required_n_real.json", "w"), indent=2)
print("\nwrote results/s1_required_n_real.json")
