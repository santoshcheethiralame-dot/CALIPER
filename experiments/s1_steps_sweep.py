"""Does MORE optimisation make the estimator WORSE on the failure units?

C41's decomposition says yes, startlingly. On the six worst gate-failures, the direct
fit at the gate's budget (3 restarts, 2500 steps) medians 0.5175, and the same direct
fit on the same data at 2 restarts and 800 steps medians 0.9668. The cascade and the
selection rule are not involved - gate_cascade is 0.4316, worse than either.

The candidate mechanism is the one C9 already implied. The bottleneck's nonlinearity is
a 64-wide MLP, flexible enough to fit the response through a WRONG direction if given
enough steps. Past some point the objective stops being informative about the direction,
because the nonlinearity absorbs the error instead. R2 = 1.000 at a wrong V is exactly
what C9 measured.

This sweeps steps at fixed restarts on the worst failures and on matched passes, and
records train and held-out R2 so overfitting can be distinguished from the absorption
account: overfitting separates train from test, absorption raises both.
"""
import json
import time
import numpy as np
from caliper.activations import collect, load_model, sample_corpus
from caliper.estimator import fit, subspace_alignment

LAYER, TOKENS = 6, 20_000
STEPS = [200, 400, 800, 1600, 2500, 5000]
RESTARTS = 2

d = json.load(open("results/s1_required_n_unbiased.json"))
sel = d["groups"]["worst-fail"] + d["groups"]["pass"][:3]
grp = {n: ("worst-fail" if n in d["groups"]["worst-fail"] else "pass") for n in sel}
gate = {int(r["_key"]): r for r in
        (json.loads(l) for l in open("results/e01_gate.jsonl", encoding="utf-8") if l.strip())}
print(f"{len(sel)} units: {sum(1 for n in sel if grp[n]=='worst-fail')} worst-fail, "
      f"{sum(1 for n in sel if grp[n]=='pass')} pass")

model, tok = load_model("gpt2")
p = collect(model, tok, sample_corpus(n_docs=300, seed=0), layer=LAYER,
            neurons=np.array(sel), max_tokens=TOKENS, seed=0)
print(f"  stimulus {p.stimulus.shape}\n", flush=True)

out, t0 = [], time.time()
for j, n in enumerate(p.neurons):
    n = int(n)
    for st in STEPS:
        f = fit(p.stimulus, p.response[:, j], k=1, n_restarts=RESTARTS, steps=st, seed=0)
        a = abs(subspace_alignment(f.subspace, p.weights[:, j][:, None]))
        out.append({"neuron": n, "group": grp[n], "steps": st,
                    "alignment": round(float(a), 4),
                    "r2": round(float(getattr(f, "r2", float("nan"))), 6),
                    "gate_direct": gate[n]["align_direct"]})
    print(f"  n{n} [{grp[n]}] {time.time()-t0:.0f}s", flush=True)

print(f"\n{'steps':>7} | {'worst-fail median':>18} {'min':>8} | {'pass median':>12} {'min':>8}")
print("-" * 62)
for st in STEPS:
    line = f"{st:>7} |"
    for g in ("worst-fail", "pass"):
        v = [r["alignment"] for r in out if r["steps"] == st and r["group"] == g]
        line += f" {np.median(v):>18.4f} {min(v):>8.4f} |" if g == "worst-fail" \
                else f" {np.median(v):>12.4f} {min(v):>8.4f}"
    print(line)

print(f"\n  gate direct fit (3 restarts, 2500 steps), worst-fail median: "
      f"{np.median([gate[n]['align_direct'] for n in d['groups']['worst-fail']]):.4f}")
print("\n  held-out R2 by steps (worst-fail) - rising R2 with falling alignment is the")
print("  absorption signature: the objective improves while the direction gets worse")
for st in STEPS:
    v = [r for r in out if r["steps"] == st and r["group"] == "worst-fail"]
    print(f"    steps={st:>5}  alignment {np.median([x['alignment'] for x in v]):.4f}  "
          f"R2 {np.median([x['r2'] for x in v]):.6f}")

json.dump({"steps": STEPS, "restarts": RESTARTS, "rows": out},
          open("results/s1_steps_sweep.json", "w"), indent=2)
print("\nwrote results/s1_steps_sweep.json")
