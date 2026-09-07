"""Is the failure class a seed lottery, and can held-out R2 pick the winning draw?

C42 refuted the over-optimisation hypothesis: at the gate's own 2500 steps with 2
restarts the worst failures median 0.9759, against the gate's 0.5175 at 3 restarts. Steps
are not the variable. What the sweep did show is an asymmetry in VARIANCE - across
nominally equivalent step counts the failure units spread by a median 0.161 and up to
0.634, while passes spread by 0.003.

So the working account is a rugged objective: for these units the recovered direction is
close to a lottery across configurations, and any single fit draws from it. This tests
that directly with independent single-restart fits at ten seeds, and asks the question
that decides whether it is repairable:

    does held-out R2 identify the good draws WITHOUT ground truth?

If it does, "fit several times, keep the best test_r2" is a working repair and the
disagreement flag is explained. If it does not, the objective cannot tell a recovered
direction from a lost one and no amount of restarting helps - which would be a much
sharper negative result about this estimator family.

Records train_r2 and test_r2 properly this time; C42 asked the Fit object for `r2`,
which does not exist, and silently logged nan.
"""
import json
import time
import numpy as np
from caliper.activations import collect, load_model, sample_corpus
from caliper.estimator import fit, subspace_alignment

LAYER, TOKENS, STEPS, SEEDS = 6, 20_000, 800, list(range(10))

d = json.load(open("results/s1_required_n_unbiased.json"))
sel = d["groups"]["worst-fail"] + d["groups"]["pass"][:2]
grp = {n: ("worst-fail" if n in d["groups"]["worst-fail"] else "pass") for n in sel}
gate = {int(r["_key"]): r for r in
        (json.loads(l) for l in open("results/e01_gate.jsonl", encoding="utf-8") if l.strip())}

model, tok = load_model("gpt2")
p = collect(model, tok, sample_corpus(n_docs=300, seed=0), layer=LAYER,
            neurons=np.array(sel), max_tokens=TOKENS, seed=0)
print(f"{len(sel)} units, {len(SEEDS)} seeds, 1 restart, {STEPS} steps\n", flush=True)

out, t0 = [], time.time()
for j, n in enumerate(p.neurons):
    n = int(n)
    for sd in SEEDS:
        f = fit(p.stimulus, p.response[:, j], k=1, n_restarts=1, steps=STEPS, seed=sd)
        a = abs(subspace_alignment(f.subspace, p.weights[:, j][:, None]))
        out.append({"neuron": n, "group": grp[n], "seed": sd,
                    "alignment": round(float(a), 4),
                    "train_r2": round(float(f.train_r2), 6),
                    "test_r2": round(float(f.test_r2), 6)})
    v = [r["alignment"] for r in out if r["neuron"] == n]
    print(f"  n{n} [{grp[n]:<10}] gate {gate[n]['align_direct']:.4f}  "
          f"seeds min {min(v):.4f} med {np.median(v):.4f} max {max(v):.4f}  "
          f"{time.time()-t0:.0f}s", flush=True)

print(f"\n=== THE QUESTION: does held-out R2 pick the good draw, with no ground truth? ===")
print(f"  {'neuron':>7} {'grp':>11} | {'best test_r2 draw':>18} {'oracle best':>12} {'regret':>8}")
tot_r, tot_o = [], []
for n in sel:
    v = [r for r in out if r["neuron"] == n]
    picked = max(v, key=lambda r: r["test_r2"])["alignment"]
    oracle = max(r["alignment"] for r in v)
    tot_r.append(oracle - picked); tot_o.append(oracle)
    print(f"  {n:>7} {grp[n]:>11} | {picked:>18.4f} {oracle:>12.4f} {oracle-picked:>8.4f}")
print(f"\n  median regret of selecting by held-out R2: {np.median(tot_r):.4f}")
print(f"  median oracle-best across seeds:           {np.median(tot_o):.4f}")

wf = [n for n in sel if grp[n] == "worst-fail"]
sel_wf = [max((r for r in out if r["neuron"] == n), key=lambda r: r["test_r2"])["alignment"] for n in wf]
print(f"\n  worst-fail, 10 seeds, keep best test_r2: median {np.median(sel_wf):.4f}")
print(f"  worst-fail, the gate's single direct fit:  median "
      f"{np.median([gate[n]['align_direct'] for n in wf]):.4f}")

corr = np.corrcoef([r["test_r2"] for r in out if r["group"] == "worst-fail"],
                   [r["alignment"] for r in out if r["group"] == "worst-fail"])[0, 1]
print(f"\n  correlation(test_r2, alignment) on worst-fail draws: {corr:+.3f}")

json.dump({"steps": STEPS, "seeds": SEEDS, "rows": out},
          open("results/s1_seed_lottery.json", "w"), indent=2)
print("\nwrote results/s1_seed_lottery.json")
