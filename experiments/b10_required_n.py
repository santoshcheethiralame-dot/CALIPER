"""B-10 - does each reliability signal's AUC depend on how much data it was given?

*Objection answered:* "your operating point is arbitrary." The bench runs at 8,000
tokens because E0.3b chose it, and every AUC in the notebook is quoted there. If a
signal only works at that one budget, the operating point is doing the work and the
recommendation is an artefact.

Question, two parts:

  1. does each signal's AUC stabilise as N grows?
  2. does the RANKING of signals change with N? A signal that wins only at 8k is a
     different recommendation from one that wins everywhere.

Design notes that are load-bearing:

  * One stimulus pass at 16,000 tokens, then N = 2k/4k/8k/16k as nested prefixes of
    it. `collect` shuffles once with a seeded rng, so the prefix at 2,000 is a random
    subset of the 16,000 - the sweep adds data without also changing the corpus.
  * --neuron-pool 300 makes the 50 units the first 50 of B-1b's 300, so the 8,000-token
    leg reproduces the primary arm's units at its own operating point and the sweep is
    anchored to an already-analysed run instead of floating beside it.
  * 2 restarts, matching B-1b, for the same reason Addendum 1 gives: holding the
    optimiser fixed is what makes the AUCs comparable, and the 8k leg is only a
    consistency check if it is the same configuration.

POWER, stated before running. At the 17% failure rate measured in B-1b, 50 units give
roughly 8-9 failures. An AUC on 8 positives carries a standard error near 0.10, so a
ranking difference smaller than about 0.15 is not resolvable at any N on this grid.
Every AUC here is printed with a bootstrap CI for that reason, and "the ranking changed"
is only claimed where the CIs on two signals exclude each other. The sweep can still
answer "does the ranking hold", which is the question that matters; it cannot resolve
small re-orderings, and that limit is reported rather than hidden.
"""
import argparse
import json
import time

import numpy as np

from b1_signal_calibration import PASS, delong, roc, wilson
from caliper.activations import collect, load_model, sample_corpus
from caliper.batched import fit_batch
from caliper.estimator import fit_cascade, subspace_alignment
from caliper.runtime import Checkpoint, pick_device

N_GRID = [2000, 4000, 8000, 16000]
# The four signals B-1b reported an AUC for, in its orientation: higher = more suspect.
SIGNALS = {
    "held-out R2": lambda r: -r["r2_k1"],
    "restart agreement": lambda r: -r["stability"],
    "disagreement": lambda r: r["disagreement"],
    "restart R2 spread": lambda r: r["r2_spread"],
}

ap = argparse.ArgumentParser()
ap.add_argument("--neurons", type=int, default=50)
ap.add_argument("--neuron-pool", type=int, default=300)
ap.add_argument("--restarts", type=int, default=2)
ap.add_argument("--steps", type=int, default=1600)
ap.add_argument("--layer", type=int, default=6)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--model", default="gpt2")
ap.add_argument("--d-mlp", type=int, default=3072)
ap.add_argument("--n-grid", type=int, nargs="+", default=N_GRID)
ap.add_argument("--device", default="auto")
ap.add_argument("--out", default="results/b10_required_n.jsonl")
a = ap.parse_args()

t0 = time.time()
device = pick_device(a.device)
ck = Checkpoint(a.out)

model, tok = load_model(a.model)
rng = np.random.default_rng(0)
neurons = rng.choice(a.d_mlp, size=a.neuron_pool, replace=False)[:a.neurons]
p = collect(model, tok, sample_corpus(n_docs=300, seed=0), layer=a.layer,
            neurons=neurons, max_tokens=max(a.n_grid), seed=0)
print(f"  stimulus {p.stimulus.shape}  ({time.time()-t0:.0f}s)", flush=True)
del model, tok  # the fit is the expensive part; free ~500 MB before it starts

alive = p.response.std(0) > 1e-4
print(f"  alive: {int(alive.sum())}/{len(alive)}", flush=True)

print(f"\n{'tokens':>7} {'pass':>7} {'rate':>7} {'fail':>5}  "
      + "  ".join(f"{k:>18}" for k in SIGNALS))
for n_tok in a.n_grid:
    X = p.stimulus[:n_tok]
    todo = [i for i in range(len(p.neurons))
            if alive[i] and not ck.done(f"{n_tok}_{int(p.neurons[i])}")]
    for start in range(0, len(todo), a.batch):
        idx = todo[start:start + a.batch]
        Y = p.response[:n_tok][:, idx]
        d1 = fit_batch(X, Y, k=1, n_restarts=a.restarts, steps=a.steps,
                       seed=0, device=device)
        d2 = fit_batch(X, Y, k=2, n_restarts=a.restarts, steps=a.steps,
                       seed=0, device=device)
        for j, i in enumerate(idx):
            wu = p.weights[:, i] / np.linalg.norm(p.weights[:, i])
            c = fit_cascade(X, p.response[:n_tok][:, i], k=1,
                            n_restarts=a.restarts, steps=a.steps, seed=0)
            ad = abs(subspace_alignment(d1[j].subspace, wu[:, None]))
            ac = abs(subspace_alignment(c.subspace, wu[:, None]))
            use_cascade = c.test_r2 > d1[j].test_r2
            sel = ac if use_cascade else ad
            ck.record(f"{n_tok}_{int(p.neurons[i])}", {
                "tokens": n_tok, "neuron": int(p.neurons[i]),
                "align_direct": round(ad, 4), "align_cascade": round(ac, 4),
                "align_selected": round(sel, 4),
                "best_available": round(max(ad, ac), 4),
                "picked": "cascade" if use_cascade else "direct",
                "r2_k1": round(float(max(d1[j].test_r2, c.test_r2)), 6),
                "k2_gain": round(float(d2[j].test_r2 - max(d1[j].test_r2, c.test_r2)), 4),
                "disagreement": round(abs(ad - ac), 4),
                "stability": round(float(d1[j].stability), 4),
                "r2_spread": round(float(d1[j].r2_spread), 6),
                "stability_pairs": d1[j].stability_pairs,
                "n_restarts": a.restarts,
            })
        el = time.time() - t0
        print(f"  {n_tok:>5}tok {len(ck.rows()):>5}/{a.neurons * len(a.n_grid)} rows  "
              f"{el:.0f}s ({el/max(len(ck.rows()),1):.1f}s each)", flush=True)


def boot_auc_ci(scores, labels, n_boot=2000, seed=0):
    """Stratified bootstrap CI on the AUC. Resamples units within failure/pass, which
    is what the AUC's sampling unit is; an unstratified resample can drop a class and
    silently return nan on small failure counts."""
    rng = np.random.default_rng(seed)
    pos = [i for i, y in enumerate(labels) if y]
    neg = [i for i, y in enumerate(labels) if not y]
    if len(pos) < 2 or len(neg) < 2:
        return (float("nan"), float("nan"))
    out = []
    for _ in range(n_boot):
        bi = ([pos[i] for i in rng.integers(0, len(pos), len(pos))]
              + [neg[i] for i in rng.integers(0, len(neg), len(neg))])
        _, auc = roc([scores[i] for i in bi], [labels[i] for i in bi])
        out.append(auc)
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)))


rows = ck.rows()
summary = {"out": a.out, "neurons": a.neurons, "restarts": a.restarts,
           "steps": a.steps, "model": a.model, "layer": a.layer, "by_tokens": {}}

for n_tok in a.n_grid:
    sub = [r for r in rows if r["tokens"] == n_tok]
    if not sub:
        continue
    fail = [r["align_selected"] < PASS for r in sub]
    nf, n = sum(fail), len(sub)
    lo, hi = wilson(n - nf, n)
    entry = {"n_units": n, "n_failures": nf, "pass_rate": round((n - nf) / n, 4),
             "wilson_95": [round(lo, 4), round(hi, 4)], "auc": {}, "ranking": []}
    if 0 < nf < n:
        for name, fn in SIGNALS.items():
            sc = [float(fn(r)) for r in sub]
            _, auc = roc(sc, fail)
            clo, chi = boot_auc_ci(sc, fail)
            entry["auc"][name] = {
                "auc": round(auc, 4) if auc == auc else None,
                "ci95": [round(clo, 4) if clo == clo else None,
                         round(chi, 4) if chi == chi else None],
                "n_failures": nf}
        entry["ranking"] = [k for k, _ in
                            sorted(entry["auc"].items(),
                                   key=lambda kv: -(kv[1]["auc"] if kv[1]["auc"] is not None
                                                   else -1))]
        # The primary contrast, at every N: is held-out R2 still ahead of restart
        # agreement once the data grows? B-1b answered yes at 8k with z = -5.22.
        r2 = [float(SIGNALS["held-out R2"](r)) for r in sub]
        st = [float(SIGNALS["restart agreement"](r)) for r in sub]
        d = delong(st, r2, fail)
        fin = lambda x: (float(x) if x == x else None)  # NaN is not valid JSON
        entry["restart_minus_r2"] = {
            "diff": round(fin(d[2]), 4) if fin(d[2]) is not None else None,
            "se": round(fin(d[3]), 4) if fin(d[3]) is not None else None,
            "z": round(fin(d[4]), 3) if fin(d[4]) is not None else None,
            "p": float(f"{d[5]:.3g}") if fin(d[5]) is not None else None,
            "n_failures": nf,
            "underpowered": nf < 20,
        }
    summary["by_tokens"][str(n_tok)] = entry

    if entry["auc"]:
        cells = "  ".join(
            f"{v['auc']:.3f} [{v['ci95'][0]:.2f},{v['ci95'][1]:.2f}]"
            if v["auc"] is not None and v["ci95"][0] is not None
            else f"{v['auc']} [ci unavailable]"
            for v in entry["auc"].values())
        print(f"{n_tok:>7} {n-nf:>3}/{n:<3} {(n-nf)/n:>7.3f} {nf:>5}  {cells}")
    else:
        print(f"{n_tok:>7} {n-nf:>3}/{n:<3} {(n-nf)/n:>7.3f} {nf:>5}  "
              f"(no contrast: {nf} failures)")

# Ranking stability is the deliverable, so state it as such.
print("\n  ranking by AUC, per token count:")
for k, v in summary["by_tokens"].items():
    if v["ranking"]:
        print(f"    {int(k):>6}tok: " + " > ".join(v["ranking"]))
print("\n  AUC CIs overlap heavily at this n. A ranking difference inside the CIs is "
      "NOT reported as a change of ranking.")

json.dump(summary, open(a.out.replace(".jsonl", "_summary.json"), "w"), indent=2)
print(f"\n  wrote {a.out.replace('.jsonl', '_summary.json')}   "
      f"total {time.time()-t0:.0f}s")