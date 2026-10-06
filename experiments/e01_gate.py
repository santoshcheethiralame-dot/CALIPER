"""E0.1 gate run at n = 100. Protocol frozen in docs/preregistration-e01-n100.md.

Dual protocol (direct + cascade, selected by held-out R^2) at the E0.3b operating point,
batched, resumable. Pass rate reported as a Wilson interval, because a bare fraction over
20 neurons cannot resolve a 0.95 threshold.
"""
import argparse, json, time
import numpy as np
from pathlib import Path
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
ap.add_argument("--restarts", type=int, default=2,
                help="B-1 calibrates restart agreement and needs 5; 2 gives a "
                     "single pairwise comparison, too thin to calibrate. C13 and "
                     "C53 both ran at 2 and their pass rates are quoted at that "
                     "configuration, not this one")
ap.add_argument("--layer", type=int, default=6)
ap.add_argument("--batch", type=int, default=32)        # speedup saturates by 32
ap.add_argument("--device", default="auto")
ap.add_argument("--model", default="gpt2",
                help="HF id. gpt2 is C13; EleutherAI/pythia-160m is the "
                     "second-family replication, prereg docs/preregistration-e01-pythia.md")
ap.add_argument("--d-mlp", type=int, default=3072,
                help="units to draw from; 3072 for both gpt2 and pythia-160m")
ap.add_argument("--neuron-pool", type=int, default=0,
                help="draw this many units and keep the first --neurons. numpy's "
                     "choice(replace=False) is NOT nested in size, so without a pool "
                     "each --neurons value is a disjoint draw: a layer sweep would "
                     "compare different units at every depth. --neuron-pool 300 makes "
                     "any n<=300 a prefix of one ordering, which is exactly the n=300 "
                     "draw B-1b used, so smaller runs nest inside the primary arm "
                     "instead of beside it. Default 0 keeps the legacy draw so no "
                     "completed run changes meaning.")
ap.add_argument("--per-neuron-seed", action="store_true",
                help="seed each unit's initialisation from its own index instead of "
                     "one stacked draw, so a unit's result stops depending on how many "
                     "units share its batch. Opt-in: it changes every number, so runs "
                     "that must stay poolable with B-1/B-1b/B-2b leave it off.")
ap.add_argument("--independent-units", action="store_true",
                help="seed each unit from its neuron id (not its position in the batch) "
                     "and give it its own early-stopping counter, so its fit is the one it "
                     "would get alone. --per-neuron-seed alone fixes neither: it seeds by "
                     "position, and the default stopping rule is shared across the batch. "
                     "Opt-in for the same reason: it changes every number.")
ap.add_argument("--no-save-directions", action="store_true",
                help="skip writing the fitted directions. By default every unit's direct, "
                     "cascade, k=2 and per-restart directions go to <out>_dirs/n<id>.npz, "
                     "because alignments alone cannot be re-scored against a different "
                     "target (the identifiable part of w, or a functional correlation) - "
                     "the gap B-14's analysis had to log as a deviation. Saving changes no "
                     "number.")
ap.add_argument("--sequence-split", action="store_true",
                help="shuffle whole sequences, not tokens, before fit_batch takes its first "
                     "20%% as the held-out set, so held-out tokens never share a sequence "
                     "with training tokens. Opt-in: it changes every held-out R2.")
ap.add_argument("--target", choices=("mlp", "sae"), default="mlp",
                help="mlp: units are MLP neurons, reference = input weight column (every run "
                     "before 7 Oct). sae: units are latents of the jbloom GPT-2 small residual "
                     "SAE at --layer, reference = the mean-removed encoder column "
                     "(caliper/sae.py, docs/preregistration-tsae-transfer.md)")
ap.add_argument("--sae-min-log-density", type=float, default=-3.0,
                help="sae only: exclude latents firing on fewer than 10^x of tokens")
ap.add_argument("--fit-seed", type=int, default=0,
                help="seed for every fit (fit_batch and fit_cascade). 0 reproduces every "
                     "earlier run; B-15 varies it")
ap.add_argument("--corpus-seed", type=int, default=0,
                help="seed for sample_corpus, which decides which tokens make the stimulus. "
                     "0 reproduces every earlier run")
ap.add_argument("--split-seed", type=int, default=0,
                help="seed for collect's row order, which decides the 20%% held-out set. "
                     "0 reproduces every earlier run")
ap.add_argument("--out", default="results/e01_gate.jsonl")
a = ap.parse_args()

t0 = time.time()
device = pick_device(a.device)
ck = Checkpoint(a.out)

model, tok = load_model(a.model)
rng = np.random.default_rng(0)
sae = None
if a.target == "sae":
    from caliper.sae import draw_latents, load_sae
    sae = load_sae(a.layer)
    # Drawn evenly across density quartiles, then shuffled once so that --neuron-pool
    # prefixes stay a fair sample of every quartile.
    pool = draw_latents(sae, a.neuron_pool or a.neurons, a.sae_min_log_density)
    neurons = rng.permutation(pool)[:a.neurons]
elif a.neuron_pool:
    neurons = rng.choice(a.d_mlp, size=a.neuron_pool, replace=False)[:a.neurons]
else:
    neurons = rng.choice(a.d_mlp, size=a.neurons, replace=False)
p = collect(model, tok, sample_corpus(n_docs=300, seed=a.corpus_seed), layer=a.layer,
            neurons=neurons, max_tokens=a.tokens, seed=a.split_seed,
            shuffle="sequence" if a.sequence_split else "token", sae=sae)
dirs = None if a.no_save_directions else Path(a.out.replace(".jsonl", "_dirs"))
if dirs is not None:
    dirs.mkdir(parents=True, exist_ok=True)
print(f"  stimulus {p.stimulus.shape}  ({time.time()-t0:.0f}s)", flush=True)

alive = p.response.std(0) > 1e-4
print(f"  alive: {int(alive.sum())}/{len(alive)} (screened units are reported, not dropped)",
      flush=True)

todo = [i for i in range(len(p.neurons)) if alive[i] and not ck.done(int(p.neurons[i]))]
for start in range(0, len(todo), a.batch):
    idx = todo[start:start + a.batch]
    Y = p.response[:, idx]
    indep = dict(unit_ids=p.neurons[idx], per_neuron_stop=True) if a.independent_units else {}
    d1 = fit_batch(p.stimulus, Y, k=1, n_restarts=a.restarts, steps=a.steps,
                   seed=a.fit_seed, device=device, per_neuron_seed=a.per_neuron_seed, **indep)
    d2 = fit_batch(p.stimulus, Y, k=2, n_restarts=a.restarts, steps=a.steps,
                   seed=a.fit_seed, device=device, per_neuron_seed=a.per_neuron_seed, **indep)
    for j, i in enumerate(idx):
        n = int(p.neurons[i])
        wu = p.weights[:, i] / np.linalg.norm(p.weights[:, i])
        c = fit_cascade(p.stimulus, p.response[:, i], k=1,
                        n_restarts=a.restarts, steps=a.steps, seed=a.fit_seed)
        ad = abs(subspace_alignment(d1[j].subspace, wu[:, None]))
        ac = abs(subspace_alignment(c.subspace, wu[:, None]))
        use_cascade = c.test_r2 > d1[j].test_r2
        sel = ac if use_cascade else ad
        if dirs is not None:
            # Written before the row, so a recorded row always has its directions.
            np.savez_compressed(
                dirs / f"n{n}.npz", w=p.weights[:, i].astype(np.float32),
                direct=d1[j].subspace.astype(np.float32),
                cascade=c.subspace.astype(np.float32),
                k2=d2[j].subspace.astype(np.float32),
                direct_restarts=np.stack([np.asarray(q, dtype=np.float32)
                                          for q in d1[j].restarts]),
                direct_r2_restarts=np.asarray(d1[j].r2_restarts, dtype=np.float32))
        ck.record(n, {
            "align_direct": round(ad, 4), "align_cascade": round(ac, 4),
            "align_selected": round(sel, 4), "best_available": round(max(ad, ac), 4),
            "picked": "cascade" if use_cascade else "direct",
            "r2_k1": round(float(max(d1[j].test_r2, c.test_r2)), 6),
            "k2_gain": round(float(d2[j].test_r2 - max(d1[j].test_r2, c.test_r2)), 4),
            # NOT ground-truth-free: both terms are alignments to w. Kept so every
            # earlier run stays comparable, but it must not be reported as a check a
            # practitioner could compute. route_agreement is the ground-truth-free
            # version: how far the two routes' own directions agree with each other.
            "disagreement": round(abs(ad - ac), 4),
            "route_agreement": round(abs(subspace_alignment(d1[j].subspace, c.subspace)), 4),
            # B-1: the field's default reliability check. The fit has always
            # computed it; nothing wrote it down. Only the direct route has a
            # restart lottery to agree about - fit_cascade runs a grid search and
            # one polish, so its stability is undefined by construction, and that
            # asymmetry is itself part of why the cascade is the better route.
            "stability": round(float(d1[j].stability), 4),
            "r2_spread": round(float(d1[j].r2_spread), 6),
            # Secondary 4 of the B-1 filing: does a cheap 2-restart stability
            # estimate agree with the 5-restart one? Only answerable offline if
            # the pairs survive the run, so keep them, not just their median.
            "stability_pairs": d1[j].stability_pairs,
            "n_restarts": a.restarts,
            **({"log_density": round(float(sae["log_density"][n]), 4)} if sae is not None else {}),
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
    "out": a.out,
    "model": a.model, "layer": a.layer, "neurons": a.neurons,
    "n_restarts": a.restarts, "steps": a.steps, "batch": a.batch,
    "tokens": a.tokens, "per_neuron_seed": bool(a.per_neuron_seed),
    "independent_units": bool(a.independent_units),
    "target": a.target, "fit_seed": a.fit_seed, "corpus_seed": a.corpus_seed,
    "split_seed": a.split_seed,
    "sequence_split": bool(a.sequence_split),
    "directions_dir": str(dirs) if dirs is not None else None,
    "neuron_pool": a.neuron_pool,
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
# Keyed off --out: a fixed path meant every run in the queue overwrote the last
# one's summary, so B-8 or a layer leg would silently replace B-2b's headline.
summary_path = a.out.replace(".jsonl", "_summary.json")
json.dump(summary, open(summary_path, "w"), indent=2)
print("\n" + "=" * 66)
for k, v in summary.items():
    print(f"  {k:.<44} {v}")
print("=" * 66)
print("  Criterion (pre-registered): Wilson 95% lower bound > 0.90")
