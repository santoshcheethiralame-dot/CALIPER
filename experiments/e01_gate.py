"""E0.1 gate run at n = 100. Protocol frozen in docs/preregistration-e01-n100.md.

Dual protocol (direct + cascade, selected by held-out R^2) at the E0.3b operating point,
batched, resumable. Pass rate reported as a Wilson interval, because a bare fraction over
20 neurons cannot resolve a 0.95 threshold.
"""
import argparse, json, time
import numpy as np
import torch
import transformers
from pathlib import Path
from caliper.activations import _blocks, _mlp_in, collect, load_model, sample_corpus
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
                     "before 6 Oct). sae: units are latents of the jbloom GPT-2 small residual "
                     "SAE at --layer, reference = the mean-removed encoder column "
                     "(caliper/sae.py, docs/preregistration-tsae-transfer.md)")
ap.add_argument("--sae-min-events", type=int, default=100,
                help="sae only: eligible latents fire at least this many times in the "
                     "stimulus actually fitted (T-SAE prereg, Amendment 1)")
ap.add_argument("--units", default=None,
                help="comma-separated unit ids to fit, overriding the random draw (used by "
                     "operating-point pilots that must fit the same units under two budgets)")
ap.add_argument("--fit-seed", type=int, default=0,
                help="seed for every fit (fit_batch and fit_cascade). 0 reproduces every "
                     "earlier run; B-15 varies it")
ap.add_argument("--corpus-seed", type=int, default=0,
                help="seed for sample_corpus, which decides which tokens make the stimulus. "
                     "0 reproduces every earlier run")
ap.add_argument("--drop-constant-coords", type=float, default=None, metavar="FRAC",
                help="fit without stimulus coordinates whose SD is below FRAC x the median SD, "
                     "then put zeros back in those positions of every fitted direction. "
                     "Per-coordinate standardisation otherwise magnifies the fit's arbitrary "
                     "weight on a near-constant coordinate (B-17b)")
ap.add_argument("--exclude-corpus-seed", type=int, default=None,
                help="drop every document that corpus seed would sample, so this run's "
                     "stimulus is document-disjoint from that seed's (X-1)")
ap.add_argument("--split-seed", type=int, default=0,
                help="seed for collect's row order, which decides the 20%% held-out set. "
                     "0 reproduces every earlier run")
ap.add_argument("--dtype", choices=("fp32", "fp16"), default="fp32",
                help="model precision. fp16 reproduces the local Pythia runs made under "
                     "transformers 5 before 6 Oct 2026, which loaded Pythia's stored float16")
ap.add_argument("--snr", type=float, default=None,
                help="add Gaussian noise to every unit's response at this signal-to-noise "
                     "variance ratio, so the attainable R2 is snr / (1 + snr) instead of 1")
ap.add_argument("--snr-range", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                help="as --snr, but each unit draws its own SNR log-uniformly from [LO, HI], "
                     "so the attainable R2 differs between units and is unknown to the checks")
ap.add_argument("--augment", choices=("targeted", "random", "natural"), default=None,
                help="F-1: add --augment-n samples to the fitting set only. 'targeted': stimulus "
                     "rows pushed into the eigen-directions carrying the bottom 1%% of stimulus "
                     "variance; 'random': pushed isotropically; both answered by the unit's exact "
                     "function. 'natural': fresh tokens from documents disjoint from the corpus. "
                     "The held-out rows are unchanged (docs/preregistration-f1-interventional.md)")
ap.add_argument("--augment-n", type=int, default=2000)
ap.add_argument("--out", default="results/e01_gate.jsonl")
a = ap.parse_args()

t0 = time.time()


def corpus():
    texts = sample_corpus(n_docs=300, seed=a.corpus_seed)
    if a.exclude_corpus_seed is not None:
        seen = set(sample_corpus(n_docs=300, seed=a.exclude_corpus_seed))
        texts = [t for t in texts if t not in seen]
    return texts


device = pick_device(a.device)
ck = Checkpoint(a.out)

model, tok = load_model(a.model, dtype={"fp32": torch.float32, "fp16": torch.float16}[a.dtype])
rng = np.random.default_rng(0)
sae = None
if a.target == "sae":
    from caliper.sae import draw_by_firing, firing_counts, load_sae
    sae = load_sae(a.layer)
    # Two passes: collect the stimulus once, count how often every latent fires on it, then
    # draw evenly across firing-count quartiles. Shuffled once so --neuron-pool prefixes stay
    # a fair sample of every quartile.
    p0 = collect(model, tok, corpus(), layer=a.layer,
                 neurons=np.arange(1), max_tokens=a.tokens, seed=a.split_seed,
                 shuffle="sequence" if a.sequence_split else "token", sae=sae)
    counts = firing_counts(sae, p0.stimulus)
    if a.units:
        neurons = np.array([int(u) for u in a.units.split(",")])
    else:
        pool = draw_by_firing(counts, a.neuron_pool or a.neurons, a.sae_min_events)
        neurons = rng.permutation(pool)[:a.neurons]
elif a.units:
    neurons = np.array([int(u) for u in a.units.split(",")])
elif a.neuron_pool:
    neurons = rng.choice(a.d_mlp, size=a.neuron_pool, replace=False)[:a.neurons]
else:
    neurons = rng.choice(a.d_mlp, size=a.neurons, replace=False)
p = collect(model, tok, corpus(), layer=a.layer,
            neurons=neurons, max_tokens=a.tokens, seed=a.split_seed,
            shuffle="sequence" if a.sequence_split else "token", sae=sae)
snr = {}
if a.snr is not None or a.snr_range is not None:
    # Seeded by unit id, so a unit's noise and SNR do not depend on its batch-mates or on
    # which other units are fitted (N-1, docs/preregistration-n1-response-noise.md).
    sd = p.response.std(0)
    for i, u in enumerate(neurons):
        g = np.random.default_rng([7919, int(u)])
        lo, hi = a.snr_range if a.snr_range is not None else (a.snr, a.snr)
        snr[int(u)] = float(np.exp(g.uniform(np.log(lo), np.log(hi))))
        p.response[:, i] += g.normal(0.0, sd[i] / np.sqrt(snr[int(u)]),
                                     len(p.response)).astype(p.response.dtype)
test_frac = 0.2
if a.augment is not None:
    if snr or a.target != "mlp":
        raise SystemExit("--augment is defined for noiseless MLP units only")
    n_test = int(len(p.stimulus) * 0.2)
    fit_rows = p.stimulus[n_test:]
    if a.augment == "natural":
        seen = set(corpus())
        fresh = [t for t in sample_corpus(n_docs=300, seed=a.corpus_seed + 1) if t not in seen]
        q = collect(model, tok, fresh, layer=a.layer, neurons=neurons,
                    max_tokens=a.augment_n, seed=a.split_seed, shuffle="token")
        extra_s, extra_r = q.stimulus[:a.augment_n], q.response[:a.augment_n]
        if len(extra_s) < a.augment_n:
            raise SystemExit(f"only {len(extra_s)} fresh tokens for --augment-n {a.augment_n}")
    else:
        g = np.random.default_rng(0)
        base = fit_rows[g.choice(len(fit_rows), a.augment_n, replace=False)].astype(np.float64)
        centred = fit_rows - fit_rows.mean(0)
        if a.augment == "targeted":
            lam, vec = np.linalg.eigh(np.cov(fit_rows, rowvar=False))
            low = np.cumsum(lam) / lam.sum() <= 0.01
            d = g.standard_normal((a.augment_n, int(low.sum()))) @ vec[:, low].T
        else:
            d = g.standard_normal((a.augment_n, fit_rows.shape[1]))
        d *= np.median(np.linalg.norm(centred, axis=1)) / np.linalg.norm(d, axis=1, keepdims=True)
        extra_s = (base + d).astype(p.stimulus.dtype)
        block = _blocks(model)[a.layer]
        with torch.no_grad():
            pre = _mlp_in(block)(torch.as_tensor(extra_s, dtype=next(model.parameters()).dtype))
            extra_r = torch.nn.functional.gelu(pre.float())[:, neurons].numpy().astype(
                p.response.dtype)
        print(f"  augment {a.augment}: {a.augment_n} rows"
              + (f", {int(low.sum())} low-variance directions" if a.augment == "targeted" else ""),
              flush=True)
    p.stimulus = np.concatenate([p.stimulus, extra_s])
    p.response = np.concatenate([p.response, extra_r])
    test_frac = n_test / len(p.stimulus)
    assert int(len(p.stimulus) * test_frac) == n_test
dirs = None if a.no_save_directions else Path(a.out.replace(".jsonl", "_dirs"))
if dirs is not None:
    dirs.mkdir(parents=True, exist_ok=True)
print(f"  stimulus {p.stimulus.shape}  ({time.time()-t0:.0f}s)", flush=True)

S_fit, pad = p.stimulus, (lambda V: V)
if a.drop_constant_coords is not None:
    sd = p.stimulus.std(0)
    keep = sd >= a.drop_constant_coords * np.median(sd)
    S_fit = p.stimulus[:, keep]

    def pad(V):
        V = np.asarray(V)
        full = np.zeros((len(keep),) + V.shape[1:], dtype=V.dtype)
        full[keep] = V
        return full
    print(f"  dropped {int((~keep).sum())} near-constant coordinate(s) "
          f"(SD < {a.drop_constant_coords} x median)", flush=True)

alive = p.response.std(0) > 1e-4
print(f"  alive: {int(alive.sum())}/{len(alive)} (screened units are reported, not dropped)",
      flush=True)

todo = [i for i in range(len(p.neurons)) if alive[i] and not ck.done(int(p.neurons[i]))]
for start in range(0, len(todo), a.batch):
    idx = todo[start:start + a.batch]
    Y = p.response[:, idx]
    indep = dict(unit_ids=p.neurons[idx], per_neuron_stop=True) if a.independent_units else {}
    d1 = fit_batch(S_fit, Y, k=1, n_restarts=a.restarts, steps=a.steps, test_frac=test_frac,
                   seed=a.fit_seed, device=device, per_neuron_seed=a.per_neuron_seed, **indep)
    d2 = fit_batch(S_fit, Y, k=2, n_restarts=a.restarts, steps=a.steps, test_frac=test_frac,
                   seed=a.fit_seed, device=device, per_neuron_seed=a.per_neuron_seed, **indep)
    for res in list(d1) + list(d2):
        res.subspace = pad(res.subspace)
        res.restarts = [pad(q) for q in res.restarts]
    for j, i in enumerate(idx):
        n = int(p.neurons[i])
        wu = p.weights[:, i] / np.linalg.norm(p.weights[:, i])
        c = fit_cascade(S_fit, p.response[:, i], k=1, test_frac=test_frac,
                        n_restarts=a.restarts, steps=a.steps, seed=a.fit_seed)
        c.subspace = pad(c.subspace)
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
            **({"snr": round(snr[int(n)], 4)} if snr else {}),
            **({"augment": a.augment, "augment_n": a.augment_n} if a.augment else {}),
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
            **({"log_density": round(float(sae["log_density"][n]), 4),
                "firing_events": int(counts[n])} if sae is not None else {}),
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
    "augment": a.augment, "augment_n": a.augment_n if a.augment else None,
    "dtype": str(next(model.parameters()).dtype).replace("torch.", ""),
    "torch_threads": torch.get_num_threads(),
    "libraries": {"torch": torch.__version__, "transformers": transformers.__version__},
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
