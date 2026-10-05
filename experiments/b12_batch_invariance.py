"""B-12 - does batch size still change the answer once each unit is seeded on its own?

The question, from B-0's n=50 extension: holding units, seeds and device fixed and
changing only how many units share a batch moved 5 of 16 units across the pass/fail
bar on CPU. The mechanism was located exactly - `batched.py` drew one stacked
`randn(n, d, k)` per restart, so unit i's MLP initialisation depended on n - and this
run is the test of the fix rather than of the bug.

Design: the same 50 units at three batch sizes, `--per-neuron-seed` on, everything
else identical. Invoked as subprocesses of e01_gate.py so the protocol under test is
literally the production one, not a re-implementation that might differ from it.

Criterion, filed before running: batch size must not move a unit across the pass/fail bar
by more than RESTART COUNT ALREADY DOES ON THE SAME UNITS. The original criterion was
"zero flips", which is not a bar this instrument can be held to: comparing B-7 layer 6 at
5 restarts against B-1b layer 6 at 2 restarts, on the same 50 units, **10 of 50 verdicts
flip (20%) with nothing changed except the restart count.** A criterion stricter than the
instrument's own repeatability cannot distinguish a residual batch effect from noise, and
the "flips remain" branch below would fire on a false positive.

So this run measures its own floor instead of importing one. A fourth arm repeats the
reference batch size at a different restart count, on the same units in the same session,
which is the closest available stand-in for "same estimator, different randomness". Batch
flips are then judged against control flips. `--control-restarts 0` disables the arm and
restores the original zero-flip criterion, for anyone who wants the strict version.

The floor is also cross-checked against the external 20% (B-7 vs B-1b), and against
`--per-neuron-seed 0`, which reproduces the unfixed defect as a positive control: without
it, a zero-flip result cannot be told apart from a batch size too small to matter.

Both branches are reportable and the second is the more interesting one:

  * flips within the control floor -> the defect is closed, batching is safe, and the
    published numbers need only the disclosure already in the notebook.
  * flips beyond the control floor -> they cannot be init, because the init is now bitwise
    identical across batch sizes (verified directly on the parameter tensors). What is left
    is float32 reduction order in the batched GEMM, which would make silent estimation
    failure sensitive to how the estimator was implemented. That is a larger claim than
    "batching changes the answer" and it constrains reproduction everywhere.

Cost note: batch 1 is the expensive arm - it forfeits the shared-stimulus GEMM that
makes batching worth doing, so per-neuron time rises several-fold. The queue runs
batch 32 first so a usable partial result exists early. The control arm runs at the
reference batch size, which is the cheap end, so it adds roughly the cost of one fast arm
multiplied by the restart ratio.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

ap = argparse.ArgumentParser()
ap.add_argument("--neurons", type=int, default=50)
ap.add_argument("--neuron-pool", type=int, default=300)
ap.add_argument("--restarts", type=int, default=2,
                help="2, matching B-1b, so these units stay poolable with the primary "
                     "arm. The batch effect is present at 2 restarts - that is where the "
                     "random MLP draws that carry it actually happen.")
ap.add_argument("--steps", type=int, default=1600)
ap.add_argument("--layer", type=int, default=6)
ap.add_argument("--batches", type=int, nargs="+", default=[32, 8, 1])
ap.add_argument("--per-neuron-seed", type=int, default=1,
                help="1 re-tests with the fix on (the primary question). 0 reproduces "
                     "the unfixed defect as a positive control - without it, a zero-flip "
                     "result cannot be distinguished from a batch size too small to matter.")
ap.add_argument("--out", default="results/b12_batch_invariance.json")
ap.add_argument("--control-restarts", type=int, default=5,
                help="restarts for the noise-floor control arm, run at the reference batch "
                     "size on the same units. This is what the batch-size flips are judged "
                     "against, because the estimator's verdicts are not reproducible to "
                     "better than ~20%% under a restart change alone (B-7 L6 vs B-1b L2). "
                     "0 disables the arm and restores the original zero-flip criterion.")
ap.add_argument("--external-floor", type=float, default=0.20,
                help="independently measured restart-only flip rate, reported alongside the "
                     "in-run control so the two can be compared rather than one trusted")
ap.add_argument("--results-dir", default=None,
                help="where the per-arm jsonl files live. Defaults to results/. Overridable "
                     "so the comparison logic can be exercised against synthetic arms "
                     "without writing files the real queue would then treat as complete.")
a = ap.parse_args()

RESULTS_DIR = Path(a.results_dir) if a.results_dir else ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# e01_gate.py imports `caliper`, which resolves only when ROOT is importable. The queue
# exports PYTHONPATH so this never bites there, but a direct `python b12_...py` then dies
# with ModuleNotFoundError on every arm - the exact failure the notebook records. Passing
# it explicitly makes the script self-sufficient instead of caller-dependent.
ENV = {**os.environ, "PYTHONPATH": str(ROOT)}

sys.path.insert(0, str(ROOT / "experiments"))
from b1_signal_calibration import PASS, wilson  # noqa: E402

arms = {}
for bs in a.batches:
    tag = f"{'fix' if a.per_neuron_seed else 'legacy'}_b{bs:03d}"
    out = RESULTS_DIR / f"b12_{tag}.jsonl"
    have = sum(1 for l in open(out, encoding="utf-8") if l.strip()) if out.exists() else 0
    if have >= a.neurons:
        print(f"[{time.strftime('%H:%M')}] {tag} already complete - skipping", flush=True)
    else:
        cmd = [sys.executable, "experiments/e01_gate.py",
               "--neurons", str(a.neurons), "--neuron-pool", str(a.neuron_pool),
               "--restarts", str(a.restarts), "--steps", str(a.steps),
               "--layer", str(a.layer), "--batch", str(bs),
               "--out", str(out)]
        if a.per_neuron_seed:
            cmd.append("--per-neuron-seed")
        print(f"[{time.strftime('%H:%M')}] {tag}: {' '.join(cmd)}", flush=True)
        t = time.time()
        rc = subprocess.call(cmd, cwd=ROOT, env=ENV)
        print(f"[{time.strftime('%H:%M')}] {tag} exit {rc}  ({time.time()-t:.0f}s)",
              flush=True)
        if rc != 0:
            raise SystemExit(f"{tag} failed with exit {rc}")
    arms[bs] = {int(r["_key"]): r for r in
                (json.loads(l) for l in open(out, encoding="utf-8") if l.strip())}

# Compare every arm against the largest, which is the production batching.
ref_bs = max(a.batches)
ref = arms[ref_bs]

# The noise-floor control: identical batch size, different restart count, same units. This
# is the only comparison that measures what the estimator does when *nothing* about the
# protocol changes, so it is the only fair bar for the batch-size flips below.
control = None
if a.control_restarts and a.control_restarts != a.restarts:
    tag = f"ctl_b{ref_bs:03d}_r{a.control_restarts}"
    out = RESULTS_DIR / f"b12_{tag}.jsonl"
    have = sum(1 for l in open(out, encoding="utf-8") if l.strip()) if out.exists() else 0
    if have >= a.neurons:
        print(f"[{time.strftime('%H:%M')}] {tag} already complete - skipping", flush=True)
    else:
        cmd = [sys.executable, "experiments/e01_gate.py",
               "--neurons", str(a.neurons), "--neuron-pool", str(a.neuron_pool),
               "--restarts", str(a.control_restarts), "--steps", str(a.steps),
               "--layer", str(a.layer), "--batch", str(ref_bs),
               "--out", str(out)]
        if a.per_neuron_seed:
            cmd.append("--per-neuron-seed")
        print(f"[{time.strftime('%H:%M')}] {tag}: noise-floor control, {' '.join(cmd)}",
              flush=True)
        t = time.time()
        rc = subprocess.call(cmd, cwd=ROOT, env=ENV)
        print(f"[{time.strftime('%H:%M')}] {tag} exit {rc}  ({time.time()-t:.0f}s)",
              flush=True)
        if rc != 0:
            raise SystemExit(f"{tag} failed with exit {rc}")
    control = {int(r["_key"]): r for r in
               (json.loads(l) for l in open(out, encoding="utf-8") if l.strip())}

report = {"per_neuron_seed": bool(a.per_neuron_seed), "neurons": a.neurons,
          "restarts": a.restarts, "steps": a.steps, "layer": a.layer,
          "reference_batch": ref_bs, "comparisons": {}}

if control:
    common = sorted(set(control) & set(ref))
    cflips = [(u, control[u]["align_selected"], ref[u]["align_selected"])
              for u in common
              if (control[u]["align_selected"] > PASS) != (ref[u]["align_selected"] > PASS)]
    cd = np.abs(np.array([control[u]["align_selected"] for u in common])
                - np.array([ref[u]["align_selected"] for u in common]))
    clo, chi = wilson(len(common) - len(cflips), len(common))
    report["noise_floor"] = {
        "kind": "restart-only, same batch size, same units, same session",
        "control_restarts": a.control_restarts, "reference_restarts": a.restarts,
        "n_units": len(common), "n_flips": len(cflips), "flips": cflips,
        "flip_rate": round(len(cflips) / len(common), 4) if common else None,
        "wilson_95": [round(clo, 4), round(chi, 4)],
        "median_abs_delta": round(float(np.median(cd)), 8) if common else None,
        "max_abs_delta": round(float(cd.max()), 8) if common else None,
    }
    report["external_floor"] = {
        "kind": "restart-only, B-7 L6 (r=5) vs B-1b L6 (r=2), 50 shared units",
        "flip_rate": a.external_floor,
    }
    print(f"\n  noise floor (restart {a.restarts} -> {a.control_restarts} at batch {ref_bs}, "
          f"same units): {len(cflips)}/{len(common)} flip  Wilson 95 [{clo:.3f}, {chi:.3f}]")
    print(f"    |delta alignment|  median {np.median(cd):.3e}  max {cd.max():.3e}")
    print(f"    external cross-run floor for comparison: {a.external_floor:.0%}")

for bs in a.batches:
    if bs == ref_bs:
        continue
    common = sorted(set(arms[bs]) & set(ref))
    got = [arms[bs][u]["align_selected"] for u in common]
    want = [ref[u]["align_selected"] for u in common]
    d = np.abs(np.array(got) - np.array(want))
    flips = [(u, arms[bs][u]["align_selected"], ref[u]["align_selected"])
             for u in common
             if (arms[bs][u]["align_selected"] > PASS) != (ref[u]["align_selected"] > PASS)]
    n = len(common)
    lo, hi = wilson(n - len(flips), n)
    name = "fix" if a.per_neuron_seed else "legacy"
    report["comparisons"][f"b{bs}_vs_b{ref_bs}"] = {
        "n_units": n, "n_flips": len(flips), "flips": flips,
        "flip_rate": round(len(flips) / n, 4) if n else None,
        "wilson_95": [round(lo, 4), round(hi, 4)],
        "median_abs_delta": round(float(np.median(d)), 8) if n else None,
        "p99_abs_delta": round(float(np.percentile(d, 99)), 8) if n else None,
        "max_abs_delta": round(float(d.max()), 8) if n else None,
        "pass": len(flips) == 0,
    }
    print(f"\n  batch {bs} vs {ref_bs}: {len(flips)}/{n} units flip pass/fail  "
          f"Wilson 95 [{lo:.3f}, {hi:.3f}]")
    print(f"    |delta alignment|  median {np.median(d):.3e}  "
          f"p99 {np.percentile(d,99):.3e}  max {d.max():.3e}")
    for u, x, y in flips:
        print(f"    unit {u}: {x:.4f} vs {y:.4f}")

total_flips = sum(c["n_flips"] for c in report["comparisons"].values())
report["n_flips_total"] = total_flips

floor = report.get("noise_floor", {}).get("n_flips")
# Each small-batch arm is one comparison against the reference, exactly as the control is
# one comparison against the reference, so each is judged against the floor on its own.
# The first version tested the SUM over arms against a single comparison's floor, which
# fires FAIL whenever two arms each sit at the floor. The sum is still reported.
worst = max((c["n_flips"] for c in report["comparisons"].values()), default=0)
report["n_flips_worst_arm"] = worst
if control is not None:
    for c in report["comparisons"].values():
        c["within_floor"] = c["n_flips"] <= floor
if control is None:
    report["VERDICT"] = (
        "PASS - zero flips, batch size no longer changes the answer"
        if total_flips == 0 else
        "FAIL - flips remain with per-neuron seeding on, so the residual is reduction "
        "order (no control arm run, so this is the strict zero-flip criterion)")
elif worst <= floor:
    report["VERDICT"] = (
        f"PASS - every batch-size arm flips at most {worst} units, within the {floor}-flip "
        f"noise floor measured by changing restart count alone on the same units")
else:
    report["VERDICT"] = (
        f"FAIL - the worst batch-size arm flips {worst} units, beyond the {floor}-flip floor "
        f"from restart changes alone. Before reading this as reduction order: arms run "
        f"without --independent-units still share early stopping across the batch and seed "
        f"by position, so batch size changes training length and initialisation too")

print(f"\n  VERDICT: {report['VERDICT']}")
suffix = f" vs noise floor {floor}" if floor is not None else ", strict zero-flip criterion"
print(f"  (worst arm {worst}, all arms {total_flips}{suffix})")

json.dump(report, open(a.out, "w"), indent=2)
print(f"  wrote {a.out}")
