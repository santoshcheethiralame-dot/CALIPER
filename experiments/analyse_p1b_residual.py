"""C57 finished: does a plant-specific signal survive removing the shared component?

P1b measured recovery 0.1557 against a same-bank null of 0.1870, with the manipulation
check passed. C58 then measured the bank's own collinearity at median |cos| 0.4216, p90
0.73. So both the recovery and the null sit far below the inter-concept floor, and the
question the raw numbers cannot answer is whether ANY plant-specific signal is present
once the component every concept vector shares is projected out.

That analysis needs the vectors, which no run saved until 2026-09-08e. This script runs
entirely offline against the dumped .npz plus the P1b rows.

Usage:
    python experiments/analyse_p1b_residual.py <vectors.npz> <p1b_plant.jsonl>
"""
import json
import sys

import numpy as np


def unit(x, axis=-1):
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + 1e-12)


def main(vec_path, rows_path):
    z = np.load(vec_path, allow_pickle=True)
    names = list(z["names"])
    V = unit(z["vectors"].astype(np.float64))          # (n_concepts, d)
    rows = [json.loads(l) for l in open(rows_path, encoding="utf-8") if l.strip()]
    rows = [r for r in rows if r.get("extract_layer") == r.get("plant_layer")]
    print(f"{len(names)} vectors, d={V.shape[1]}, {len(rows)} plant rows at the plant layer")

    G = V @ V.T
    off = G[~np.eye(len(V), dtype=bool)]
    print(f"bank collinearity: median |cos| {np.median(np.abs(off)):.4f}  "
          f"mean {off.mean():.4f}  p90 {np.quantile(np.abs(off), 0.9):.4f}")

    # The shared component: top principal direction of the concept bank. If the bank has
    # a dominant common direction, this is it, and it is what both recovery and the
    # same-bank null are mostly measuring.
    U, S, _ = np.linalg.svd(V - 0, full_matrices=False)
    pc1 = unit(np.linalg.svd(V, full_matrices=False)[2][0])
    var1 = S[0] ** 2 / (S ** 2).sum()
    print(f"top PC explains {var1:.1%} of the bank's variance; "
          f"median |cos(v_i, PC1)| = {np.median(np.abs(V @ pc1)):.4f}")

    # Residual bank: every concept vector with PC1 removed.
    Vr = unit(V - np.outer(V @ pc1, pc1))
    offr = (Vr @ Vr.T)[~np.eye(len(V), dtype=bool)]
    print(f"after removing PC1: median |cos| {np.median(np.abs(offr)):.4f}  "
          f"(was {np.median(np.abs(off)):.4f})")

    idx = {n: i for i, n in enumerate(names)}
    print(f"\n{'alpha%':>7} {'plant':>10} {'steer':>7} | "
          f"{'rec':>8} {'null':>9} | {'cos(vp,vq)':>11} {'same, no PC1':>13}")
    print("-" * 74)
    agg = {}
    for r in sorted(rows, key=lambda r: (r["alpha_frac"], r["plant_name"])):
        p, q = r["plant_name"], r["null_name"]
        if p not in idx or q not in idx:
            continue
        # Reconstruct the extracted difference direction from the reported cosines is not
        # possible; instead score what IS reconstructable - how much of the raw recovery
        # is explained by the shared component alone.
        vp, vq = V[idx[p]], V[idx[q]]
        vpr, vqr = Vr[idx[p]], Vr[idx[q]]
        # Expected null under a pure-shared-component difference: cos(v_p, v_q).
        expected_null = abs(float(vp @ vq))
        resid_sim = abs(float(vpr @ vqr))
        k = r["alpha_frac"]
        agg.setdefault(k, []).append(
            (r["recovery"], r["null_a"], expected_null, resid_sim, r["steer_rate"]))
        print(f"{k:>6.0%} {p:>10} {r['steer_rate']:>6.2f} | "
              f"{r['recovery']:>8.4f} {r['null_a']:>9.4f} | "
              f"{expected_null:>10.4f} {resid_sim:>11.4f}")

    print(f"\n{'alpha%':>7} | {'med rec':>8} {'med null':>9} {'med cos(vp,vq)':>15} "
          f"{'rec - null':>11}")
    print("-" * 60)
    for k, v in sorted(agg.items()):
        rec = np.median([x[0] for x in v]); nul = np.median([x[1] for x in v])
        exp = np.median([x[2] for x in v])
        print(f"{k:>6.0%} | {rec:>8.4f} {nul:>9.4f} {exp:>15.4f} {rec - nul:>+11.4f}")

    print("\nHow to read this:")
    print("  cos(vp, vq) is what the same-bank null WOULD be if the extracted difference")
    print("  were exactly the planted vector - the ceiling the null reaches under perfect")
    print("  recovery. Recovery far below it means the difference is not the plant;")
    print("  recovery below the null itself means it is no closer to the plant than to")
    print("  an unrelated concept.")
    print()
    print("  WHAT THIS SCRIPT CANNOT DO, and why a re-run is needed: P1b saved only the")
    print("  COSINES, not the extracted difference vectors. The test that matters -")
    print("  project the shared component out of the DIFFERENCE and re-score against the")
    print("  plant - is not computable from these files. Version 2026-09-08f saves the")
    print("  diffs; re-run P1b once and this becomes permanent offline work.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2])
