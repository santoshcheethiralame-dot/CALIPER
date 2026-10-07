"""Figures for Paper 1, regenerated from archived results only.

    python paper1/make_figures.py

Reads results/b14_primary_gpt2_indep.jsonl, results/b14_analysis.json,
results/cross_arm_analysis.json and data/b11/b11c_pythia-14b_s3200_indep.jsonl, and writes
PDFs to paper1/figures/. Style follows paper/make_figures.py.
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from b1_signal_calibration import PASS, roc  # noqa: E402

OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)

# Categorical slots fixed per signal across every figure.
COL = {"held-out R2": "#2a78d6", "restart agreement": "#eb6834",
       "route agreement": "#1baf7a"}
DASH = {"held-out R2": "-", "restart agreement": "--", "route agreement": ":"}
LABEL = {"held-out R2": "held-out $R^2$", "restart agreement": "restart agreement",
         "route agreement": "route agreement"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "sans-serif",
})


def rows(path):
    return [json.loads(l) for l in open(os.path.join(ROOT, path), encoding="utf-8") if l.strip()]


SCORE = {  # higher = more suspect, as in every analysis script
    "held-out R2": lambda r: -r["r2_k1"],
    "restart agreement": lambda r: -r["stability"],
    "route agreement": lambda r: -r["route_agreement"],
}


def roc_panel(ax, R, signals, title):
    fail = [r["align_selected"] < PASS for r in R]
    for name in signals:
        pts, auc = roc([SCORE[name](r) for r in R], fail)
        fpr = [0.0] + sorted(p[2] for p in pts) + [1.0]
        tpr = [0.0] + sorted(p[1] for p in pts) + [1.0]
        ax.plot(fpr, tpr, DASH[name], color=COL[name], lw=1.3,
                label=f"{LABEL[name]} ({auc:.2f})")
    ax.plot([0, 1], [0, 1], color=GRID, lw=0.8)
    ax.set_xlabel("false-alarm rate (good units flagged)")
    ax.set_ylabel("failures caught")
    ax.set_title(title, loc="left")
    ax.legend(loc="lower right")


def fig_roc():
    b14 = rows("results/b14_primary_gpt2_indep.jsonl")
    b11c = rows("data/b11/b11c_pythia-14b_s3200_indep.jsonl")
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.6))
    roc_panel(axes[0], b14, ["held-out R2", "restart agreement"],
              f"GPT-2 small L6, n=300, {sum(r['align_selected'] < PASS for r in b14)} failures")
    roc_panel(axes[1], b11c, ["held-out R2", "route agreement", "restart agreement"],
              f"Pythia-1.4B L12, n=50, {sum(r['align_selected'] < PASS for r in b11c)} failures")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_roc.pdf"))
    plt.close(fig)


def fig_threshold_and_classes():
    a = json.load(open(os.path.join(ROOT, "results/b14_analysis.json")))
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.4))
    ax = axes[0]
    bars = sorted(a["threshold_sweep"], key=float)
    for name in ("held-out R2", "restart agreement"):
        ax.plot([float(b) for b in bars], [a["threshold_sweep"][b][name] for b in bars],
                DASH[name], marker="o", ms=3.5, mec="white", color=COL[name], lw=1.3,
                label=LABEL[name])
    ax.set_xlabel("failure bar (alignment below)")
    ax.set_ylabel("AUC")
    ax.set_ylim(0.5, 1.0)
    ax.set_title("Ordering at every bar", loc="left")
    ax.legend(loc="lower left")

    ax = axes[1]
    classes = [("under_fitted", "under-fitted"), ("wrong_basin", "converged-wrong")]
    cut = a["by_failure_class"]["0.99"]
    w = 0.36
    for k, name in enumerate(("held-out R2", "restart agreement")):
        xs, ys, lo, hi = [], [], [], []
        for c, (key, _) in enumerate(classes):
            v = cut[key][name]
            xs.append(c + (k - 0.5) * w)
            ys.append(v["auc"])
            lo.append(v["auc"] - v["ci95"][0])
            hi.append(v["ci95"][1] - v["auc"])
        ax.bar(xs, ys, width=w, color=COL[name], alpha=0.85, label=LABEL[name])
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor=INK2, lw=0.8, capsize=2)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels([f"{lab}\n(n={cut[key]['held-out R2']['n_class']})"
                        for key, lab in classes])
    ax.axhline(0.5, color=INK2, lw=0.6, ls=":")
    ax.set_ylim(0.4, 1.02)
    ax.set_ylabel("AUC vs passing units")
    ax.set_title("Where the advantage comes from", loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_threshold_classes.pdf"))
    plt.close(fig)


def readable(arm):
    """Paper-facing condition names; internal run codes stay out of the figures."""
    model, rest = arm.split(" (")[0], arm.split(" (")[-1]
    tag = " (fixed)" if "fixed" in rest else " (coupled)"
    return model.replace("GPT-2 L", "GPT-2 small, layer ").replace(" L", ", layer ").replace(
        ", 3200 steps", "").replace("1.4b", "1.4B") + tag


def fig_forest():
    m = json.load(open(os.path.join(ROOT, "results/cross_arm_analysis.json")))["meta"]
    arms = m["pooled_arms"]
    fig, ax = plt.subplots(figsize=(5.5, 2.3))
    for y, name in enumerate(reversed(arms)):
        a = m["per_arm"][name]
        ax.errorbar(a["diff"], y, xerr=1.96 * a["se"], fmt="o", ms=3.5, color=INK,
                    ecolor=INK2, lw=1.0, capsize=2, mec="white")
        ax.annotate(f"{readable(name)}, {a['failures']} failures", (0.06, y), fontsize=6.5,
                    color=INK2, va="center")
    re = m["random_effects"]
    y = -1.2
    ax.errorbar(re["pooled_diff"], y, xerr=[[re["pooled_diff"] - re["ci95"][0]],
                [re["ci95"][1] - re["pooled_diff"]]], fmt="D", ms=4.5,
                color=COL["held-out R2"], ecolor=COL["held-out R2"], lw=1.4, capsize=2)
    ax.annotate(f"random effects  {re['pooled_diff']:.3f} "
                f"[{re['ci95'][0]:.3f}, {re['ci95'][1]:.3f}]", (0.06, y), fontsize=6.5,
                color=COL["held-out R2"], va="center")
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks([])
    ax.set_xlim(-0.6, 0.75)
    ax.set_xlabel("AUC(restart agreement) - AUC(held-out $R^2$)")
    ax.set_title("Restart agreement trails held-out fit in every condition", loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_forest.pdf"))
    plt.close(fig)


def fig_restart_curve():
    rc = json.load(open(os.path.join(ROOT, "results/cross_arm_analysis.json")))["restart_curve"]
    # Two coupled-estimator runs, then B-15a's corrected-estimator re-fit (square markers).
    runs = [("run A", "o", d) for d in rc.values()][:1] + \
           [("run B", "o", d) for d in rc.values()][1:2] + \
           [("corrected", "s", json.load(open(os.path.join(
               ROOT, "results/b15_analysis.json")))["restart_curve"])]
    fig, ax = plt.subplots(figsize=(2.7, 2.3))
    for tag, mk, d in runs:
        ks = [2, 3, 4, 5]
        ax.plot(ks, [d[f"k={k}"] for k in ks], "--", marker=mk, ms=3.5, mec="white",
                color=COL["restart agreement"], lw=1.2)
        ax.annotate(f"restart, {tag}", (5.1, d["k=5"]), fontsize=6.5,
                    color=COL["restart agreement"], va="center")
        ax.plot([2, 5], [d["held-out R2"]] * 2, color=COL["held-out R2"], lw=1.0)
        ax.annotate(f"$R^2$, {tag}", (5.1, d["held-out R2"]), fontsize=6.5,
                    color=COL["held-out R2"], va="center")
    ax.set_xlabel("restarts used for agreement")
    ax.set_ylabel("AUC")
    ax.set_xticks([2, 3, 4, 5])
    ax.set_ylim(0.5, 1.0)
    ax.set_xlim(1.8, 6.6)
    ax.set_title("More restarts do not help", loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_restart_curve.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    for f in (fig_roc, fig_threshold_and_classes, fig_forest, fig_restart_curve):
        f()
    print("wrote", sorted(os.listdir(OUT)))
