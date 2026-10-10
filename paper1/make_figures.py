"""Figures for Paper 1, regenerated from archived results only.

    python paper1/make_figures.py

Reads results/b14_primary_gpt2_indep.jsonl, results/b14_analysis.json, results/every_run.json,
results/cross_arm_analysis.json, results/b15_analysis.json and
data/b11/b11c_pythia-14b_s3200_indep.jsonl; writes PDF and PNG to paper1/figures/.

Each figure is drawn at its printed size (TMLR text width 6.5 in), so 8 pt here is 8 pt on the
page, in the paper's own typeface (Latin Modern, via matplotlib's Computer Modern). Colours are the
first three slots of the validated categorical palette, one per check, fixed across figures;
every series also carries a line style or marker, so the figures survive greyscale printing.
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from b1_signal_calibration import PASS, roc  # noqa: E402

OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)
WIDTH = 6.5

COL = {"held-out R2": "#2a78d6", "restart agreement": "#eb6834", "route agreement": "#1baf7a"}
DASH = {"held-out R2": "-", "restart agreement": (0, (4, 2)), "route agreement": (0, (1, 1.6))}
LABEL = {"held-out R2": "held-out $R^2$", "restart agreement": "restart agreement",
         "route agreement": "route agreement"}
INK, INK2, INK3, GRID, SHADE = "#1a1a19", "#52514e", "#8a8984", "#e8e7e3", "#f3f2ee"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True, "axes.unicode_minus": False,
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5, "text.color": INK, "axes.labelcolor": INK,
    "axes.edgecolor": INK2, "axes.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK2,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
    "ytick.major.size": 2.5, "xtick.direction": "out", "ytick.direction": "out",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
    "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "legend.frameon": False, "legend.handlelength": 2.2, "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.dpi": 300, "figure.dpi": 150,
})


def rows(path):
    return [json.loads(l) for l in open(os.path.join(ROOT, path), encoding="utf-8") if l.strip()]


def load(path):
    return json.load(open(os.path.join(ROOT, path), encoding="utf-8"))


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def panel(ax, letter, text=None):
    """A bold panel letter and, beside it, a plain identifying label (not a sentence)."""
    ax.text(-0.02, 1.06, letter, transform=ax.transAxes, fontweight="bold", fontsize=9,
            ha="right", va="bottom")
    if text:
        ax.text(0.02, 1.06, text, transform=ax.transAxes, fontsize=7.5, color=INK2,
                ha="left", va="bottom")


def ygrid(ax):
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)


def spread(ys, gap):
    """Nudge label positions apart, in order, so that neighbours are at least `gap` apart."""
    order = np.argsort(ys)
    out = np.array(ys, float)
    for a, b in zip(order[:-1], order[1:]):
        if out[b] - out[a] < gap:
            out[b] = out[a] + gap
    return out


SCORE = {  # higher = more suspect, as in every analysis script
    "held-out R2": lambda r: -r["r2_k1"],
    "restart agreement": lambda r: -r["stability"],
    "route agreement": lambda r: -r["route_agreement"],
}


# ------------------------------------------------------------------------------------- ROC
def roc_panel(ax, R, signals):
    fail = [r["align_selected"] < PASS for r in R]
    ax.plot([0, 1], [0, 1], color=INK3, lw=0.6, ls=(0, (1, 2)))
    ax.text(0.62, 0.55, "chance", rotation=45, color=INK3, fontsize=7, ha="center", va="center",
            rotation_mode="anchor")
    for name in signals:
        pts, auc = roc([SCORE[name](r) for r in R], fail)
        fpr = [0.0] + sorted(p[2] for p in pts) + [1.0]
        tpr = [0.0] + sorted(p[1] for p in pts) + [1.0]
        ax.plot(fpr, tpr, ls=DASH[name], color=COL[name], lw=1.5, solid_capstyle="round",
                label=f"{LABEL[name]}  {auc:.2f}")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.02)
    ax.set_aspect("equal")
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xlabel("false-alarm rate (passing units flagged)")
    ax.set_ylabel("failures caught")
    leg = ax.legend(loc="lower right", title="AUC", title_fontsize=7.5, alignment="right")
    leg._legend_box.align = "right"


def fig_roc():
    b14 = rows("results/b14_primary_gpt2_indep.jsonl")
    b11c = rows("data/b11/b11c_pythia-14b_s3200_indep.jsonl")
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.0), gridspec_kw={"wspace": 0.35})
    roc_panel(axes[0], b14, ["held-out R2", "restart agreement"])
    panel(axes[0], "a", f"GPT-2 small, layer 6: {len(b14)} units, "
                        f"{sum(r['align_selected'] < PASS for r in b14)} failures")
    roc_panel(axes[1], b11c, ["held-out R2", "route agreement", "restart agreement"])
    panel(axes[1], "b", f"Pythia-1.4B, layer 12: {len(b11c)} units, "
                        f"{sum(r['align_selected'] < PASS for r in b11c)} failures")
    save(fig, "fig_roc")


# ---------------------------------------------------------------------- threshold and classes
def fig_threshold_and_classes():
    a = load("results/b14_analysis.json")
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.6), gridspec_kw={"wspace": 0.42,
                                                                       "width_ratios": [1.15, 1]})
    ax = axes[0]
    bars = sorted(a["threshold_sweep"], key=float)
    x = [float(b) for b in bars]
    ends = {}
    for name, mk in (("held-out R2", "o"), ("restart agreement", "s")):
        y = [a["threshold_sweep"][b][name] for b in bars]
        ax.plot(x, y, ls=DASH[name], marker=mk, ms=4, mec="white", mew=0.8, color=COL[name], lw=1.5)
        ends[name] = y[-1]
    for (name, yv), yl in zip(ends.items(), spread(list(ends.values()), 0.04)):
        ax.text(x[-1] + 0.003, yl, LABEL[name], color=INK, fontsize=7.5, va="center")
    ax.axvline(PASS, color=INK3, lw=0.6, ls=(0, (1, 2)))
    ax.text(PASS, 0.515, " filed bar", color=INK3, fontsize=7, va="bottom")
    ax.set_xlim(0.895, 0.985 + 0.03)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:.2f}" if v != 0.97 else "" for v in x])
    ax.set_ylim(0.5, 1.0)
    ax.set_xlabel("failure bar (alignment below which a fit fails)")
    ax.set_ylabel("AUC for failure")
    ygrid(ax)
    panel(ax, "a", "all failures, by failure bar")

    ax = axes[1]
    classes = [("under_fitted", "under-fitted"), ("wrong_basin", "converged-wrong")]
    cut = a["by_failure_class"]["0.99"]
    ax.axhspan(0.4, 0.5, color=SHADE, lw=0)
    ax.axhline(0.5, color=INK3, lw=0.6, ls=(0, (1, 2)))
    ax.text(1.55, 0.505, "chance", color=INK3, fontsize=7, ha="right", va="bottom")
    for k, (name, mk) in enumerate((("held-out R2", "o"), ("restart agreement", "s"))):
        for c, (key, _) in enumerate(classes):
            v = cut[key][name]
            xx = c + (k - 0.5) * 0.28
            ax.plot([xx, xx], v["ci95"], color=COL[name], lw=1.6, solid_capstyle="round")
            ax.plot(xx, v["auc"], marker=mk, ms=5.5, color=COL[name], mec="white", mew=0.9, ls="none",
                    label=LABEL[name] if c == 0 else None)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels([f"{lab}\n{cut[key]['held-out R2']['n_class']} fits" for key, lab in classes])
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(0.4, 1.02)
    ax.set_ylabel("AUC against passing units")
    ygrid(ax)
    ax.legend(loc="upper right", handlelength=1.0, borderaxespad=0.2)
    panel(ax, "b", "by failure class, 95% intervals")
    save(fig, "fig_threshold_classes")


# --------------------------------------------------------------------------------- forest
FOREST = [  # every_run.json arm, label, estimator note
    ("GPT-2 L6 (B-14)", "GPT-2 small, layer 6", "corrected"),
    ("Pythia-160m L6 (B-2c)", "Pythia-160m, layer 6", "corrected"),
    ("GPT-2 L2 (B-7)", "GPT-2 small, layer 2", "coupled"),
    ("GPT-2 L10 (B-7)", "GPT-2 small, layer 10", "coupled"),
    ("GPT-Neo-125m L10 (B-8b)", "GPT-Neo-125m, layer 10", "corrected"),
    ("Pythia-1.4B L12, 3200 steps (B-11c)", "Pythia-1.4B, layer 12", "corrected"),
]


def signed(x):
    """A signed number in math mode, so the minus is a true minus in Computer Modern."""
    return f"${x:+.3f}$"


def fig_forest():
    from matplotlib.transforms import blended_transform_factory
    d = load("results/every_run.json")
    arms, pool = d["arms"], d["pool_filed"]
    n = len(FOREST)
    fig = plt.figure(figsize=(WIDTH, 2.6))
    ax = fig.add_axes([0.36, 0.2, 0.36, 0.68])
    lo_x, hi_x = -0.40, 0.25
    ax.set_xlim(lo_x, hi_x)
    ax.set_ylim(-0.8, n + 0.9)
    row = blended_transform_factory(fig.transFigure, ax.transData)
    ax.axvspan(lo_x, 0, color=SHADE, lw=0, zorder=0)
    ax.axvline(0, color=INK2, lw=0.7, zorder=1)
    w = np.array([1 / arms[k]["delong_se"] ** 2 for k, _, _ in FOREST])
    size = 3.5 + 4.5 * np.sqrt(w / w.max())
    for i, (key, lab, est) in enumerate(FOREST):
        y = n - i
        r = arms[key]
        lo, hi = r["boot_ci95"]
        ax.plot([lo, hi], [y, y], color=INK, lw=1.0, solid_capstyle="butt", zorder=2)
        ax.plot(r["diff"], y, marker="s", ms=size[i], color=INK, mec="white", mew=0.6, zorder=3)
        fig.text(0.015, y, lab, transform=row, fontsize=7.5, va="center")
        fig.text(0.29, y, f"{r['failures']}/{r['n']}" + ("" if est == "corrected" else "*"),
                 transform=row, fontsize=7.5, va="center", ha="right")
        fig.text(0.99, y, f"{signed(r['diff'])}  [{signed(lo)}, {signed(hi)}]", transform=row,
                 fontsize=7.5, va="center", ha="right")
    yp = 0
    p, (plo, phi) = pool["pooled"], pool["hksj_ci95"]
    ax.add_patch(Polygon([[plo, yp], [p, yp + 0.33], [phi, yp], [p, yp - 0.33]],
                         closed=True, color=COL["held-out R2"], lw=0, zorder=3))
    fig.text(0.015, yp, "pooled, random effects", transform=row, fontsize=7.5, va="center",
             fontweight="bold")
    fig.text(0.99, yp, f"{signed(p)}  [{signed(plo)}, {signed(phi)}]", transform=row,
             fontsize=7.5, va="center", ha="right", color=INK)
    ax.axhline(0.55, color=GRID, lw=0.6)
    for xf, txt, ha in ((0.015, "condition", "left"), (0.29, "failures/units", "right"),
                        (0.99, r"$\Delta$AUC  [95% interval]", "right")):
        fig.text(xf, n + 0.75, txt, transform=row, fontsize=7.5, color=INK2, va="center", ha=ha)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xticks([-0.4, -0.2, 0, 0.2])
    ax.set_xlabel("AUC(restart agreement) $-$ AUC(held-out $R^2$)")
    ax.text(lo_x + 0.012, n + 0.75, "held-out $R^2$ better", fontsize=7, color=INK2, va="center")
    ax.text(hi_x - 0.012, n + 0.75, "restart better", fontsize=7, color=INK2, va="center",
            ha="right")
    fig.text(0.015, 0.015, "* coupled-estimator run.  Per-arm intervals: stratified bootstrap;  "
             "pooled: Hartung-Knapp-Sidik-Jonkman.  Marker area tracks inverse variance.",
             fontsize=6.5, color=INK2)
    save(fig, "fig_forest")


# ---------------------------------------------------------------------------- restart curve
def fig_restart_curve():
    rc = load("results/cross_arm_analysis.json")["restart_curve"]
    runs = list(zip(("coupled run A", "coupled run B"), ("o", "^"), rc.values())) + \
        [("corrected re-fit", "s", load("results/b15_analysis.json")["restart_curve"])]
    ks = [2, 3, 4, 5]
    fig, ax = plt.subplots(figsize=(3.25, 2.55))
    labels = []
    for tag, mk, d in runs:
        ax.plot(ks, [d["held-out R2"]] * 4, color=COL["held-out R2"], lw=1.4, marker=mk, ms=3.6,
                mec="white", mew=0.7, markevery=[0, 3])
        ax.plot(ks, [d[f"k={k}"] for k in ks], ls=DASH["restart agreement"], marker=mk, ms=3.6,
                mec="white", mew=0.7, color=COL["restart agreement"], lw=1.4)
        labels += [(d["held-out R2"], f"$R^2$, {tag}"), (d["k=5"], f"restart, {tag}")]
    for (yv, txt), yl in zip(labels, spread([l[0] for l in labels], 0.026)):
        ax.text(5.18, yl, txt, fontsize=6.8, va="center", color=INK)
        ax.plot([5.05, 5.15], [yv, yl], color=INK3, lw=0.4)
    ax.set_xticks(ks)
    ax.set_xlim(1.85, 5.1)
    ax.set_ylim(0.6, 1.0)
    ax.set_xlabel("restarts used for agreement")
    ax.set_ylabel("AUC for failure")
    ygrid(ax)
    save(fig, "fig_restart_curve")


if __name__ == "__main__":
    for f in (fig_roc, fig_threshold_and_classes, fig_forest, fig_restart_curve):
        f()
    print("wrote", sorted(os.listdir(OUT)))
