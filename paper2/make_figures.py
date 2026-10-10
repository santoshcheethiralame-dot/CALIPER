"""Figures for Paper 2, regenerated from the results files.

    python paper2/make_figures.py

fig_health_auc  each health check's AUC at telling live from dead vectors, pooled across recipes,
                per model (S-2, S-1 Gemma-3-12B, P2-L logit-lens accessibility)
fig_dose        a: one fraction of the residual norm, three models, next-token KL;
                b: the KL-calibrated grid on Gemma-3-4B, real and on-manifold vectors against
                off-manifold ones
fig_live_kl     share of real vectors steered against each vector's own next-token KL, one panel
                per model with the others in grey; intervals resample vectors (internal review A7)
fig_logodds     P2-F on the log-odds scale (internal review, 10 Oct): factual-NO against
                introspective change per cell; above the diagonal, factual answers move more

Drawn at printed size (NeurIPS text width 5.5 in) in the paper's typeface (Times). Colours are the
validated categorical palette in fixed slot order; every series also has its own marker, so the
figures survive greyscale printing.
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)
WIDTH = 5.5

SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
MARK = ["o", "s", "^", "D", "v", "P"]
INK, INK2, INK3, GRID, SHADE, GHOST = "#1a1a19", "#52514e", "#8a8984", "#e8e7e3", "#f3f2ee", "#cfcdc6"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "text.color": INK, "axes.labelcolor": INK,
    "axes.edgecolor": INK2, "axes.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK2,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.minor.size": 1.5, "ytick.minor.size": 1.5, "xtick.direction": "out",
    "ytick.direction": "out", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
})


def load(p):
    return json.load(open(os.path.join(ROOT, p), encoding="utf-8"))


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def panel(ax, letter, text=None):
    ax.text(-0.02, 1.05, letter, transform=ax.transAxes, fontweight="bold", fontsize=9,
            ha="right", va="bottom")
    if text:
        ax.text(0.02, 1.05, text, transform=ax.transAxes, fontsize=7.5, color=INK2, va="bottom")


def label_at(ax, x, y, text, dx=4, dy=0, ha="left", color=INK, size=7.2):
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", va="center",
                ha=ha, fontsize=size, color=color)


# --------------------------------------------------------------------------- health checks
def fig_health_auc():
    runs = [("Qwen2.5-3B", "results/s2_qwen3b_analysis.json", "S-2 Qwen2.5-3B"),
            ("Qwen2.5-7B", "results/s2_qwen7b_analysis.json", "S-2 Qwen2.5-7B"),
            ("Gemma-3-4B, KL grid", "results/s2_gemma4b_kl_analysis.json", "S-2 Gemma-3-4B, KL grid"),
            ("Gemma-3-12B, 4-bit", "results/s1_gemma12_analysis.json", "S-1 Gemma-3-12B, 4-bit")]
    stats = ["norm", "distinctness", "stability", "probe", "logit steering", "P(YES) shift",
             "logit-lens accessibility"]
    lap = load("results/p2l_logit_lens_health.json")
    sens = load("results/p2_rule_sensitivity.json")  # holds the Gemma-12B intervals
    fig, ax = plt.subplots(figsize=(WIDTH, 2.9))
    ax.axvspan(0, 0.5, color=SHADE, lw=0, zorder=0)
    ax.axvline(0.5, color=INK3, lw=0.6, ls=(0, (1, 2)))
    for j in range(len(stats) - 1):
        ax.axhline(j + 0.5, color=GRID, lw=0.5, zorder=1)
    for i, (name, path, key) in enumerate(runs):
        a = load(path)
        if "cells" in a:
            auc = dict(sens["S-1 Gemma-3-12B, 4-bit"]["filed"]["auc"])
            gate = a["cells"]["4bit"]["pass_rate"]["0.5 nat (gate)"]
            live, total = sum(n for n, _ in gate.values()), sum(m for _, m in gate.values())
        else:
            auc, live, total = dict(a["auc"]), a["live"], a["vectors"]
        auc["logit-lens accessibility"] = {"auc": lap[key]["auc"], "ci95": lap[key]["ci95"]}
        first = True
        for j, s_ in enumerate(stats):
            if s_ not in auc:
                continue
            v = auc[s_]
            yy = j + (i - 1.5) * 0.19
            ax.plot(v["ci95"], [yy, yy], color=SLOT[i], lw=1.3, solid_capstyle="round", zorder=2)
            ax.plot(v["auc"], yy, marker=MARK[i], color=SLOT[i], ms=4.6, mec="white", mew=0.7,
                    ls="none", zorder=3, label=f"{name}  ({live} live of {total})" if first else None)
            first = False
    ax.set_yticks(range(len(stats)), stats)
    ax.tick_params(axis="y", length=0)
    ax.invert_yaxis()
    ax.set_ylim(len(stats) - 0.5, -0.5)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("AUC, live against dead vectors, pooled across recipes (95% interval)")
    ax.text(0.25, -0.75, "worse than chance", color=INK3, fontsize=7, ha="center", va="bottom")
    ax.text(0.75, -0.75, "better than chance", color=INK3, fontsize=7, ha="center", va="bottom")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.17), ncol=2, handletextpad=0.3,
              columnspacing=1.4)
    save(fig, "fig_health_auc")


# ----------------------------------------------------------------------------------- dose
def fig_dose():
    d = load("results/p2d_dose_transfer.json")
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.45), gridspec_kw={"wspace": 0.42})
    ax = axes[0]
    for i, (name, pts) in enumerate(d["transfer_at_same_fraction"].items()):
        x = [p["fraction"] for p in pts if p["fraction"] > 0]
        y = [p["median_kl"] for p in pts if p["fraction"] > 0]
        ax.plot(x, y, color=SLOT[i], lw=1.5, marker=MARK[i], ms=4.6, mec="white", mew=0.7)
        label_at(ax, x[0], y[0], name, dx=-6, ha="right")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([0.25, 0.5, 1.0], ["0.25", "0.5", "1"])
    ax.minorticks_off()
    ax.set_xlim(0.06, 1.3)
    ax.yaxis.grid(True)
    ax.set_xlabel("dose, fraction of the residual norm")
    ax.set_ylabel("median next-token KL (nats)")
    panel(ax, "a", "same fraction, three models")

    ax = axes[1]
    rows = d["real_vs_random_on_kl_grid"]
    arms = [("concept", "concept"), ("span", "span"), ("random", "random"), ("shuffle", "shuffled")]
    x = [r["target_nats"] for r in rows]
    ax.plot([0.03, 8], [0.03, 8], ls=(0, (3, 2)), lw=0.8, color=INK3)
    ys = {}
    for i, (arm, name) in enumerate(arms):
        y = [r[arm] for r in rows]
        ax.plot(x, y, color=SLOT[i], lw=1.5, marker=MARK[i], ms=4.6, mec="white", mew=0.7)
        ys[name] = y[-1]
    order = sorted(ys, key=ys.get)
    pos = np.log10([ys[k] for k in order])
    for a_, b_ in zip(range(len(pos) - 1), range(1, len(pos))):
        if pos[b_] - pos[a_] < 0.16:
            pos[b_] = pos[a_] + 0.16
    for k, p in zip(order, pos):
        label_at(ax, x[-1], 10 ** p, k, dx=6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(x, [f"{v:g}" for v in x])
    ax.minorticks_off()
    ax.set_xlim(0.035, 14)
    ax.yaxis.grid(True)
    ax.set_xlabel("dose, calibrated KL target (nats)")
    ax.set_ylabel("median next-token KL (nats)")
    panel(ax, "b", "Gemma-3-4B, KL-calibrated grid")
    save(fig, "fig_dose")


# --------------------------------------------------------------------------- live vs own KL
def fig_live_kl():
    d = load("results/p2_review_analysis.json")["A7 live rate against own KL"]
    names = list(d)
    fig, axes = plt.subplots(2, 3, figsize=(WIDTH, 3.4), sharex=True, sharey=True,
                             gridspec_kw={"wspace": 0.12, "hspace": 0.42})
    series = {}
    for name in names:
        bins = [b for b in d[name]["bins"] if b["median_kl"] > 0]
        series[name] = (np.array([b["median_kl"] for b in bins]), np.array([b["live_rate"] for b in bins]),
                        np.array([b["ci95_vector_bootstrap"] for b in bins]))
    for k, (ax, name) in enumerate(zip(axes.flat, names)):
        for other in names:
            if other != name:
                ax.plot(series[other][0], series[other][1], color=GHOST, lw=0.9, zorder=1)
        x, y, ci = series[name]
        ax.fill_between(x, ci[:, 0], ci[:, 1], color=SLOT[k], alpha=0.14, lw=0, zorder=2)
        ax.plot(x, y, color=SLOT[k], lw=1.5, marker=MARK[k], ms=4.2, mec="white", mew=0.7, zorder=3)
        ax.set_title(name.replace(" (S-1M)", ""), fontsize=7.5, loc="left", color=INK, pad=3)
        ax.set_xscale("log")
        ax.yaxis.grid(True)
        ax.minorticks_off()
    for ax in axes[1]:
        ax.set_xlabel("own next-token KL (nats)")
        ax.set_xticks([1e-4, 1e-2, 1, 100], ["$10^{-4}$", "$10^{-2}$", "1", "100"])
    for ax in axes[:, 0]:
        ax.set_ylabel("share steered")
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].set_xlim(5e-6, 200)
    save(fig, "fig_live_kl")


# ------------------------------------------------------------------------------- log-odds
def fig_logodds():
    d = load("results/p2_review_analysis.json")["A1 C8 on log-odds"]
    series = {}
    for k, v in d.items():
        name = k.split(",")[0].replace(" released", ", released vectors")
        series.setdefault(name, []).append((k.split(", ")[-1], v["logodds_change_median"]))
    fig, ax = plt.subplots(figsize=(3.05, 3.05))
    lim = (-3, 28)
    ax.fill_between(lim, lim, [lim[1]] * 2, color=SHADE, lw=0, zorder=0)
    ax.plot(lim, lim, color=INK3, lw=0.7, ls=(0, (3, 2)))
    ax.text(-2, 27.3, "factual moves more", color=INK2, fontsize=7.2, va="center")
    ax.text(27, 12, "factual moves less", color=INK2, fontsize=7.2, ha="right", va="center")
    for i, (name, pts) in enumerate(series.items()):
        x = [c["introspective"] for _, c in pts]
        y = [c["factual_no"] for _, c in pts]
        ax.plot(x, y, color=SLOT[i], lw=1.4, marker=MARK[i], ms=4.6, mec="white", mew=0.7,
                label=name.replace("Qwen2.5-7B ", "Qwen2.5-7B, ") + ("" if "released" in name else " vectors"))
        if "tail" in name:
            label_at(ax, x[0], y[0], "0.25 of the norm", dx=5, dy=-5, color=INK2, size=7)
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_aspect("equal")
    ax.set_xlabel("introspective change, log-odds of YES")
    ax.set_ylabel("factual-NO change, log-odds of YES")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=1, handletextpad=0.4)
    save(fig, "fig_logodds")


def main():
    for f in (fig_health_auc, fig_dose, fig_live_kl, fig_logodds):
        f()
    print("wrote", sorted(os.listdir(OUT)))


if __name__ == "__main__":
    main()
