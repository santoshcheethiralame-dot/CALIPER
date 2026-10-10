"""Figures for Paper 2, regenerated from the results files.

    python paper2/make_figures.py

fig_dose_fraction   one fraction of the residual norm, three models: next-token KL
fig_dose_kl         the KL-calibrated grid on Gemma-3-4B: real and on-manifold vectors against
                    off-manifold ones, with the calibration target
fig_health_auc      each health check's AUC at telling live from dead vectors, per model, with
                    S-1 Gemma-3-12B (no intervals stored) and logit-lens accessibility (P2-L)
fig_yesbias         P2-F: the factual-NO P(YES) change against the introspective change at the
                    same arm and dose; the yes-bias appears only at the highest doses
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

# Paper 1's categorical slots, in the same fixed order.
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MARK = ["o", "s", "^", "D"]
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


def load(p):
    return json.load(open(os.path.join(ROOT, p)))


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight", dpi=200)
    plt.close(fig)


def label_at(ax, x, y, text, dx=4, dy=0, ha="left"):
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", va="center",
                ha=ha, fontsize=7.5, color=INK2)


def fig_dose_fraction(d):
    t = d["transfer_at_same_fraction"]
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    for i, (name, pts) in enumerate(t.items()):
        x = [p["fraction"] for p in pts if p["fraction"] > 0]
        y = [p["median_kl"] for p in pts if p["fraction"] > 0]
        ax.plot(x, y, color=SLOT[i], lw=2, marker=MARK[i], ms=5, mec="white", mew=1, label=name)
        label_at(ax, x[0], y[0], name, dx=-7, ha="right")   # the doses part most at the start
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([0.25, 0.5, 1.0], ["0.25", "0.5", "1.0"])
    ax.set_xlim(0.07, 1.3)
    ax.set_xlabel("dose, fraction of the residual norm")
    ax.set_ylabel("median next-token KL (nats)")
    ax.set_title("The same fraction is not the same dose", loc="left")
    ax.legend(loc="lower right")
    save(fig, "fig_dose_fraction")


def fig_dose_kl(d):
    rows = d["real_vs_random_on_kl_grid"]
    arms = [("concept", "concept vectors"), ("span", "span of concept vectors"),
            ("random", "random directions"), ("shuffle", "shuffled vectors")]
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    x = [r["target_nats"] for r in rows]
    ax.plot([0.03, 8], [0.03, 8], ls="--", lw=1, color=INK2, label="calibration target")
    for i, (arm, name) in enumerate(arms):
        y = [r[arm] for r in rows]
        ax.plot(x, y, color=SLOT[i], lw=2, marker=MARK[i], ms=5, mec="white", mew=1, label=name)
        label_at(ax, x[-1], y[-1], name.split()[0], dy={"random": 7, "shuffled": -8}.get(
            name.split()[0], 0))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(x, [f"{v:g}" for v in x])
    ax.set_xlim(0.035, 30)
    ax.set_xlabel("dose, calibrated KL target (nats)")
    ax.set_ylabel("median next-token KL (nats)")
    ax.set_title("Real vectors are gentler than random ones", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2)
    save(fig, "fig_dose_kl")


def fig_health_auc():
    runs = [("Qwen2.5-3B", "results/s2_qwen3b_analysis.json", "S-2 Qwen2.5-3B"),
            ("Qwen2.5-7B", "results/s2_qwen7b_analysis.json", "S-2 Qwen2.5-7B"),
            ("Gemma-3-4B, KL grid", "results/s2_gemma4b_kl_analysis.json", "S-2 Gemma-3-4B, KL grid"),
            ("Gemma-3-12B, 4-bit", "results/s1_gemma12_analysis.json", "S-1 Gemma-3-12B, 4-bit")]
    stats = ["norm", "distinctness", "stability", "probe", "logit steering", "P(YES) shift",
             "logit-lens accessibility"]
    lap = load("results/p2l_logit_lens_health.json")
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    ax.axvline(0.5, color=INK2, lw=1, ls="--")
    for i, (name, path, key) in enumerate(runs):
        a = load(path)
        if "cells" in a:  # S-1: AUCs at the gate, no intervals stored
            auc = {k: {"auc": v} for k, v in a["cells"]["4bit"]["health_auc_at_gate"].items()}
            live = sum(n for n, _ in a["cells"]["4bit"]["pass_rate"]["0.5 nat (gate)"].values())
            total = sum(m for _, m in a["cells"]["4bit"]["pass_rate"]["0.5 nat (gate)"].values())
        else:
            auc, live, total = a["auc"], a["live"], a["vectors"]
        auc = dict(auc)
        auc["logit-lens accessibility"] = {"auc": lap[key]["auc"], "ci95": lap[key]["ci95"]}
        first = True
        for j, s_ in enumerate(stats):
            if s_ not in auc:
                continue
            v = auc[s_]
            yy = j + (i - 1.5) * 0.18
            if "ci95" in v:
                ax.plot(v["ci95"], [yy, yy], color=SLOT[i], lw=1.5, solid_capstyle="round")
            ax.plot(v["auc"], yy, marker=MARK[i], color=SLOT[i], ms=5, mec="white", mew=1,
                    ls="none", label=f"{name} ({live} live / {total})" if first else None)
            first = False
    ax.set_yticks(range(len(stats)), stats)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("AUC, live vs dead (0.5 = chance)")
    ax.set_title("No health check finds live vectors on every model", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=1)
    save(fig, "fig_health_auc")


def fig_yesbias():
    d = load("results/p2f_analysis.json")
    fig, ax = plt.subplots(figsize=(3.3, 2.9))
    ax.plot([0, 1], [0, 1], color=INK2, lw=1, ls="--")
    label_at(ax, 1.0, 0.72, "all yes-bias", dx=0, ha="right")
    for i, (name, rows) in enumerate(d.items()):
        x = [r["introspective_change"] for r in rows]
        y = [r["factual_no_change"] for r in rows]
        model, arm = name.split(" / ")
        ax.plot(x, y, color=SLOT[i], lw=1.2, marker=MARK[i], ms=5, mec="white", mew=1,
                label=f"{model.split(' (')[0]}, {arm} vectors")
        left = x[-1] < 0.35  # keep the Gemma label clear of Qwen's random arm
        label_at(ax, x[-1], y[-1], rows[-1]["dose"], dx=-5 if left else 5, ha="right" if left else "left")
    ax.set_xlim(-0.02, 1)
    ax.set_ylim(-0.02, 1)
    ax.set_xlabel("introspective P(YES) change")
    ax.set_ylabel("factual-NO P(YES) change")
    ax.set_title("The yes-bias appears only at the highest doses", loc="left")
    ax.legend(loc="upper left", fontsize=7)
    save(fig, "fig_yesbias")


def main():
    d = load("results/p2d_dose_transfer.json")
    fig_dose_fraction(d)
    fig_dose_kl(d)
    fig_health_auc()
    fig_yesbias()
    print("wrote", sorted(os.listdir(OUT)))


if __name__ == "__main__":
    main()
