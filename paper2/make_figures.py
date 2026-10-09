"""Figures for Paper 2, regenerated from the results files.

    python paper2/make_figures.py

fig_dose_fraction   one fraction of the residual norm, three models: next-token KL
fig_dose_kl         the KL-calibrated grid on Gemma-3-4B: real and on-manifold vectors against
                    off-manifold ones, with the calibration target
fig_health_auc      each standard health check's AUC at telling live from dead vectors, per model
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
    runs = [("Qwen2.5-3B", "results/s2_qwen3b_analysis.json"),
            ("Qwen2.5-7B", "results/s2_qwen7b_analysis.json"),
            ("Gemma-3-4B, KL grid", "results/s2_gemma4b_kl_analysis.json")]
    stats = ["norm", "distinctness", "stability", "probe", "logit steering", "P(YES) shift"]
    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    ax.axvline(0.5, color=INK2, lw=1, ls="--")
    for i, (name, path) in enumerate(runs):
        a = load(path)
        for j, s in enumerate(stats):
            v = a["auc"][s]
            yy = j + (i - 1) * 0.22
            ax.plot(v["ci95"], [yy, yy], color=SLOT[i], lw=1.5, solid_capstyle="round")
            ax.plot(v["auc"], yy, marker=MARK[i], color=SLOT[i], ms=5, mec="white", mew=1,
                    ls="none", label=f"{name} ({a['live']} live / {a['vectors']})" if j == 0 else None)
    ax.set_yticks(range(len(stats)), stats)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("AUC, live vs dead (0.5 = chance)")
    ax.set_title("No standard health check finds the live vectors", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1)
    save(fig, "fig_health_auc")


def main():
    d = load("results/p2d_dose_transfer.json")
    fig_dose_fraction(d)
    fig_dose_kl(d)
    fig_health_auc()
    print("wrote", sorted(os.listdir(OUT)))


if __name__ == "__main__":
    main()
