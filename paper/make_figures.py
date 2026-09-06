"""Figures for the Study 3 preprint, regenerated from the archived per-trial data.

Run from the repository root:  python paper/make_figures.py
Writes PDFs to paper/figures/. No GPU, no network; every number in the paper's figures is
computed here from data/s3/*.jsonl.

Design rules followed (from the dataviz skill): one y-axis per chart, categorical hues in a
fixed order that passes the colour-vision check, distinct markers and dashes so identity
survives greyscale, legend plus a direct label for every series, text in ink not series
colour, thin marks, recessive grid.
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
from rescore_s3 import first_answer, names_concept  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(ROOT, "data", "s3")
OUT = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT, exist_ok=True)

# Validated categorical slots 1-3 (all-pairs safe in light mode), fixed order.
COL = {"real": "#2a78d6", "random": "#eb6834", "shuffle": "#1baf7a"}
MARK = {"real": "o", "random": "s", "shuffle": "^"}
DASH = {"real": "-", "random": "--", "shuffle": ":"}
LABEL = {"real": "concept vector", "random": "random, norm-matched", "shuffle": "shuffled"}
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


def load_forced(name):
    out = {}
    for line in open(os.path.join(DATA, name), encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            out.setdefault((r["framing"], r["alpha"]), {})[r["concept"]] = r["p_yes"]
    return out


F = {k: load_forced(f"s3_forced_{k}.jsonl") for k in ("real", "random", "shuffle")}
GEN = [json.loads(l) for l in open(os.path.join(DATA, "s3_generation.jsonl"), encoding="utf-8")
       if l.strip()]
ALPHAS = [0.0, 2.0, 4.0, 6.0]
PROMPTS = [("introspective", "Introspective prompt"), ("neutral_matched", "Neutral prompt")]
RNG = np.random.default_rng(0)


def boot_ci(values, n=2000):
    v = np.asarray(values)
    idx = RNG.integers(0, len(v), size=(n, len(v)))
    means = v[idx].mean(axis=1)
    return v.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)


def series(kind, framing):
    m, lo, hi = [], [], []
    for a in ALPHAS:
        mu, l, h = boot_ci(list(F[kind][(framing, a)].values()))
        m.append(mu); lo.append(l); hi.append(h)
    return np.array(m), np.array(lo), np.array(hi)


# ---------------------------------------------------------------- Figure 1: dose-response
fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.4), sharey=True)
for ax, (framing, title) in zip(axes, PROMPTS):
    for kind in ("real", "random", "shuffle"):
        m, lo, hi = series(kind, framing)
        ax.fill_between(ALPHAS, lo, hi, color=COL[kind], alpha=0.12, linewidth=0)
        ax.plot(ALPHAS, m, DASH[kind], color=COL[kind], marker=MARK[kind], markersize=4,
                linewidth=1.3, label=LABEL[kind], markeredgecolor="white",
                markeredgewidth=0.6)
        ax.annotate(LABEL[kind].split(",")[0], (ALPHAS[-1], m[-1]), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=6.5, color=INK2)
    ax.set_title(title, loc="left")
    ax.set_xlabel(r"injection strength $\alpha$")
    ax.set_xticks(ALPHAS)
    ax.set_xlim(-0.3, 7.6)
axes[0].set_ylabel("first-token P(YES)")
axes[0].set_ylim(0, 0.7)
axes[0].legend(loc="upper left", handlelength=2.2)
fig.tight_layout(w_pad=1.5)
fig.savefig(os.path.join(OUT, "fig_dose_response.pdf"))
plt.close(fig)

# ---------------------------------------------------------------- Figure 2: readout swing
gen_alphas = sorted({r["alpha"] for r in GEN})
gen_yes = [np.mean([first_answer(r["text"]) == "YES" for r in GEN if r["alpha"] == a])
           for a in gen_alphas]
# The first forced-choice run swept alpha 0,2,4,5,6,8 (introspective prompt; its rows are
# bit-identical to the later run where they overlap), so the readout figure uses it.
SWEEP = load_forced("s3_forced_real_sweep.jsonl")
ft = {a: np.mean(list(SWEEP[("introspective", float(a))].values()))
      for a in gen_alphas if ("introspective", float(a)) in SWEEP}
fig, ax = plt.subplots(figsize=(2.7, 2.3))
ax.plot(gen_alphas, gen_yes, "--", color=COL["random"], marker="s", markersize=4,
        linewidth=1.3, label="generated text: says YES", markeredgecolor="white",
        markeredgewidth=0.6)
xs = sorted(ft)
ax.plot(xs, [ft[a] for a in xs], "-", color=COL["real"], marker="o", markersize=4,
        linewidth=1.3, label="first token: P(YES)", markeredgecolor="white",
        markeredgewidth=0.6)
ax.set_xlabel(r"injection strength $\alpha$")
ax.set_ylabel("proportion")
ax.set_ylim(0, 0.7)
ax.set_xticks(gen_alphas)
ax.legend(loc="upper right")
ax.set_title("Same injection, two readouts", loc="left")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_readout.pdf"))
plt.close(fig)

# ---------------------------------------------------------------- Figure 3: paired concepts
fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.6), sharey=True)
for ax, a in zip(axes, (2.0, 6.0)):
    real = F["real"][("introspective", a)]
    rnd = F["random"][("introspective", a)]
    ks = sorted(real)
    for k in ks:
        up = real[k] > rnd[k]
        ax.plot([0, 1], [rnd[k], real[k]], "-", color=INK2 if up else COL["random"],
                alpha=0.35, linewidth=0.8, zorder=1)
    ax.scatter([0] * len(ks), [rnd[k] for k in ks], s=14, color=COL["random"],
               edgecolor="white", linewidth=0.5, zorder=2, marker="s")
    ax.scatter([1] * len(ks), [real[k] for k in ks], s=14, color=COL["real"],
               edgecolor="white", linewidth=0.5, zorder=2)
    mr, mreal = np.mean(list(rnd.values())), np.mean(list(real.values()))
    ax.plot([0, 1], [mr, mreal], "-", color=INK, linewidth=2, zorder=3)
    ax.annotate(f"mean {mr:.2f}", (0, mr), xytext=(-6, 0), textcoords="offset points",
                ha="right", va="center", fontsize=6.5)
    ax.annotate(f"mean {mreal:.2f}", (1, mreal), xytext=(6, 0), textcoords="offset points",
                ha="left", va="center", fontsize=6.5)
    n_up = sum(real[k] > rnd[k] for k in ks)
    ax.set_title(rf"$\alpha$={a:.0f}: concept higher on {n_up}/{len(ks)}", loc="left")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["random,\nnorm-matched", "concept\nvector"])
    ax.set_xlim(-0.55, 1.55)
    ax.grid(axis="x", visible=False)
axes[0].set_ylabel("first-token P(YES), per concept")
axes[0].set_ylim(-0.02, 1.02)
fig.tight_layout(w_pad=2.5)
fig.savefig(os.path.join(OUT, "fig_paired.pdf"))
plt.close(fig)

# ---------------------------------------------------------------- Figure 4: content-free share
fig, ax = plt.subplots(figsize=(2.7, 2.3))
for (framing, title), kind_marker in zip(PROMPTS, ("o", "s")):
    base = np.mean(list(F["real"][(framing, 0.0)].values()))
    shares = []
    for a in ALPHAS[1:]:
        r = np.mean(list(F["real"][(framing, a)].values())) - base
        c = np.mean([np.mean(list(F[k][(framing, a)].values())) for k in ("random", "shuffle")]) - base
        shares.append(c / r)
    col = COL["real"] if framing == "introspective" else COL["shuffle"]
    ax.plot(ALPHAS[1:], shares, "-", color=col, marker=kind_marker, markersize=4.5,
            linewidth=1.3, label=title.replace(" prompt", ""), markeredgecolor="white",
            markeredgewidth=0.6)
    ax.annotate(f"{shares[-1]:.0%}", (ALPHAS[-1], shares[-1]), xytext=(5, 0),
                textcoords="offset points", va="center", fontsize=6.5, color=INK2)
ax.axhline(1.0, color=INK2, linewidth=0.7, linestyle=(0, (2, 2)))
ax.set_xlabel(r"injection strength $\alpha$")
ax.set_ylabel("share of effect from\ncontent-free vector")
ax.set_ylim(0, 1.05)
ax.set_xticks(ALPHAS[1:])
ax.set_xlim(1.5, 7.2)
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
ax.legend(loc="upper left")
ax.set_title("Content-free share of the shift", loc="left")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_share.pdf"))
plt.close(fig)

print("wrote:", ", ".join(sorted(f for f in os.listdir(OUT) if f.endswith(".pdf"))))
