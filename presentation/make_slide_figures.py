"""Figures for the Review 1 deck.

Slide-scale, not paper-scale: large type, wide aspect, template colours.
Deliberately NOT reusing paper/make_figures.py - those are Study 3 results at
paper-column size, and Review 1 carries no results.

    python presentation/make_slide_figures.py

Writes 300-DPI PNGs to presentation/figures/.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, FancyArrowPatch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)

ORANGE = "#BD582B"
TEAL = "#33CCCC"
TEAL_DK = "#1E9A9A"
NAVY = "#1F3864"
GREY = "#7F7F7F"
LIGHT = "#F2F2F2"
INK = "#1A1A1A"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=300)
    plt.close(fig)
    print("  wrote", os.path.relpath(p))


def _clean(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


# --------------------------------------------------------------- F1 thermometer
def f1_thermometer():
    fig, ax = plt.subplots(figsize=(11, 4.2))
    _clean(ax)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4.2)

    panels = [
        (1.8, "Ice water", "0 °C", 0.16, TEAL_DK, "known"),
        (5.5, "Boiling water", "100 °C", 0.82, ORANGE, "known"),
        (9.2, "The patient", "?", 0.52, GREY, "unknown"),
    ]
    for cx, label, reading, fill, col, kind in panels:
        # stem
        ax.add_patch(FancyBboxPatch((cx - 0.17, 1.15), 0.34, 1.95,
                                    boxstyle="round,pad=0,rounding_size=0.17",
                                    fc="white", ec=INK, lw=2.2, zorder=2))
        # mercury
        h = 1.85 * fill
        ax.add_patch(FancyBboxPatch((cx - 0.10, 1.2), 0.20, max(h, 0.08),
                                    boxstyle="round,pad=0,rounding_size=0.10",
                                    fc=col, ec="none", zorder=3))
        # bulb
        ax.add_patch(Circle((cx, 1.05), 0.36, fc=col, ec=INK, lw=2.2, zorder=4))
        # ticks
        for t in np.linspace(1.4, 2.95, 6):
            ax.plot([cx + 0.18, cx + 0.30], [t, t], color=GREY, lw=1.2, zorder=2)

        ax.text(cx, 0.42, label, ha="center", va="center",
                fontsize=15, color=INK, fontweight="bold")
        ax.text(cx, 3.42, reading, ha="center", va="center",
                fontsize=22, color=col, fontweight="bold")
        ax.text(cx, 3.90, " ".join(kind.upper()), ha="center", va="center",
                fontsize=10, color=GREY, fontweight="bold")

    # bracket over the two calibration panels
    ax.plot([1.8, 1.8, 5.5, 5.5], [0.12, -0.05, -0.05, 0.12],
            color=TEAL_DK, lw=2.4, clip_on=False)
    ax.text(3.65, -0.38, "calibrate here first", ha="center", va="center",
            fontsize=13, color=TEAL_DK, fontweight="bold", clip_on=False)

    ax.annotate("", xy=(8.5, -0.05), xytext=(6.4, -0.05),
                arrowprops=dict(arrowstyle="-|>", lw=2.4, color=INK),
                annotation_clip=False)
    ax.text(9.2, -0.38, "only then, use here", ha="center", va="center",
            fontsize=13, color=INK, clip_on=False)

    save(fig, "f1_thermometer.png")


# ------------------------------------------------------------ F2 three levels
def f2_levels():
    rows = [
        ("UNIT", "Subspace / probe search",
         "“This neuron responds to X”"),
        ("TRAIT", "Persona / steering vectors",
         "“This vector is the honesty trait”"),
        ("SELF", "Introspective prompting",
         "“The model detects an injected thought”"),
    ]
    fig, ax = plt.subplots(figsize=(13.6, 4.6))
    _clean(ax)
    ax.set_xlim(0, 13.6)
    ax.set_ylim(0, 4.6)

    X_LVL, X_TOOL, X_CLAIM, X_CHK = 0.95, 2.95, 6.25, 12.00
    ROW_L, ROW_R = 0.5, 13.1

    for x, h, ha in [(X_LVL, "LEVEL", "left"),
                     (X_TOOL, "READOUT UNDER TEST", "left"),
                     (X_CLAIM, "WHAT IT CLAIMS", "left"),
                     (X_CHK, "CHECKED?", "center")]:
        ax.text(x, 4.10, h, fontsize=12, color=GREY, fontweight="bold", ha=ha)
    ax.plot([ROW_L, ROW_R], [3.92, 3.92], color=TEAL, lw=3)

    cols = [NAVY, TEAL_DK, ORANGE]
    for i, (lvl, tool, claim) in enumerate(rows):
        y = 3.15 - i * 1.05
        ax.add_patch(Rectangle((ROW_L, y - 0.36), ROW_R - ROW_L, 0.86,
                               fc=LIGHT if i % 2 == 0 else "white",
                               ec="none", zorder=0))
        ax.add_patch(Rectangle((ROW_L, y - 0.36), 0.10, 0.86,
                               fc=cols[i], ec="none", zorder=1))
        ax.text(X_LVL, y + 0.08, lvl, fontsize=17, color=cols[i],
                fontweight="bold", va="center")
        ax.text(X_TOOL, y + 0.08, tool, fontsize=13, color=INK, va="center")
        ax.text(X_CLAIM, y + 0.08, claim, fontsize=12.5, color=INK,
                va="center", style="italic")
        ax.text(X_CHK, y + 0.08, "NEVER", fontsize=16, color="#C00000",
                fontweight="bold", va="center", ha="center")

    # divider reserving the CHECKED? band
    ax.plot([11.15, 11.15], [0.72, 3.80], color="#DDDDDD", lw=1.4, zorder=1)

    ax.text(6.8, 0.20,
            "Three levels. Three published families of tool. "
            "Zero checks against a known answer.",
            fontsize=14.5, color=INK, ha="center", fontweight="bold")
    save(fig, "f2_levels.png")


# --------------------------------------------------------------- F3 timeline
def f3_preparations():
    items = [
        ("1939", "Squid giant axon", "wide enough for an\nelectrode inside", NAVY),
        ("1970s", "Aplysia", "few, large, countable\nneurons", TEAL_DK),
        ("1986", "C. elegans", "all 302 neurons\nmapped", ORANGE),
        ("2026", "Language model", "the weights are\nthe answer", "#7030A0"),
    ]
    fig, ax = plt.subplots(figsize=(12.4, 3.9))
    _clean(ax)
    ax.set_xlim(0, 12.4)
    ax.set_ylim(0, 3.9)

    ax.plot([0.9, 11.5], [2.0, 2.0], color=GREY, lw=2.5, zorder=1)
    xs = np.linspace(1.5, 10.9, 4)
    for x, (yr, name, why, col) in zip(xs, items):
        ax.add_patch(Circle((x, 2.0), 0.235, fc=col, ec="white", lw=3.5, zorder=3))
        ax.text(x, 2.62, yr, ha="center", fontsize=15, color=col,
                fontweight="bold")
        ax.text(x, 3.12, name, ha="center", fontsize=15.5, color=INK,
                fontweight="bold")
        ax.text(x, 1.30, why, ha="center", va="top", fontsize=12.5, color=GREY)

    ax.text(6.2, 0.28,
            "Each was chosen because the answer was knowable — "
            "so a new instrument could be proven before it was trusted.",
            ha="center", fontsize=14, color=INK, fontweight="bold")
    save(fig, "f3_preparations.png")


# ----------------------------------------------------------- F4 ground truth
def f4_ground_truth():
    fig, ax = plt.subplots(figsize=(12.0, 4.3))
    _clean(ax)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.3)

    def box(x, y, w, h, label, sub, fc, ec, fs=14):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.02,rounding_size=0.10",
                                    fc=fc, ec=ec, lw=2.2, zorder=2))
        ax.text(x + w / 2, y + h * 0.62, label, ha="center", va="center",
                fontsize=fs, color=INK, fontweight="bold", zorder=3)
        if sub:
            ax.text(x + w / 2, y + h * 0.25, sub, ha="center", va="center",
                    fontsize=11.5, color=GREY, zorder=3)

    def arrow(x0, x1, y=2.55):
        ax.add_patch(FancyArrowPatch((x0, y), (x1, y),
                                     arrowstyle="-|>", mutation_scale=20,
                                     lw=2.2, color=INK, zorder=4))

    box(0.30, 2.05, 2.75, 1.0, "residual stream", "s", "white", GREY, fs=13)
    arrow(3.17, 3.40)
    box(3.55, 2.05, 2.0, 1.0, "w · s", "dot product", "white", NAVY)
    arrow(5.67, 6.30)
    box(6.45, 2.05, 1.7, 1.0, "GELU", "nonlinearity", "white", GREY)
    arrow(8.27, 8.90)
    box(9.05, 2.05, 2.65, 1.0, "activation", "what we observe", LIGHT, GREY)

    # the weight column, highlighted
    ax.add_patch(FancyBboxPatch((3.55, 0.55), 2.0, 0.95,
                                boxstyle="round,pad=0.02,rounding_size=0.10",
                                fc=ORANGE, ec=ORANGE, lw=2, zorder=2))
    ax.text(4.55, 1.20, "w", ha="center", va="center", fontsize=17,
            color="white", fontweight="bold", zorder=3)
    ax.text(4.55, 0.83, "the weight column", ha="center", va="center",
            fontsize=11.5, color="white", zorder=3)
    ax.add_patch(FancyArrowPatch((4.55, 1.55), (4.55, 2.0),
                                 arrowstyle="-|>", mutation_scale=18,
                                 lw=2.2, color=ORANGE, zorder=4))

    ax.text(6.05, 3.72,
            "The neuron’s true direction is not estimated — it is read off the weights.",
            ha="center", fontsize=14.5, color=INK, fontweight="bold")
    ax.text(4.55, 0.18, "known in advance  →  free ground truth",
            ha="center", fontsize=13, color=ORANGE, fontweight="bold")
    save(fig, "f4_ground_truth.png")


# -------------------------------------------------------------- F5 pipeline
def f5_pipeline():
    NL = chr(10)
    stages = [
        ("Stimulus", NL.join(["public-domain", "text"]), NAVY),
        ("Capture", NL.join(["activations at", "a chosen layer"]), NAVY),
        ("Readout", NL.join(["run the tool", "under test"]), TEAL_DK),
        ("Compare", NL.join(["score against", "the known answer"]), ORANGE),
        ("Calibrate", NL.join(["null, required-N,", "interval, flag"]), ORANGE),
        ("Report", NL.join(["one error rate", "per tool"]), "#7030A0"),
    ]
    W, H, Y = 1.95, 1.30, 1.70
    xs = [0.40 + i * (W + 0.22) for i in range(6)]

    for n in range(1, 7):
        fig, ax = plt.subplots(figsize=(13.6, 3.5))
        _clean(ax)
        ax.set_xlim(0, 13.6)
        ax.set_ylim(0, 3.5)
        for i, (x, (name, sub, col)) in enumerate(zip(xs, stages)):
            on = i < n
            ec = col if on else "#E8E8E8"
            tc = INK if on else "#DDDDDD"
            sc = GREY if on else "#E4E4E4"
            ax.add_patch(FancyBboxPatch((x, Y), W, H,
                                        boxstyle="round,pad=0.02,rounding_size=0.12",
                                        fc="white", ec=ec, lw=2.4, zorder=2))
            ax.add_patch(Rectangle((x, Y + H - 0.11), W, 0.11,
                                   fc=ec, ec="none", zorder=3))
            ax.text(x + W / 2, Y + 0.86, name, ha="center", va="center",
                    fontsize=14, color=tc, fontweight="bold", zorder=4)
            ax.text(x + W / 2, Y + 0.38, sub, ha="center", va="center",
                    fontsize=10, color=sc, zorder=4, linespacing=1.35)
            if i < 5:
                ac = INK if i < n - 1 else "#E8E8E8"
                ax.add_patch(FancyArrowPatch((x + W + 0.02, Y + H / 2),
                                             (xs[i + 1] - 0.02, Y + H / 2),
                                             arrowstyle="-|>", mutation_scale=15,
                                             lw=2.0, color=ac, zorder=1))
        if n == 6:
            ax.text(6.8, 0.78,
                    "The same six stages at every level. "
                    "Only the source of the known answer changes.",
                    ha="center", fontsize=13.5, color=INK, fontweight="bold")
            ax.text(6.8, 0.33,
                    "unit: free from the weights          "
                    "trait: plant a direction          self: plant a state",
                    ha="center", fontsize=12, color=GREY)
        save(fig, "f5_pipeline_" + str(n) + ".png")


# ------------------------------------------------------------ F6 literature
def f6_literature():
    strands = [
        ("Unit level", 12, "Tuning curves\nand probes",
         "Rich methods,\nno ground-truth check", NAVY),
        ("Trait level", 11, "Persona and\nsteering vectors",
         "Validated by effect,\nnever by recovery", TEAL_DK),
        ("Self level", 9, "Introspection\nand self-report",
         "Controlled against\nno injection only", ORANGE),
        ("Neuroscience", 8, "Estimators and\ncalibration",
         "Mature discipline,\nnever transferred", "#7030A0"),
    ]
    fig, ax = plt.subplots(figsize=(12.4, 4.6))
    _clean(ax)
    ax.set_xlim(0, 12.4)
    ax.set_ylim(0, 4.6)

    xs = np.linspace(0.7, 9.55, 4)
    w = 2.4
    for x, (name, n, what, gap, col) in zip(xs, strands):
        ax.add_patch(FancyBboxPatch((x, 1.05), w, 3.0,
                                    boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc="white", ec=col, lw=2.4, zorder=2))
        ax.add_patch(Rectangle((x, 3.55), w, 0.5, fc=col, ec="none", zorder=3))
        ax.text(x + w / 2, 3.80, name, ha="center", va="center",
                fontsize=14.5, color="white", fontweight="bold", zorder=4)
        ax.text(x + w / 2, 3.10, f"{n}", ha="center", va="center",
                fontsize=30, color=col, fontweight="bold", zorder=4)
        ax.text(x + w / 2, 2.72, "papers", ha="center", va="center",
                fontsize=11, color=GREY, zorder=4)
        ax.text(x + w / 2, 2.28, what, ha="center", va="center",
                fontsize=12, color=INK, zorder=4)
        ax.plot([x + 0.3, x + w - 0.3], [1.90, 1.90], color="#E0E0E0", lw=1.5,
                zorder=4)
        ax.text(x + w / 2, 1.48, gap, ha="center", va="center",
                fontsize=11.5, color=ORANGE, style="italic", zorder=4)

    ax.text(6.2, 0.52,
            "≈ 40 papers surveyed. Every strand is missing the same thing: "
            "a case where the answer was already known.",
            ha="center", fontsize=14, color=INK, fontweight="bold")
    save(fig, "f6_literature.png")


# ----------------------------------------------------------------- F7 gantt
def f7_gantt():
    tasks = [
        ("Literature survey and framing", 0.0, 1.6, NAVY),
        ("Unit-level preparation", 0.6, 2.2, NAVY),
        ("Calibration protocol", 1.4, 2.4, TEAL_DK),
        ("Self-level preparation", 2.0, 3.6, TEAL_DK),
        ("Trait-level preparation", 3.4, 5.2, ORANGE),
        ("Cross-level analysis", 4.6, 6.4, ORANGE),
        ("Thesis and defence", 6.2, 8.0, "#7030A0"),
    ]
    fig, ax = plt.subplots(figsize=(12.2, 4.4))
    ax.set_xlim(-0.15, 8.15)
    ax.set_ylim(-0.9, len(tasks) - 0.3)

    for i, (name, s, e, col) in enumerate(tasks):
        y = len(tasks) - 1 - i
        ax.add_patch(FancyBboxPatch((s, y - 0.28), e - s, 0.56,
                                    boxstyle="round,pad=0,rounding_size=0.14",
                                    fc=col, ec="none", zorder=3))
        ax.text(-0.25, y, name, ha="right", va="center", fontsize=12.5,
                color=INK)

    for sem, x0, x1 in [("Sem 5", 0, 2), ("Sem 6", 2, 4),
                        ("Sem 7", 4, 6), ("Sem 8", 6, 8)]:
        ax.axvspan(x0, x1, color=LIGHT if (x0 // 2) % 2 == 0 else "white",
                   zorder=0)
        ax.text((x0 + x1) / 2, -0.62, sem, ha="center", fontsize=13,
                color=GREY, fontweight="bold")
        ax.axvline(x0, color="#DDDDDD", lw=1, zorder=1)

    ax.set_yticks([])
    ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8])
    ax.set_xticklabels(["Sep\n2026", "Nov", "Jan\n2027", "Mar", "May",
                        "Sep", "Nov", "Mar\n2028", "May"], fontsize=11)
    ax.tick_params(axis="x", length=0, pad=6, colors=GREY)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_color("#CCCCCC")
    save(fig, "f7_gantt.png")


if __name__ == "__main__":
    print("building slide figures ...")
    f1_thermometer()
    f2_levels()
    f3_preparations()
    f4_ground_truth()
    f5_pipeline()
    f6_literature()
    f7_gantt()
    print("done ->", OUT)
