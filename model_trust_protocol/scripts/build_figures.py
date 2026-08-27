"""Render the manuscript's two figures.

The journal requires greyscale or black-and-white exhibits at high resolution
and asks that their number and complexity be kept low. Two are drawn: the
attrition of applications across the six constraints, and the walk-forward
design under which every figure in the paper was produced. Both carry
information that prose carries badly.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

OUT = Path("submission/ijf/figures")
DPI = 600

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9,
        "axes.linewidth": 0.8,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
)

GREY = "0.35"
LIGHT = "0.82"
MID = "0.62"


def figure_2() -> Path:
    """Attrition of seven applications across six ascending constraints."""
    labels = [
        "0  Existence",
        "1  Persistence",
        "2  Availability",
        "3  Competition",
        "4  Detectability",
        "5  Execution",
    ]
    standing = [7, 7, 6, 4, 2, 1]
    lost = [0, 0, 1, 2, 2, 1]

    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    for i, (n, drop) in enumerate(zip(standing, lost, strict=True)):
        width = n / 7
        left = (1 - width) / 2
        shade = LIGHT if drop == 0 else MID
        ax.add_patch(
            mpatches.Rectangle(
                (left, -i - 0.32), width, 0.64,
                facecolor=shade, edgecolor="black", linewidth=0.7,
            )
        )
        ax.text(0.5, -i, str(n), ha="center", va="center", fontsize=10, color="black")
        if drop:
            ax.text(
                left + width + 0.035, -i, f"less {drop}",
                ha="left", va="center", fontsize=8.5, color="black",
            )
        else:
            ax.text(
                left + width + 0.035, -i, "none lost",
                ha="left", va="center", fontsize=8, color=GREY,
            )

    for i, lab in enumerate(labels):
        ax.text(-0.06, -i, lab, ha="right", va="center", fontsize=9)

    ax.text(
        0.5, -len(labels) + 0.42,
        "one application of the record survives every constraint",
        ha="center", va="top", fontsize=8.5, color=GREY, style="italic",
    )
    ax.set_xlim(-0.52, 1.22)
    ax.set_ylim(-len(labels) - 0.1, 0.6)
    ax.axis("off")
    fig.tight_layout()

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "figure_2_attrition.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def figure_1() -> Path:
    """The frozen split, the masked embargo, and the walk-forward vintages."""
    fig, ax = plt.subplots(figsize=(6.4, 3.2))

    x0, x1 = 2011.0, 2025.25
    split, resume = 2021.33, 2021.5

    ax.add_patch(mpatches.Rectangle((x0, 2.35), split - x0, 0.5,
                                    facecolor=LIGHT, edgecolor="black", lw=0.7))
    ax.add_patch(mpatches.Rectangle((split, 2.35), resume - split, 0.5,
                                    facecolor="white", edgecolor="black", lw=0.7,
                                    hatch="////"))
    ax.add_patch(mpatches.Rectangle((resume, 2.35), x1 - resume, 0.5,
                                    facecolor=MID, edgecolor="black", lw=0.7))
    ax.text((x0 + split) / 2, 2.6, "Training era\n618 instruments",
            ha="center", va="center", fontsize=8.5)
    ax.text((resume + x1) / 2, 2.6, "Test era\n851 instruments",
            ha="center", va="center", fontsize=8.5)
    ax.annotate("Two months absent at source,\nused as the embargo",
                xy=(split + 0.09, 2.33), xytext=(split - 2.6, 1.72),
                fontsize=8, color=GREY,
                arrowprops={"arrowstyle": "-", "lw": 0.7, "color": GREY})

    ax.text(x0 - 0.15, 2.6, "Sample", ha="right", va="center", fontsize=9)

    ax.text(x0 - 0.15, 0.89, "Vintages", ha="right", va="center", fontsize=9)
    starts = [2013.4, 2015.4, 2017.4, 2019.4]
    for k, s in enumerate(starts):
        ax.plot([x0, s], [1.28 - k * 0.26] * 2, color="black", lw=1.6,
                solid_capstyle="butt")
        ax.plot(s, 1.28 - k * 0.26, marker="|", color="black", ms=7, mew=1.2)
        ax.annotate("", xy=(s + 1.25, 1.28 - k * 0.26), xytext=(s, 1.28 - k * 0.26),
                    arrowprops={"arrowstyle": "->", "lw": 0.9, "color": GREY})
    ax.text(2012.9, 1.47, "fitted on past data only", fontsize=7.8, color=GREY)
    ax.text(2021.5, 1.02, "applied forward,\nnever restated", fontsize=7.8, color=GREY)
    ax.text(2020.15, 0.30, ". . .", fontsize=11, color=GREY, ha="center")
    ax.text(
        x1 + 0.25, 0.30,
        "106 monthly vintages in all;\nfour shown",
        fontsize=7.6, color=GREY, ha="right", va="center",
    )

    ax.annotate("", xy=(x1, 0.12), xytext=(x0, 0.12),
                arrowprops={"arrowstyle": "->", "lw": 0.9, "color": "black"})
    for yr in (2011, 2014, 2017, 2021, 2025):
        ax.plot([yr, yr], [0.06, 0.18], color="black", lw=0.8)
        ax.text(yr, -0.06, str(yr), ha="center", va="top", fontsize=8)

    ax.set_xlim(x0 - 3.5, x1 + 0.4)
    ax.set_ylim(-0.35, 3.15)
    ax.axis("off")
    fig.tight_layout()

    path = OUT / "figure_1_design.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    for f in (figure_1(), figure_2()):
        print("written", f)
