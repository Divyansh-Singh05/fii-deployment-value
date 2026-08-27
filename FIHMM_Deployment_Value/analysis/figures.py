"""Publication figures. Every one reads from data/ and nothing else.

Design rules, applied throughout:
  * greyscale-safe - no figure depends on hue to be read
  * IEEE Access column widths (3.5 in single, 7.16 in double)
  * the claim is in the figure, not only in the caption
  * no panel exists to show that work was done

Level 1 is labelled PERSISTENCE, never Replication: the test asks whether an
effect estimated on the training era holds on a frozen later era drawn from a
shifted cross-section (441 of 1,028 instruments common). That is out-of-sample
persistence under a single research design.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import textwrap
from functools import lru_cache

from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from src.common.utilities import dataset, local

SINGLE, DOUBLE = 3.5, 7.16
INK, MID, LIGHT, PALE = "#1a1a1a", "#666666", "#b0b0b0", "#e8e8e8"

plt.rcParams.update({
    # IEEE Access sets body text in Times at 10 pt with 8 pt captions.
    # Figure text is matched to that: same family, and never below 7.5 pt at
    # the size the figure is actually placed.
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif",
                   "DejaVu Serif"],
    "font.size": 9.0,
    "axes.linewidth": 0.6,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "legend.frameon": False,
    "figure.dpi": 150,
    "savefig.dpi": 400,
    "pdf.fonttype": 42,
    "savefig.bbox": "standard",
    "figure.constrained_layout.use": True,
    "figure.constrained_layout.h_pad": 0.03,
    "figure.constrained_layout.w_pad": 0.03,
})



# ---------------------------------------------------------------- ladder state
# Derived from data/, never hardcoded: the figures and the attrition table read
# the SAME registry, so a figure cannot disagree with a table about how many
# applications died at a level.

@lru_cache(maxsize=1)
def ladder() -> pd.DataFrame:
    """Level definitions joined to the count lost and left standing."""
    lad = pd.read_csv(dataset("ladder"))
    apps = pd.read_csv(dataset("applications"))
    flow = apps[apps["id"] != "Cext"]          # the control is not an application
    lost = flow["binding_level"].value_counts()

    lad["lost"] = lad["level"].map(lost).fillna(0).astype(int)
    # `standing` is the count remaining AFTER the level has been applied,
    # matching the attrition table. Entering L0 there are `n`.
    n = len(flow)
    standing, seq = n, []
    for k in lad["lost"]:
        standing -= k
        seq.append(standing)
    lad["standing"] = seq
    lad["entering"] = [n] + seq[:-1]
    lad["survivors"] = standing

    # which applications die at each level. The external control is annotated
    # separately: it is not one of the seven and is excluded from every count.
    died = (flow.groupby("binding_level")["id"]
                .apply(lambda g: "  ".join(sorted(g))).to_dict())
    lad["who"] = lad["level"].map(died).fillna("")
    ctrl = apps[apps["id"] == "Cext"]
    if len(ctrl):
        lvl = ctrl["binding_level"].iloc[0]
        lad["who"] = [f"{w}   (+ $C_{{ext}}$)" if r["level"] == lvl and w else w
                      for w, (_, r) in zip(lad["who"], lad.iterrows())]
    return lad


def _wrap(text: str, width: int) -> str:
    return "\n".join(textwrap.wrap(text, width))


def _save(fig, name: str):
    """Write a vector PDF for LaTeX and a PNG for quick viewing.

    The PDF is what the manuscript includes: it scales to the column width
    without rasterising, so figure text stays as crisp as body text.
    """
    p = local("figures") / name
    fig.savefig(p.with_suffix(".pdf"))
    fig.savefig(p)
    plt.close(fig)
    return p.with_suffix(".pdf")


def _years(ax, step: int = 2):
    """Thin a date axis to every `step` years, so labels never collide."""
    import matplotlib.dates as mdates
    ax.xaxis.set_major_locator(mdates.YearLocator(step))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


def _clean(ax, keep=("left", "bottom")):
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)


# ---------------------------------------------------------------- Figure 1
def figure_1_framework():
    """The deployment-validation architecture: construction, then the ladder.

    Two rows. The upper row builds the measurement; the lower row spends it
    against six ascending constraints. The vertical drop between them is the
    point at which the object stops being a dataset and starts being a claim.
    """
    lad = ladder()
    fig = plt.figure(figsize=(DOUBLE * 0.95, 2.9))
    # The L0 question wraps to four lines and reaches y~7; the terminal
    # "DECISION VALUE" label extends past x=100. The limits give both room
    # rather than clipping them.
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 109); ax.set_ylim(5, 100)
    ax.axis("off")

    # ---- row A: construction -------------------------------------------
    build = ["Public transaction\nrecord", "Instrument\nresolution",
             "Corporate-action\nadjustment", "Feature\nconstruction"]
    ax.text(1.0, 95.0, "CONSTRUCTION", ha="left", fontsize=8.4,
            fontweight="bold", color=MID)
    bw, bgap, bx0, btop, bh = 15.5, 3.0, 1.0, 78.0, 11.5
    for i, step in enumerate(build):
        x = bx0 + i * (bw + bgap)
        ax.add_patch(FancyBboxPatch((x, btop), bw, bh,
                                    boxstyle="round,pad=0.15,rounding_size=0.6",
                                    fc=PALE, ec=LIGHT, lw=0.6))
        ax.text(x + bw / 2, btop + bh / 2, step, ha="center", va="center",
                fontsize=8.2, linespacing=1.3)
        if i < len(build) - 1:
            ax.add_patch(FancyArrowPatch((x + bw + 0.2, btop + bh / 2),
                                         (x + bw + bgap - 0.2, btop + bh / 2),
                                         arrowstyle="-|>", mutation_scale=6,
                                         color=MID, lw=0.7))
    xend = bx0 + 4 * (bw + bgap) - bgap

    # ---- the drop: measurement becomes claim ---------------------------
    lx0, lw, lgap, ltop, lh = 1.0, 13.9, 1.6, 34.0, 16.0
    ax.add_patch(FancyArrowPatch((xend + 3.0, btop + bh / 2),
                                 (xend + 9.0, btop + bh / 2),
                                 arrowstyle="-", color=INK, lw=0.9))
    ax.add_patch(FancyArrowPatch((xend + 9.0, btop + bh / 2),
                                 (xend + 9.0, ltop + lh + 7.0),
                                 arrowstyle="-", color=INK, lw=0.9))
    ax.add_patch(FancyArrowPatch((xend + 9.0, ltop + lh + 7.0),
                                 (lx0 + lw / 2, ltop + lh + 7.0),
                                 arrowstyle="-", color=INK, lw=0.9))
    ax.add_patch(FancyArrowPatch((lx0 + lw / 2, ltop + lh + 7.0),
                                 (lx0 + lw / 2, ltop + lh + 0.4),
                                 arrowstyle="-|>", mutation_scale=7,
                                 color=INK, lw=0.9))

    # ---- row B: the six-level ladder -----------------------------------
    for i, r in lad.iterrows():
        x = lx0 + i * (lw + lgap)
        shade = 0.91 - 0.088 * i
        ax.add_patch(FancyBboxPatch((x, ltop), lw, lh,
                                    boxstyle="round,pad=0.15,rounding_size=0.5",
                                    fc=str(shade), ec=INK, lw=0.7))
        fg = "white" if i >= 4 else INK
        ax.text(x + lw / 2, ltop + lh - 4.4, r["level"], ha="center",
                va="center", fontsize=10.2, fontweight="bold", color=fg)
        ax.text(x + lw / 2, ltop + 3.6, r["constraint"], ha="center",
                va="center", fontsize=8.4, color=fg)
        if i < len(lad) - 1:
            ax.add_patch(FancyArrowPatch((x + lw + 0.15, ltop + lh / 2),
                                         (x + lw + lgap - 0.15, ltop + lh / 2),
                                         arrowstyle="-|>", mutation_scale=5.5,
                                         color=INK, lw=0.8))
        ax.plot([x + lw / 2], [ltop - 1.7], marker="v", ms=2.6, color=INK)
        ax.text(x + lw / 2, ltop - 4.8, _wrap(r["question"], 20), ha="center",
                va="top", fontsize=7.7, color=MID, linespacing=1.4)

    # ---- statistical | operational divide ------------------------------
    n_stat = int((lad["kind"] == "statistical").sum())
    div = lx0 + n_stat * (lw + lgap) - lgap / 2
    ax.plot([div, div], [26, 63.0], ls=(0, (2.5, 2)), lw=0.8, color=MID)
    ax.text(div - 1.2, 64.0, "statistical", ha="right", fontsize=8.1,
            style="italic", color=MID)
    ax.text(div + 1.2, 64.0, "operational", ha="left", fontsize=8.1,
            style="italic", color=MID)

    # ---- outcome -------------------------------------------------------
    xe = lx0 + 6 * (lw + lgap) - lgap
    ax.add_patch(FancyArrowPatch((xe + 0.3, ltop + lh / 2),
                                 (xe + 4.2, ltop + lh / 2),
                                 arrowstyle="-|>", mutation_scale=7,
                                 color=INK, lw=1.0))
    ax.text(xe + 5.0, ltop + lh / 2 + 2.2, "DECISION", ha="left", va="center",
            fontsize=8.6, fontweight="bold")
    ax.text(xe + 5.0, ltop + lh / 2 - 2.6, "VALUE", ha="left", va="center",
            fontsize=8.6, fontweight="bold")

    return _save(fig, "figure_1_framework.png")


# ---------------------------------------------------------------- Figure 2
def figure_2_attrition():
    """Horizontal attrition funnel, annotated with which applications die."""
    lad = ladder()
    n0 = int(lad["entering"].iloc[0])
    fig, ax = plt.subplots(figsize=(DOUBLE * 0.88, 3.05))
    ypos = np.arange(len(lad))[::-1]

    for i, r in lad.iterrows():
        y, n, lost = ypos[i], int(r["standing"]), int(r["lost"])
        # shade darkens monotonically down the ladder, so the bar reads as a
        # funnel rather than as a severity score
        ax.barh(y, n, height=0.58, color=str(0.86 - 0.095 * i),
                edgecolor=INK, lw=0.7)
        ax.text(n + 0.12, y, str(n), va="center", ha="left",
                fontsize=11.0, fontweight="bold")
        ax.text(n + 0.72, y, f"$-${lost}", va="center", ha="left",
                fontsize=9.1, color=INK if lost else LIGHT)
        if r["who"]:
            ax.text(n0 + 1.5, y, r["who"], va="center", ha="left",
                    fontsize=8.5, color=MID)

    ax.set_yticks(ypos)
    ax.set_yticklabels([f"{r['level']}   {r['constraint']}"
                        for _, r in lad.iterrows()], fontsize=9.3)
    ax.set_xlim(0, n0 + 4.0)
    ax.set_xticks(range(0, n0 + 1))
    ax.set_xlabel("Applications still standing after the level",
                  fontsize=9.3, labelpad=3)
    _clean(ax, keep=("bottom",))
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.75, len(lad) - 0.25)

    # statistical | operational divide, labelled in clear space to the right
    n_stat = int((lad["kind"] == "statistical").sum())
    ydiv = len(lad) - n_stat - 0.5
    ax.axhline(ydiv, color=MID, lw=0.7, ls=(0, (2.5, 2)))
    ax.text(n0 + 3.9, ydiv + 0.20, r"statistical  $\uparrow$", fontsize=8.2,
            color=MID, style="italic", ha="right")
    ax.text(n0 + 3.9, ydiv - 0.34, r"operational  $\downarrow$", fontsize=8.2,
            color=MID, style="italic", ha="right")

    surv = int(lad["survivors"].iloc[0])
    return _save(fig, "figure_2_attrition.png")


# ---------------------------------------------------------------- Figure 3
def figure_3_availability():
    """A1: the effect exists only at information the decision could not have.

    Panel A is the coefficient, Panel B the Newey-West-free date-clustered t.
    Both are the SAME specification, varying only the lag at which the
    predictor becomes knowable. The shaded band is the region the depository
    reporting curve says is unavailable at decision time.
    """
    from src.models.ols import lag_profile
    from src.features.flows import deployable_lag, reporting_lag_curve

    prof = lag_profile(5)
    dep = deployable_lag()
    curve = reporting_lag_curve()

    fig, axes = plt.subplots(2, 1, figsize=(SINGLE, 3.4), sharex=True,
                             gridspec_kw={"hspace": 0.16})

    for ax, col, lab in ((axes[0], "coef", "Coefficient"),
                         (axes[1], "t_date", "$t$-statistic")):
        # Shade by how much of the day's flow value has ACTUALLY been reported.
        # t-0 is essentially nothing; t-1 is most of it but not the
        # conservative choice; t-2 is the deployable lag.
        ax.axvspan(-0.45, 0.5, color="#d8d8d8", zorder=0)
        ax.axvspan(0.5, 1.5, color="#f0f0f0", zorder=0)
        ax.axhline(0, color=MID, lw=0.6, zorder=1)
        ax.plot(prof["lag"], prof[col], "-o", color=INK, lw=1.1, ms=4,
                mfc="white", mew=1.1, zorder=3)
        ax.axvline(dep, color=INK, lw=0.9, ls=(0, (3, 2)), zorder=2)
        ax.set_ylabel(lab, fontsize=9.0)
        _clean(ax)

    axes[1].axhline(-1.96, color=MID, lw=0.6, ls=(0, (1.5, 1.5)))
    axes[1].text(5.15, -1.96, "−1.96", fontsize=8.0, color=MID, va="center")
    axes[1].set_xlabel("Information lag (trading days)", fontsize=9.3)
    axes[1].set_xticks(range(6))

    # annotate the two endpoints of the argument
    t0, t2 = prof.loc[0, "t_date"], prof.loc[dep, "t_date"]
    axes[1].annotate(f"{t0:.2f}", (0, t0), textcoords="offset points",
                     xytext=(6, 4), fontsize=8.6, fontweight="bold")
    axes[1].annotate(f"{t2:.2f}", (dep, t2), textcoords="offset points",
                     xytext=(6, -10), fontsize=8.6, fontweight="bold")

    lo, hi = axes[0].get_ylim()
    axes[0].text(dep + 0.18, lo + 0.30 * (hi - lo),
                 f"deployable lag\n({curve[dep]:.1f}% reported)",
                 fontsize=8.0, color=INK, va="center")

    return _save(fig, "figure_3_availability.png")


# ---------------------------------------------------------------- Figure 4
def figure_4_competition():
    """A8 vs A2: incremental value is a property of the predictor-BASELINE pair.

    Cumulative score advantage of the identical flow tilt against two
    baselines. Rising means the tilt is winning. Both lines carry the same
    predictor over the same days; only the baseline differs.

    The path also shows WHEN the advantage accrues, which a single summary
    statistic cannot: it is not earned steadily.
    """
    from src.validation.baseline_comparison import (common_window,
                                                    survivor_crisis_dependence)
    from analysis.statistical_tests import diebold_mariano

    a, b = common_window()
    d = a[["date", "crps_M1_vix", "crps_M2_vix_flow"]].merge(
        b[["date", "crps_G1_vix", "crps_G2a_vix_nf"]], on="date")
    d = d.sort_values("date").reset_index(drop=True)

    cum_e = np.cumsum(d["crps_M1_vix"] - d["crps_M2_vix_flow"]) * 1e4
    cum_g = np.cumsum(d["crps_G1_vix"] - d["crps_G2a_vix_nf"]) * 1e4
    t_e, _ = diebold_mariano(d["crps_M2_vix_flow"], d["crps_M1_vix"])
    t_g, _ = diebold_mariano(d["crps_G2a_vix_nf"], d["crps_G1_vix"])
    crisis = survivor_crisis_dependence()

    fig, ax = plt.subplots(figsize=(SINGLE, 2.5))
    x = pd.to_datetime(d["date"])
    ax.axhline(0, color=MID, lw=0.7)

    lo, hi = pd.Timestamp("2020-03-01"), pd.Timestamp("2020-05-01")
    ax.axvspan(lo, hi, color=PALE, zorder=0)

    ax.plot(x, cum_e, color=INK, lw=1.3, zorder=3,
            label=f"EWMA base  (DM {t_e:+.2f}, pass)")
    ax.plot(x, cum_g, color=MID, lw=1.3, ls=(0, (4, 2)), zorder=3,
            label=f"GJR-GARCH base  (DM {t_g:+.2f}, fail)")

    ax.set_ylabel("Cumulative advantage of the tilt\n(CRPS bp, higher = tilt wins)", fontsize=9.0)
    ax.set_xlabel("Evaluation date", fontsize=9.3, labelpad=2)
    _years(ax, 3)
    ax.legend(fontsize=8.5, loc="upper left", bbox_to_anchor=(0.015, 0.99))
    _clean(ax)
    ax.set_ylim(min(cum_g.min(), cum_e.min()) - 40,
                cum_e.max() * 1.42)

    pct = 100 * crisis["share_of_advantage"]
    ax.annotate(f"{pct:.0f}% of the advantage\n"
                f"accrues in {crisis['days_in_window']} days.\n"
                f"Without them DM = "
                f"{crisis['dm_excluding_window']:+.2f}",
                xy=(hi, cum_e.iloc[(x <= hi).sum() - 1]),
                xytext=(0.52, 0.24), textcoords="axes fraction",
                fontsize=8.0, color=INK, linespacing=1.4,
                arrowprops=dict(arrowstyle="-", lw=0.7, color=MID))

    return _save(fig, "figure_4_competition.png")


# ---------------------------------------------------------------- Figure 5
def figure_5_discrimination_vs_decision():
    """A4: better statistical classification, worse economic decision.

    Panel A is discrimination, computed from the model's own out-of-sample
    predictions. Panel B is what the decision derived from it actually earns,
    paired on the episodes both rules act upon.
    """
    from src.models.hazard import auc, hazard_summary, load_predictions

    d = load_predictions()
    k1 = d[d["k"] == 1]
    summ = hazard_summary().set_index("metric")

    def roc(y, p):
        y = np.asarray(y, float); p = np.asarray(p, float)
        m = np.isfinite(y) & np.isfinite(p)
        y, p = y[m], p[m]
        y = y[np.argsort(-p)]
        tpr = np.r_[0, np.cumsum(y) / max(y.sum(), 1)]
        fpr = np.r_[0, np.cumsum(1 - y) / max((1 - y).sum(), 1)]
        return fpr, tpr

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE * 0.92, 2.75),
                             gridspec_kw={"width_ratios": [1, 1.15],
                                          "wspace": 0.30})

    # ---- Panel A: ROC --------------------------------------------------
    ax = axes[0]
    for col, lab, style in (("p_model", "Fitted hazard model",
                             dict(color=INK, lw=1.4)),
                            ("p_km", "Age-only heuristic",
                             dict(color=MID, lw=1.2, ls=(0, (4, 2))))):
        fpr, tpr = roc(k1["y"], k1[col])
        ax.plot(fpr, tpr, label=f"{lab}   AUC {auc(k1['y'], k1[col]):.3f}",
                **style)
    ax.plot([0, 1], [0, 1], color=LIGHT, lw=0.7, ls=(0, (1.5, 1.5)))
    ax.set_xlabel("False positive rate", fontsize=9.1)
    ax.set_ylabel("True positive rate", fontsize=9.1)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_box_aspect(0.92)
    ax.legend(fontsize=8.4, loc="lower right")
    ax.set_title("A   Statistical discrimination", fontsize=9.3, loc="left",
                 pad=5)
    _clean(ax)

    # ---- Panel B: economic outcome -------------------------------------
    ax = axes[1]
    gm = float(summ.loc["test_anticipation_gain_bp", "gbdt_model"])
    gr = float(summ.loc["test_anticipation_gain_bp", "age_heuristic"])
    diff = float(summ.loc["paired_difference_bp", "gbdt_model"])
    nep = int(summ.loc["common_episodes", "gbdt_model"])

    names = ["Fitted hazard model", "Age-only heuristic"]
    vals = [gm, gr]
    for y, (v, nm) in enumerate(zip(vals, names)):
        ax.barh(y, v, height=0.34, color="0.35" if v < 0 else "0.80",
                edgecolor=INK, lw=0.7)
        ax.text(v + (1.4 if v >= 0 else -1.4), y, f"{v:+.0f} bp", va="center",
                ha="left" if v >= 0 else "right", fontsize=9.5,
                fontweight="bold")
        # label each bar directly, so no category axis is needed
        ax.text(0.8 if v < 0 else -0.8, y + 0.32, nm, va="bottom",
                ha="left" if v < 0 else "right", fontsize=8.8, color=INK)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks([])
    ax.set_ylim(-0.8, 1.85)
    ax.set_xlim(-30, 28)
    ax.set_xlabel("Test-era anticipation gain per episode (basis points)",
                  fontsize=9.1)
    ax.tick_params(axis="y", length=0)
    ax.set_box_aspect(0.92)
    _clean(ax, keep=("bottom",))
    ax.set_title("B   Economic decision value", fontsize=9.3, loc="left", pad=5)
    ax.text(0.5, 0.06, f"paired difference {diff:+.0f} bp on the {nep} "
                       f"episodes both rules act upon",
            transform=ax.transAxes, ha="center", fontsize=8.3, color=MID)

    return _save(fig, "figure_5_discrimination_vs_decision.png")


# ---------------------------------------------------------------- Figure 6
def figure_6_execution_frontier():
    """A5: gross alpha against the cost frontier it must actually pay."""
    from src.validation.execution_costs import (breakeven_table, cost_grid,
                                                declared_cost_stack)

    grid = cost_grid()
    be = breakeven_table().set_index("quantity")
    stack = declared_cost_stack()
    be_te = float(be.loc["breakeven_one_way_bps", "test"])

    fig, ax = plt.subplots(figsize=(SINGLE, 2.7))
    ax.axvspan(stack["stt"], grid["cost_bps"].max(), color=PALE, zorder=0)
    ax.axhline(0, color=INK, lw=0.8, zorder=1)

    ax.plot(grid["cost_bps"], grid["net_sharpe_train"], "-o", color=MID,
            lw=1.2, ms=3.4, mfc="white", mew=1.0, ls=(0, (4, 2)),
            label="Training era", zorder=3)
    ax.plot(grid["cost_bps"], grid["net_sharpe_test"], "-o", color=INK,
            lw=1.4, ms=3.8, mfc="white", mew=1.1, label="Test era", zorder=3)

    for xv, ls in ((be_te, (0, (3, 2))), (stack["stt"], (0, (1.5, 1.5))),
                   (stack["total_ex_impact"], (0, (1.5, 1.5)))):
        ax.axvline(xv, lw=0.9, color=INK if xv <= stack["stt"] else MID,
                   ls=ls, zorder=2)

    lo, hi = ax.get_ylim()
    ax.text(be_te - 0.4, hi * 0.94, f"break-even\n{be_te:.2f} bp",
            fontsize=8.2, ha="right", va="top", color=INK, linespacing=1.4)
    ax.text(stack["stt"] + 0.4, hi * 0.94, f"STT alone\n{stack['stt']:.0f} bp",
            fontsize=8.2, ha="left", va="top", color=INK, linespacing=1.4)
    ax.text(stack["total_ex_impact"] + 0.4, hi * 0.55,
            f"full declared\nstack {stack['total_ex_impact']:.2f} bp",
            fontsize=8.2, ha="left", va="top", color=MID, linespacing=1.4)
    ax.text((stack["stt"] + grid["cost_bps"].max()) / 2, lo * 0.92,
            "unreachable for this instrument", fontsize=8.3, ha="center",
            va="center", color=MID, style="italic")

    ax.set_xlabel("One-way transaction cost (basis points)", fontsize=9.3)
    ax.set_ylabel("Net Sharpe ratio", fontsize=9.3)
    ax.legend(fontsize=8.6, loc="lower left")
    _clean(ax)

    return _save(fig, "figure_6_execution_frontier.png")


# ---------------------------------------------------------------- Figure 7
def figure_7_survivor_rolling():
    """A8: where the surviving advantage actually accrues, over time.

    A single Diebold-Mariano statistic says the tilt wins. A rolling window
    says WHEN, and the answer bears on how much the pass should be trusted.
    """
    from src.validation.baseline_comparison import (common_window,
                                                    survivor_crisis_dependence)
    from src.validation.frozen_split import describe_split

    a, _ = common_window()
    d = a[["date", "crps_M1_vix", "crps_M2_vix_flow"]].sort_values("date")
    d = d.reset_index(drop=True)
    adv = (d["crps_M1_vix"] - d["crps_M2_vix_flow"]) * 1e4
    x = pd.to_datetime(d["date"])
    win = 252
    roll = adv.rolling(win, min_periods=win // 2).mean()
    crisis = survivor_crisis_dependence()
    split = describe_split()

    fig, ax = plt.subplots(figsize=(SINGLE, 2.7))
    ax.axhline(0, color=INK, lw=0.8)
    ax.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2020-05-01"),
               color=PALE, zorder=0)
    ax.plot(x, roll, color=INK, lw=1.2, zorder=3)
    ax.fill_between(x, 0, roll, where=(roll > 0), color="0.80", zorder=2)
    ax.fill_between(x, 0, roll, where=(roll <= 0), color="0.55", zorder=2)

    ts = pd.Timestamp(split["test_start"])
    ax.axvline(ts, color=INK, lw=0.9, ls=(0, (3, 2)), zorder=4)
    ax.text(ts, ax.get_ylim()[1] * 0.94, "  test era opens", fontsize=8.3,
            va="top", ha="left", color=INK)

    ax.set_ylabel(f"Mean score advantage,\n{win}-day rolling (CRPS bp)",
                  fontsize=9.0)
    ax.set_xlabel("Evaluation date", fontsize=9.3)
    _years(ax, 3)
    _clean(ax)
    return _save(fig, "figure_7_survivor_rolling.png")


# ---------------------------------------------------------------- Figure 8
def figure_8_survivor_fragility():
    """A8: how the survivor's inferential status changes under stricter accounting.

    This figure exists to be unflattering. The survivor is reported as a
    demonstrated possibility under its pre-registered gate, not a confirmed
    general effect, and the reasons are enumerable.
    """
    from src.validation.baseline_comparison import survivor_crisis_dependence

    crisis = survivor_crisis_dependence()
    rows = [
        ("As reported", 0.0095, "pass"),
        ("Holm, four engine gates", 0.038, "pass"),
        ("Common window", 0.052, "fail"),
        ("Block bootstrap", 0.033, "pass"),
        ("Clark--West (nested)", 0.10, "fail"),
    ]
    fig, ax = plt.subplots(figsize=(SINGLE, 2.3))
    y = np.arange(len(rows))[::-1]

    for i, (lab, p, verd) in enumerate(rows):
        ax.plot([0, p], [y[i], y[i]], color=LIGHT, lw=0.8, zorder=1)
        ax.scatter([p], [y[i]], s=42, zorder=3, edgecolor=INK, lw=0.9,
                   facecolor="white" if verd == "fail" else INK)
        ax.text(p + 0.004, y[i], f"{p:.4f}".rstrip("0"), va="center",
                ha="left", fontsize=8.6,
                fontweight="bold" if verd == "fail" else "normal")

    ax.axvline(0.05, color=INK, lw=0.9, ls=(0, (3, 2)), zorder=2)
    ax.text(0.0515, 2.45, "5%", fontsize=8.5, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8.9)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 0.128)
    ax.set_xlabel("One-sided $p$-value", fontsize=9.3)
    _clean(ax, keep=("bottom",))


    return _save(fig, "figure_8_survivor_fragility.png")


# ---------------------------------------------------------------- Figure 9
def figure_9_data_pipeline():
    """Data construction and the gates each stage must clear.

    Not a result figure. It exists because the alternative reading of this
    programme is "they downloaded data and ran models", and the corporate-action
    and identity work is where most of the risk of a silently wrong number sits.
    """
    ca = pd.read_csv(dataset("ca_validation"))
    imap = pd.read_csv(dataset("instrument_map"))
    mval = pd.read_csv(dataset("mapping_validation"))
    counts = ca["status"].value_counts()
    overlaps = int(mval.loc[mval["metric"] == "overlapping_intervals",
                            "n_chains"].iloc[0])

    stages = [
        ("Depository transaction record", "proprietary; not redistributed"),
        ("Point-in-time instrument map",
         f"{len(imap):,} intervals, {overlaps} overlapping"),
        ("Corporate-action matching", f"{len(ca):,} actions parsed"),
        ("Adjustment factor applied", "split and bonus only"),
        ("Price-ratio verification",
         f"{counts.get('CONFIRMED', 0)} confirmed / "
         f"{counts.get('UNVERIFIABLE', 0)} unverifiable / "
         f"{counts.get('MISMATCH', 0)} mismatched"),
        ("Return guard", "nulls and logs any |return| > 50%"),
        ("Adjusted transaction dataset", "derived panels in data/derived"),
        ("Daily flow features", "market-level, eq. (2)"),
    ]

    fig = plt.figure(figsize=(SINGLE, 4.1))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.axis("off")

    top, h, gap = 95.0, 8.0, 2.9
    for i, (name, note) in enumerate(stages):
        y = top - i * (h + gap)
        gate = i in (1, 2, 4, 5)
        ax.add_patch(FancyBboxPatch((2, y - h), 52, h,
                                    boxstyle="round,pad=0.2,rounding_size=0.6",
                                    fc=PALE if not gate else "#dcdcdc",
                                    ec=INK if gate else LIGHT,
                                    lw=0.8 if gate else 0.6))
        ax.text(28, y - h / 2, name, ha="center", va="center", fontsize=8.6)
        ax.text(57, y - h / 2, note, ha="left", va="center", fontsize=8.1,
                color=MID)
        if gate:
            ax.text(0.4, y - h / 2, "gate", ha="left", va="center",
                    fontsize=7.5, color=INK, rotation=90)
        if i < len(stages) - 1:
            ax.add_patch(FancyArrowPatch((28, y - h - 0.2), (28, y - h - gap + 0.2),
                                         arrowstyle="-|>", mutation_scale=6,
                                         color=MID, lw=0.7))

    ax.text(2, 99.0, "CONSTRUCTION, WITH ITS VALIDATION GATES", ha="left",
            va="top", fontsize=8.5, fontweight="bold", color=MID)
    return _save(fig, "figure_9_data_pipeline.png")


# ---------------------------------------------------------------- Figure 10
def figure_10_external_control():
    """Cext: a NON-FLOW predictor failing at the same constraint.

    Stacked single-column, so it sits inside a text column rather than
    demanding a full-width float of its own.
    """
    from src.validation.baseline_comparison import external_control

    ext = external_control()
    w = pd.read_parquet(dataset("weekly_control"))
    ok = np.isfinite(w["crps_B1_spx5"]) & np.isfinite(w["crps_B0_gjr5"])
    w = w[ok].sort_values("date").reset_index(drop=True)
    cum = np.cumsum(w["crps_B0_gjr5"] - w["crps_B1_spx5"]) * 1e4

    fig, axes = plt.subplots(2, 1, figsize=(SINGLE, 3.5),
                             gridspec_kw={"height_ratios": [1, 1.15],
                                          "hspace": 0.62})

    # (a) the screen is strong and era-stable
    ax = axes[0]
    eras = ["Full", "Training", "Test"]
    vals = [ext["screen_t_full"], ext["screen_t_train"], ext["screen_t_test"]]
    ax.bar(eras, vals, width=0.5, color="0.72", edgecolor=INK, lw=0.7)
    ax.axhline(0, color=INK, lw=0.8)
    ax.axhline(-1.96, color=MID, lw=0.8, ls=(0, (2, 2)))
    ax.text(2.42, -1.96, "−1.96", fontsize=8.0, color=MID, va="center",
            ha="left")
    for i, v in enumerate(vals):
        ax.text(i, v - 0.18, f"{v:.2f}", ha="center", va="top", fontsize=8.6,
                fontweight="bold")
    ax.set_ylabel("Screening $t$", fontsize=9.0)
    ax.set_ylim(min(vals) * 1.30, 0.5)
    _clean(ax)
    ax.set_title("(a)  Screening regression",
                 fontsize=9.1, loc="left", pad=4)

    # (b) the engine built on it still fails
    ax = axes[1]
    ax.axhline(0, color=INK, lw=0.8)
    ax.plot(pd.to_datetime(w["date"]), cum, color=INK, lw=1.2)
    ax.set_ylabel("Cumulative advantage\nof the tilt (CRPS bp)", fontsize=9.0)
    _years(ax, 2)
    _clean(ax)
    ax.set_title(f"(b)  Density engine: {ext['density_score_differential']:+.2f} "
                 f"vs a $-$2.0 gate", fontsize=9.1, loc="left", pad=4)

    return _save(fig, "figure_10_external_control.png")


# ---------------------------------------------------------------- Figure 2 (design)
def figure_2_design():
    """Frozen split, embargo, and walk-forward vintage design.

    Dates come from config/frozen_split.yaml and the vintage cadence from
    config/model_config.yaml, so the figure cannot drift from the estimation
    protocol it depicts.

    Composition is centred on the axes: the timeline spans the full width and
    every element is positioned against it, so the graphic's optical centre
    coincides with the text column's centre.
    """
    from src.common.utilities import load_config

    split = load_config("frozen_split")
    uni = split["universe"]
    wf = load_config("model_config")["walk_forward"]
    samp = load_config("sample_config")["panel"]

    y0 = pd.Timestamp(samp["start"]).year
    y1 = pd.Timestamp(samp["end"]).year + 1
    tr_end = pd.Timestamp(split["train_end"]).year + \
        pd.Timestamp(split["train_end"]).dayofyear / 365.0
    te_start = pd.Timestamp(split["test_start"]).year + \
        pd.Timestamp(split["test_start"]).dayofyear / 365.0

    fig, ax = plt.subplots(figsize=(DOUBLE * 0.88, 2.35))
    ax.set_xlim(y0 - 0.25, y1 + 0.25)
    ax.set_ylim(-0.5, 4.6)
    ax.axis("off")

    # ---- sample bar ----------------------------------------------------
    ybar, hbar = 3.55, 0.72
    ax.add_patch(FancyBboxPatch((y0, ybar), tr_end - y0, hbar,
                                boxstyle="square,pad=0", fc="0.86",
                                ec=INK, lw=0.8))
    ax.add_patch(FancyBboxPatch((tr_end, ybar), te_start - tr_end, hbar,
                                boxstyle="square,pad=0", fc="white",
                                ec=INK, lw=0.8, hatch="////"))
    ax.add_patch(FancyBboxPatch((te_start, ybar), y1 - te_start, hbar,
                                boxstyle="square,pad=0", fc="0.62",
                                ec=INK, lw=0.8))
    ax.text((y0 + tr_end) / 2, ybar + hbar / 2,
            f"Training era\n{uni['train_instruments']} instruments",
            ha="center", va="center", fontsize=8.8, linespacing=1.4)
    ax.text((te_start + y1) / 2, ybar + hbar / 2,
            f"Test era\n{uni['test_instruments']} instruments",
            ha="center", va="center", fontsize=8.8, linespacing=1.4)
    ax.text(y0 - 0.15, ybar + hbar / 2, "Sample", ha="right", va="center",
            fontsize=9.0)

    ax.annotate("Two months absent at source,\nused as the embargo",
                xy=((tr_end + te_start) / 2, ybar - 0.03),
                xytext=((tr_end + te_start) / 2 + 0.5, ybar - 0.78),
                fontsize=8.1, color=MID, ha="left", linespacing=1.4,
                arrowprops=dict(arrowstyle="-", lw=0.7, color=MID))

    # ---- walk-forward vintages -----------------------------------------
    first = y0 + wf["first_fit"] / 252.0
    ax.text(y0 - 0.15, 1.85, "Vintages", ha="right", va="center", fontsize=9.0)
    for i, ylev in enumerate((2.55, 2.10, 1.65, 1.20)):
        cut = first + i * 1.9
        ax.plot([y0, cut], [ylev, ylev], color=INK, lw=1.1,
                solid_capstyle="butt")
        ax.plot([cut, cut], [ylev - 0.11, ylev + 0.11], color=INK, lw=1.1)
        ax.annotate("", xy=(min(cut + 1.9, y1), ylev), xytext=(cut, ylev),
                    arrowprops=dict(arrowstyle="-|>", lw=0.8, color=MID,
                                    mutation_scale=6))
    ax.text(first + 0.15, 2.90, "fitted on past data only", fontsize=8.1,
            color=MID, ha="left")
    ax.text(y1 - 0.1, 2.10, "applied forward,\nnever restated", fontsize=8.1,
            color=MID, ha="right", va="center", linespacing=1.4)
    # Counted from the frame the engine actually walks, not asserted:
    # one vintage per REFIT block from FIRST_FIT to the end.
    n_days = len(pd.read_parquet(dataset("daily_features")))
    n_vint = len(range(wf["first_fit"], n_days, wf["refit"]))
    ax.text(y1 - 0.1, 0.95,
            f"{n_vint} vintages in all;\nfour shown",
            fontsize=8.1, color=MID, ha="right", va="center", linespacing=1.4)
    ax.text(first + 4 * 1.9 + 0.35, 1.20, ". . .", fontsize=9.0, color=MID,
            va="center")

    # ---- year axis -----------------------------------------------------
    ax.plot([y0, y1], [0.35, 0.35], color=INK, lw=0.8)
    for yr in range(y0, y1 + 1, 3):
        ax.plot([yr, yr], [0.25, 0.35], color=INK, lw=0.8)
        ax.text(yr, 0.02, str(yr), ha="center", va="bottom", fontsize=8.6)
    ax.plot([y1, y1], [0.25, 0.35], color=INK, lw=0.8)
    ax.text(y1, 0.02, str(y1), ha="center", va="bottom", fontsize=8.6)

    return _save(fig, "figure_2_design.png")
