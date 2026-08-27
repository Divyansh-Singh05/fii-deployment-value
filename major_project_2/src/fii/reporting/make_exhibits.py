# ============================================================================
# PAPER EXHIBITS — every table/figure the write-up needs, saved as files.
#
# Reporting layer ONLY: each exhibit re-uses the exact spec of its certified
# stage (module7b-B for the main table, module6b for the arc, module11-T2
# for PIN, module12 engine metrics for backtests) and exports tidy CSVs,
# LaTeX (booktabs) and PNG figures into outputs/tables and outputs/figures.
#
# VALIDATION: T1 must reproduce Module 7B-B on the REBUILT (post-audit)
# object: TRAIN HOSTAGE +56.7***, TEST HOSTAGE +42.8***, with SHARK_DIST
# insignificant in both eras. If it doesn't, the exhibit layer is wrong —
# the stages remain the truth.
#
# The pre-audit targets this comment used to name (TRAIN SHARK_DIST +65.4,
# SHARK_ACC -87.9; TEST +48.6 / -47.6) were produced by the defective
# concentration feature retracted in AUDIT_RETRACTIONS.md R1 and must not
# be used as a check. Exhibits generated before 2026-08-23 are preserved
# under outputs/tables.PREAUDIT_20260712/ and are NOT paper-eligible.
# ============================================================================
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from linearmodels.panel import PanelOLS

from fii.backtest.engine import metrics
from fii.paths import (FIGURES, ISIN_MAPPING, TABLES, VALIDATION_DATA,
                       ensure_output_tree)

warnings.filterwarnings("ignore")
ensure_output_tree()
DRIVE = VALIDATION_DATA
ARCHES = ["HOSTAGE", "SHARK_ACC", "SHARK_DIST", "ROBOT"]
DUM = ["D_" + a for a in ARCHES]
CTL2 = ["beta120", "mombp", "amihud", "logto", "vol20", "relvol",
        "logclose", "logeplen", "pre20bp"]
STAR = lambda p: "***" if p < .01 else "**" if p < .05 else \
    "*" if p < .10 else ""


def save_tex(df: pd.DataFrame, path: Path, caption: str) -> None:
    lines = ["% auto-generated — do not edit",
             "\\begin{table}[t]\\centering",
             f"\\caption{{{caption}}}",
             "\\begin{tabular}{l" + "r" * (df.shape[1]) + "}",
             "\\toprule",
             " & ".join([""] + list(df.columns)) + " \\\\ \\midrule"]
    for idx, row in df.iterrows():
        lines.append(" & ".join([str(idx)] + [str(v) for v in row])
                     + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# shared: the module7b episode panel (beta-null fix = restored sample)
# ---------------------------------------------------------------------------
def build_episode_panel() -> pl.DataFrame:
    p = (pl.read_parquet(DRIVE / "returns_panel_v3.parquet")
           .select("isin", "date", "close", "volume", "ret_adj",
                   "ret_adj_mktadj", "nifty50_ret")
           .sort(["isin", "date"]))
    p = p.with_columns(pl.col("nifty50_ret").fill_null(0.0))
    p = p.with_columns(
        pl.col("ret_adj_mktadj").clip(-.5, .5).fill_null(0.0).alias("ar"),
        pl.col("ret_adj").clip(-.5, .5).fill_null(0.0).alias("r"),
        (pl.col("close") * pl.col("volume")).alias("to"))
    p = p.with_columns(pl.col("ar").cum_sum().over("isin").alias("cum"))
    W = 120
    p = p.with_columns(
        (pl.col("r") * pl.col("nifty50_ret")).alias("_xy"),
        (pl.col("nifty50_ret") ** 2).alias("_y2"))
    p = p.with_columns(
        pl.col("_xy").rolling_mean(window_size=W).over("isin").alias("_mxy"),
        pl.col("r").rolling_mean(window_size=W).over("isin").alias("_mx"),
        pl.col("nifty50_ret").rolling_mean(window_size=W).over("isin")
          .alias("_my"),
        pl.col("_y2").rolling_mean(window_size=W).over("isin").alias("_my2"))
    p = p.with_columns(
        ((pl.col("_mxy") - pl.col("_mx") * pl.col("_my"))
         / (pl.col("_my2") - pl.col("_my") ** 2))
        .shift(1).over("isin").alias("beta120"))
    p = p.with_columns(
        (pl.col("cum").shift(21).over("isin")
         - pl.col("cum").shift(127).over("isin")).alias("mom"),
        (pl.col("cum").shift(1).over("isin")
         - pl.col("cum").shift(21).over("isin")).alias("pre20"),
        pl.col("ar").rolling_std(window_size=20).over("isin")
          .shift(1).alias("vol20"),
        (pl.col("ar").abs() / (pl.col("to") + 1.0)).alias("_ilq"),
        pl.col("to").rolling_mean(window_size=20).over("isin").alias("_toma"))
    for k in (10, 20, 30, 60):
        p = p.with_columns(
            (pl.coalesce(pl.col("cum").shift(-k).over("isin"),
                         pl.col("cum").last().over("isin"))
             - pl.col("cum")).alias("post" + str(k)))
    p = p.with_columns(
        (pl.col("_ilq").rolling_mean(window_size=20).over("isin").shift(1)
         * 1e9 + 1e-9).log().alias("amihud"),
        (pl.col("_toma").shift(1).over("isin") + 1.0).log().alias("logto"),
        (pl.col("volume") * pl.col("close")
         / pl.col("_toma").shift(1).over("isin")).alias("relvol"),
        pl.col("close").log().alias("logclose"))
    anchors = p.select("isin", "date", "pre20", "post10", "post20",
                       "post30", "post60", "vol20", "relvol", "logclose",
                       "beta120", "mom", "amihud", "logto")
    states = (pl.read_parquet(DRIVE / "states_v3.parquet")
                .sort(["cisin", "TR_DATE"]))
    runs = states.with_columns(
        ((pl.col("archetype") != pl.col("archetype").shift(1))
         .fill_null(True)).cum_sum().over("cisin").alias("_r"))
    runs = runs.group_by("cisin", "_r").agg(
        pl.col("archetype").first(), pl.col("era").first(),
        pl.col("TR_DATE").last().alias("ed"), pl.len().alias("eplen"))
    ev = runs.join(anchors, left_on=["cisin", "ed"],
                   right_on=["isin", "date"], how="inner")
    ev = ev.with_columns(
        pl.col("ed").dt.strftime("%Y-%m").alias("month"),
        (1e4 * pl.col("pre20")).alias("pre20bp"),
        (1e4 * pl.col("mom")).alias("mombp"),
        pl.col("eplen").cast(pl.Float64).log().alias("logeplen"))
    for k in (10, 20, 30, 60):
        ev = ev.with_columns((1e4 * pl.col("post" + str(k)))
                             .alias("y" + str(k)))
    for a in ARCHES:
        ev = ev.with_columns((pl.col("archetype") == a)
                             .cast(pl.Float64).alias("D_" + a))
    return ev


def fit_r2(ev: pl.DataFrame, era: str, ycol: str) -> pd.DataFrame:
    need = [ycol, "cisin", "ed", "month"] + DUM + CTL2
    d = (ev.filter(pl.col("era") == era).drop_nulls(need)
           .select(need).to_pandas().set_index(["cisin", "ed"]))
    res = PanelOLS(d[ycol], d[DUM + CTL2], entity_effects=True,
                   time_effects=True).fit(
        cov_type="clustered", cluster_entity=True, clusters=d[["month"]])
    # full coefficient table (dummies + all controls) for the appendix
    from fii.paths import REGRESSIONS
    full = pd.DataFrame({"coef": res.params, "t": res.tstats,
                         "p": res.pvalues, "se": res.std_errors})
    full.round(4).to_csv(REGRESSIONS / f"panelols_{era}_{ycol}_full.csv")
    rows = []
    for v in DUM:
        rows.append({"era": era, "horizon": ycol, "var": v,
                     "coef_bp": round(float(res.params[v]), 1),
                     "t": round(float(res.tstats[v]), 2),
                     "p": round(float(res.pvalues[v]), 4),
                     "stars": STAR(float(res.pvalues[v])),
                     "n": int(res.nobs)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
def t1_main_table(ev: pl.DataFrame) -> None:
    out = pd.concat([fit_r2(ev, era, "y20")
                     for era in ("TRAIN", "TEST")], ignore_index=True)
    out.to_csv(TABLES / "T1_panel_regression_main.csv", index=False)
    wide = out.pivot(index="var", columns="era", values="coef_bp")
    stars = out.pivot(index="var", columns="era", values="stars")
    disp = wide.round(1).astype(str) + stars
    disp = disp.loc[DUM][["TRAIN", "TEST"]]
    save_tex(disp, TABLES / "T1_panel_regression_main.tex",
             "Post-episode 20-day abnormal returns (bp) — PanelOLS, "
             "stock+date FE, two-way clustered SE, full controls (R2)")
    print("T1 main table:")
    print(disp.to_string())


def t4_horizons(ev: pl.DataFrame) -> None:
    out = pd.concat([fit_r2(ev, era, f"y{k}")
                     for era in ("TRAIN", "TEST")
                     for k in (10, 20, 30, 60)], ignore_index=True)
    out.to_csv(TABLES / "T4_horizons.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, era in zip(axes, ("TRAIN", "TEST")):
        e = out[out.era == era]
        for v, lab in [("D_SHARK_DIST", "SHARK_DIST (conc. sell)"),
                       ("D_SHARK_ACC", "SHARK_ACC (conc. buy)"),
                       ("D_HOSTAGE", "HOSTAGE (dispersed sell)")]:
            s = e[e["var"] == v]
            ks = [int(h[1:]) for h in s.horizon]
            ax.plot(ks, s.coef_bp, marker="o", label=lab)
        ax.axhline(0, color="k", lw=.5)
        ax.set_title(f"{era}"), ax.set_xlabel("horizon (days)")
    axes[0].set_ylabel("post-episode abnormal return (bp)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Reversal builds and persists; dispersed selling stays "
                 "permanent")
    fig.tight_layout()
    fig.savefig(FIGURES / "F2_horizons.png", dpi=200)
    plt.close(fig)
    print("T4/F2 horizons written")


def t2_census() -> None:
    st = pl.read_parquet(DRIVE / "states_v3.parquet")
    cen = (st.group_by("era", "archetype").agg(pl.len().alias("n"))
             .with_columns((100 * pl.col("n")
                            / pl.col("n").sum().over("era"))
                           .round(2).alias("pct"))
             .sort(["era", "archetype"]).to_pandas())
    cen.to_csv(TABLES / "T2_archetype_census.csv", index=False)
    print("T2 census written")


def t3_mechanism_arc() -> None:
    p = (pl.read_parquet(DRIVE / "returns_panel_v3.parquet")
           .select("isin", "date", "volume", "ret_adj_mktadj")
           .sort(["isin", "date"]))
    p = p.with_columns(pl.col("ret_adj_mktadj").clip(-.5, .5)
                       .fill_null(0.0).alias("ar"))
    p = p.with_columns(pl.col("ar").cum_sum().over("isin").alias("cum"),
                       pl.col("volume").rolling_mean(window_size=20)
                       .over("isin").alias("_vma"))
    p = p.with_columns(
        (pl.col("volume") / pl.col("_vma").shift(1).over("isin"))
        .alias("rvol"),
        (pl.col("cum").shift(1).over("isin")
         - pl.col("cum").shift(21).over("isin")).alias("pre20"),
        (pl.coalesce(pl.col("cum").shift(-20).over("isin"),
                     pl.col("cum").last().over("isin"))
         - pl.col("cum")).alias("post20"))
    anchors = p.select("isin", "date", "ar", "cum", "pre20", "post20",
                       "rvol")
    st = (pl.read_parquet(DRIVE / "states_v3.parquet")
            .select("cisin", "TR_DATE", "era", "archetype")
            .sort(["cisin", "TR_DATE"]))
    runs = st.with_columns(
        ((pl.col("archetype") != pl.col("archetype").shift(1))
         .fill_null(True)).cum_sum().over("cisin").alias("_r"))
    runs = runs.group_by("cisin", "_r").agg(
        pl.col("archetype").first(), pl.col("era").first(),
        pl.col("TR_DATE").first().alias("sd"),
        pl.col("TR_DATE").last().alias("ed"))
    ev = (runs.join(anchors, left_on=["cisin", "sd"],
                    right_on=["isin", "date"], how="inner")
              .rename({"ar": "d0", "cum": "cum_s", "pre20": "pre",
                       "rvol": "rvol0"}).drop("post20"))
    ev = ev.join(anchors.select("isin", "date",
                                pl.col("cum").alias("cum_e"),
                                pl.col("post20").alias("post")),
                 left_on=["cisin", "ed"], right_on=["isin", "date"],
                 how="inner")
    ev = ev.with_columns((pl.col("cum_e") - pl.col("cum_s")
                          + pl.col("d0")).alias("epcar"))
    # day-0 volume vs the ALL-LABELED baseline day (module 6b statistic:
    # 1.12-1.13x for SHARK_DIST) — raw medians sit <1 by skew and mislead
    base_rv = (st.join(anchors.select("isin", "date", "rvol"),
                       left_on=["cisin", "TR_DATE"],
                       right_on=["isin", "date"], how="inner")
                 .group_by("era").agg(pl.col("rvol").mean()
                                      .alias("rvol_base")))
    arc = (ev.group_by("era", "archetype").agg(
        pl.len().alias("n"),
        (1e4 * pl.col("pre").mean()).round(0).alias("pre20_bp"),
        (1e4 * pl.col("d0").mean()).round(1).alias("day0_bp"),
        (1e4 * pl.col("epcar").mean()).round(0).alias("epCAR_bp"),
        (1e4 * pl.col("post").mean()).round(0).alias("post20_bp"),
        pl.col("rvol0").mean().alias("_rv"))
        .join(base_rv, on="era")
        .with_columns((pl.col("_rv") / pl.col("rvol_base")).round(3)
                      .alias("rvol0_vs_base"))
        .drop("_rv", "rvol_base")
        .sort(["era", "archetype"]).to_pandas())
    arc.to_csv(TABLES / "T3_mechanism_arc.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(3)
    for ax, era in zip(axes, ("TRAIN", "TEST")):
        e = arc[arc.era == era].set_index("archetype")
        w = 0.25
        for i, a in enumerate(["SHARK_DIST", "SHARK_ACC", "HOSTAGE"]):
            vals = [e.loc[a, "pre20_bp"], e.loc[a, "epCAR_bp"],
                    e.loc[a, "post20_bp"]]
            ax.bar(x + (i - 1) * w, vals, w,
                   label=f"{a} (day-0 vol {e.loc[a,'rvol0_vs_base']:.2f}x base)")
        ax.axhline(0, color="k", lw=.5)
        ax.set_xticks(x, ["pre-20d", "episode", "post-20d"])
        ax.set_title(era), ax.set_ylabel("abnormal return (bp)")
        ax.legend(fontsize=7)
    fig.suptitle("The event arc: pressure, climax volume, then reversal "
                 "only where flow was concentrated")
    fig.tight_layout()
    fig.savefig(FIGURES / "F1_mechanism_arc.png", dpi=200)
    plt.close(fig)
    print("T3/F1 mechanism arc written")


def t5_pin() -> None:
    import statsmodels.api as sm
    pin = pl.read_parquet(DRIVE / "fii_pin_stockyear.parquet")
    st = (pl.read_parquet(ISIN_MAPPING / "stockday_states_calibrated.parquet")
            .select("cisin", "TR_DATE", "era", "archetype"))
    st = st.with_columns(pl.col("TR_DATE").dt.year().alias("yr"))
    sh = st.group_by("cisin", "yr", "era").agg(
        (pl.col("archetype") == "HOSTAGE").mean().alias("sh_host"),
        (pl.col("archetype") == "SHARK_DIST").mean().alias("sh_sd"),
        (pl.col("archetype") == "SHARK_ACC").mean().alias("sh_sa"))
    p = (pl.read_parquet(DRIVE / "returns_panel_v3.parquet")
           .select("isin", "date", "close", "volume"))
    p = p.with_columns((pl.col("close") * pl.col("volume")).alias("to"),
                       pl.col("date").dt.year().alias("yr"))
    toy = (p.group_by("isin", "yr")
             .agg((pl.col("to").mean() + 1.0).log().alias("logto")))
    t2 = (pin.join(sh, on=["cisin", "yr"], how="inner")
             .join(toy, left_on=["cisin", "yr"], right_on=["isin", "yr"],
                   how="left"))
    rows = []
    for era in ("TRAIN", "TEST"):
        e = t2.filter((pl.col("era") == era)
                      & pl.col("logto").is_not_null()).to_pandas()
        X = sm.add_constant(e[["sh_host", "sh_sd", "sh_sa", "logto"]])
        res = sm.OLS(e["pin"], X).fit(
            cov_type="cluster", cov_kwds={"groups": e["cisin"]})
        for v in ("sh_host", "sh_sd", "sh_sa", "logto"):
            rows.append({"era": era, "var": v,
                         "coef": round(float(res.params[v]), 4),
                         "t": round(float(res.tvalues[v]), 2),
                         "p": round(float(res.pvalues[v]), 4),
                         "stars": STAR(float(res.pvalues[v])),
                         "n": int(res.nobs)})
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "T5_pin_loadings.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    labels = {"sh_host": "HOSTAGE share\n(dispersed sell)",
              "sh_sd": "SHARK_DIST share\n(concentrated sell)",
              "sh_sa": "SHARK_ACC share"}
    x = np.arange(3)
    for i, era in enumerate(("TRAIN", "TEST")):
        e = out[(out.era == era) & (out["var"] != "logto")]
        ax.bar(x + (i - .5) * .35, e.coef, .35, label=era)
    ax.set_xticks(x, [labels[v] for v in ("sh_host", "sh_sd", "sh_sa")],
                  fontsize=8)
    ax.set_ylabel("PIN loading"), ax.axhline(0, color="k", lw=.5)
    ax.set_title("Informed-trading probability loads on DISPERSED selling")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "F4_pin_loadings.png", dpi=200)
    plt.close(fig)
    print("T5/F4 PIN written")


def t6_backtests() -> None:
    frames = []
    for f in ("bt12_baselines.parquet", "bt12_hmm.parquet",
              "bt12_style.parquet"):
        fp = DRIVE / f
        if fp.exists():
            frames.append(pl.read_parquet(fp).to_pandas())
    if not frames:
        print("T6 skipped (no bt12 parquets — run --phase backtest)")
        return
    bt = pd.concat(frames, ignore_index=True)
    rows = []
    for (strat, era), g in bt.groupby(["strat", "era"]):
        if era == "MASK":
            continue
        for c in (0.0, 15.0):
            pnl = g.pnl_gross.to_numpy() - c * 1e-4 * g.turnover.to_numpy()
            m = metrics(pnl, g.turnover.to_numpy())
            if m.get("n", 0) < 20:
                continue
            rows.append({"strategy": strat, "era": era, "cost_bps": c,
                         **{k: round(v, 2) for k, v in m.items()}})
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "T6_backtest_metrics.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for strat, g in bt[bt.era == "TEST"].groupby("strat"):
        pnl = (g.pnl_gross - 15e-4 * g.turnover).cumsum()
        ax.plot(pd.to_datetime(g.date), 1e2 * pnl, label=strat, lw=1)
    ax.axhline(0, color="k", lw=.5)
    ax.set_ylabel("cumulative net pnl (% of book)")
    ax.set_title("TEST era, net of 15 bps — nothing survives costs "
                 "(limits to arbitrage)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / "F3_backtest_equity.png", dpi=200)
    plt.close(fig)
    print("T6/F3 backtests written")


if __name__ == "__main__":
    print("=" * 70)
    print("PAPER EXHIBITS -> outputs/tables + outputs/figures")
    print("=" * 70)
    failures = []
    ev = build_episode_panel()
    for fn in (lambda: t1_main_table(ev), lambda: t4_horizons(ev),
               t2_census, t3_mechanism_arc, t5_pin, t6_backtests):
        try:
            fn()
        except Exception as e:  # keep building the rest
            failures.append(f"{fn}: {e!r}")
            print("EXHIBIT FAILED:", e)
    print("=" * 70)
    print("failures:", failures or "none")
    print("VALIDATE T1 against the rebuilt object: TRAIN HOSTAGE +56.7***,"
          " TEST HOSTAGE +42.8***, SHARK_DIST ns in both eras")
    print("  (pre-audit targets SD +65.4 / SA -87.9 are RETRACTED — see"
          " AUDIT_RETRACTIONS.md R1)")
