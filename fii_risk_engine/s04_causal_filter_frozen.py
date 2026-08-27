# [Phase II — see docs/PHASE2_PLAN.md for the pre-registered charter]
from paths import VALIDATION_DATA, ISIN_MAPPING, TRAINED_MODELS  # noqa
# ============================================================================
# MODULE 16A · CAUSAL FILTERING — Phase II foundation
#
# Phase-I labels are full-sequence Viterbi: the state at day t is
# informed by observations AFTER t. Phase II must run on the FILTERED
# posterior P(S_t | x_{1:t}) — forward recursion only, frozen Phase-I
# parameters, no refitting. 13A already de-risked the economics (a
# fully causal rule labeler reproduces Table 1); this stage makes the
# HMM itself causal and re-verifies.
#
# D1 · label fidelity: filtered vs smoothed backbone agreement, flip
#     rate, onset lag at regime starts (informational, no gate)
# A2 · PRE-REGISTERED GATE (economic invariance): Table-1 R2 on fully
#     FILTERED archetype labels keeps D_SHARK_DIST and D_SHARK_ACC at
#     |t|>=2 in BOTH eras, coefficients within +/-50% of Phase-I ->
#     "CAUSAL FOUNDATION CONFIRMED"; else Phase II STOPS here.
# Output: phase2_filtered_states.parquet (posteriors + filtered labels)
# ============================================================================
import json
import numpy as np
import polars as pl

from linearmodels.panel import PanelOLS

DRIVE = VALIDATION_DATA

# ---- frozen Phase-I parameters ------------------------------------------------
mp = json.loads((TRAINED_MODELS / "hmm_backbone_params.json").read_text())
thr = json.loads((TRAINED_MODELS / "overlay_thresholds.json").read_text())
FEATS = mp["features"]
MU = np.array(mp["means"])                     # (3, 4)
SIG2 = np.array(mp["covars_diag"])             # (3, 4) diagonal variances
if SIG2.ndim == 3:                             # hmmlearn stores (k, d, d)
    SIG2 = np.array([np.diag(s) for s in SIG2])
LOGA = np.log(np.array(mp["transmat"]) + 1e-300)
LOGPI = np.log(np.array(mp["startprob"]) + 1e-300)
SNAME = {int(k): v for k, v in mp["state_labels"].items()}
TH_H = thr["hostage_f_entity_s_max"]
TH_SD = thr["shark_dist_f_entity_s_min"]
TH_SA = thr["shark_acc_f_entity_buy_s_min"]
print("frozen params loaded:", FEATS, "| states:", SNAME)
print(f"overlays: HO<{TH_H:+.3f} SD>{TH_SD:+.3f} SA>{TH_SA:+.3f}")

st = (pl.read_parquet(DRIVE / "states_v3.parquet")
        .sort(["cisin", "TR_DATE"]))
X = st.select(FEATS).to_numpy()
# per-observation state log-likelihoods (diagonal Gaussian)
const = -0.5 * (np.log(2 * np.pi * SIG2)).sum(axis=1)      # (3,)
LL = np.stack([const[k]
               - 0.5 * (((X - MU[k]) ** 2) / SIG2[k]).sum(axis=1)
               for k in range(3)], axis=1)                  # (N, 3)

# ---- forward filtering per stock (log-space) ----------------------------------
cis = st["cisin"].to_numpy()
starts = np.flatnonzero(np.r_[True, cis[1:] != cis[:-1]])
ends = np.r_[starts[1:], len(cis)]
POST = np.empty((len(cis), 3))
for s, e in zip(starts, ends):
    a = LOGPI + LL[s]


    
    a -= a.max()
    POST[s] = np.exp(a) / np.exp(a).sum()
    for t in range(s + 1, e):
        # log alpha_t = log sum_i exp(log alpha_{t-1,i} + logA[i,j]) + ll
        m = a[:, None] + LOGA
        mm = m.max(axis=0)
        a = mm + np.log(np.exp(m - mm).sum(axis=0)) + LL[t]
        a -= a.max()
        p = np.exp(a)
        POST[t] = p / p.sum()

fstate_idx = POST.argmax(axis=1)
fstate = np.array([SNAME[i] for i in fstate_idx])
st = st.with_columns(
    pl.Series("p_sell", POST[:, [k for k, v in SNAME.items()
                                 if v == "SELL_REGIME"][0]]),
    pl.Series("p_neutral", POST[:, [k for k, v in SNAME.items()
                                    if v == "NEUTRAL"][0]]),
    pl.Series("p_buy", POST[:, [k for k, v in SNAME.items()
                                if v == "BUY_REGIME"][0]]),
    pl.Series("fstate", fstate))
st = st.with_columns(
    pl.when((pl.col("fstate") == "SELL_REGIME")
            & (pl.col("F_entity_s") < TH_H)).then(pl.lit("HOSTAGE"))
     .when((pl.col("fstate") == "SELL_REGIME")
           & (pl.col("F_entity_s") > TH_SD)).then(pl.lit("SHARK_DIST"))
     .when((pl.col("fstate") == "BUY_REGIME")
           & (pl.col("F_entity_buy_s") > TH_SA)).then(pl.lit("SHARK_ACC"))
     .when(pl.col("fstate") == "NEUTRAL").then(pl.lit("ROBOT"))
     .otherwise(pl.lit("UNTAGGED_DIRECTIONAL")).alias("farch"))

# ---- D1 · fidelity diagnostics --------------------------------------------------
print("\n=== D1 · filtered vs smoothed (Viterbi) ===")
agree_bb = float((st["fstate"] == st["state"]).mean())
agree_ar = float((st["farch"] == st["archetype"]).mean())
print(f"backbone agreement {100*agree_bb:.1f}% | archetype "
      f"{100*agree_ar:.1f}%")
# flip rate: day-to-day state changes, filtered vs smoothed
for c, tag in (("state", "smoothed"), ("fstate", "filtered")):
    fl = st.with_columns(
        (pl.col(c) != pl.col(c).shift(1)).over("cisin").alias("_f"))
    print(f"  {tag:9s} daily flip rate "
          f"{100*float(fl['_f'].drop_nulls().mean()):.2f}%")
# onset lag: for smoothed regime starts, days until filtered agrees
sm = st.select("cisin", "TR_DATE", "state", "fstate").with_columns(
    ((pl.col("state") != pl.col("state").shift(1)).fill_null(True))
    .cum_sum().over("cisin").alias("_r"))
lag = (sm.with_columns(
        pl.int_range(pl.len()).over("cisin", "_r").alias("_i"),
        (pl.col("fstate") == pl.col("state")).alias("_ok"))
       .filter(pl.col("_ok"))
       .group_by("cisin", "_r").agg(pl.col("_i").min().alias("lag")))
print("  onset lag (days until filtered matches a new smoothed regime):"
      f" median {float(lag['lag'].median()):.0f}, "
      f"p90 {float(lag['lag'].quantile(0.9)):.0f}")
print(st.group_by("era", "farch").len()
        .with_columns((100 * pl.col("len")
                       / pl.col("len").sum().over("era")).round(2)
                      .alias("pct")).sort(["era", "farch"]))

# ---- A2 · Table-1 on filtered labels -------------------------------------------
print("\n=== A2 · Table-1 R2 on FILTERED labels ===")
p = (pl.read_parquet(DRIVE / "returns_panel_v3.parquet")
       .select("isin", "date", "close", "volume", "ret_adj",
               "ret_adj_mktadj", "nifty50_ret").sort(["isin", "date"]))
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
    pl.col("to").rolling_mean(window_size=20).over("isin").alias("_toma"),
    (pl.coalesce(pl.col("cum").shift(-20).over("isin"),
                 pl.col("cum").last().over("isin"))
     - pl.col("cum")).alias("post20"))
p = p.with_columns(
    (pl.col("_ilq").rolling_mean(window_size=20).over("isin").shift(1)
     * 1e9 + 1e-9).log().alias("amihud"),
    (pl.col("_toma").shift(1).over("isin") + 1.0).log().alias("logto"),
    (pl.col("volume") * pl.col("close")
     / pl.col("_toma").shift(1).over("isin")).alias("relvol"),
    pl.col("close").log().alias("logclose"))
anch = p.select("isin", "date", "pre20", "post20", "vol20", "relvol",
                "logclose", "beta120", "mom", "amihud", "logto")

DUM = ["D_HOSTAGE", "D_SHARK_ACC", "D_SHARK_DIST", "D_ROBOT"]
CTL = ["beta120", "mombp", "amihud", "logto", "vol20", "relvol",
       "logclose", "logeplen", "pre20bp"]
LBL = {"HOSTAGE": "D_HOSTAGE", "SHARK_ACC": "D_SHARK_ACC",
       "SHARK_DIST": "D_SHARK_DIST", "ROBOT": "D_ROBOT"}

def table1(col):
    runs = st.with_columns(
        ((pl.col(col) != pl.col(col).shift(1)).fill_null(True))
        .cum_sum().over("cisin").alias("_r"))
    ep = runs.group_by("cisin", "_r").agg(
        pl.col(col).first().alias("arch"), pl.col("era").first(),
        pl.col("TR_DATE").last().alias("ed"), pl.len().alias("eplen"))
    ev = ep.join(anch, left_on=["cisin", "ed"],
                 right_on=["isin", "date"], how="inner")
    ev = ev.with_columns(
        pl.col("ed").dt.strftime("%Y-%m").alias("month"),
        (1e4 * pl.col("pre20")).alias("pre20bp"),
        (1e4 * pl.col("mom")).alias("mombp"),
        (1e4 * pl.col("post20")).alias("y20"),
        pl.col("eplen").cast(pl.Float64).log().alias("logeplen"))
    for a, d in LBL.items():
        ev = ev.with_columns((pl.col("arch") == a).cast(pl.Float64)
                             .alias(d))
    out = {}
    for era in ("TRAIN", "TEST"):
        need = ["y20", "cisin", "ed", "month"] + DUM + CTL
        d = (ev.filter(pl.col("era") == era).drop_nulls(need)
               .select(need).to_pandas().set_index(["cisin", "ed"]))
        res = PanelOLS(d["y20"], d[DUM + CTL], entity_effects=True,
                       time_effects=True).fit(
            cov_type="clustered", cluster_entity=True,
            clusters=d[["month"]])
        out[era] = res
    return out

ref = {"TRAIN": {"D_SHARK_DIST": 65.4, "D_SHARK_ACC": -87.9},
       "TEST": {"D_SHARK_DIST": 48.5, "D_SHARK_ACC": -47.6}}
res = table1("farch")
ok = True
for era in ("TRAIN", "TEST"):
    r = res[era]
    print(f"\n FILTERED {era} (n={r.nobs}):")
    for v in ("D_SHARK_DIST", "D_SHARK_ACC", "D_HOSTAGE"):
        pv = float(r.pvalues[v])
        s = "***" if pv < .01 else "**" if pv < .05 else \
            "*" if pv < .10 else ""
        c, t = float(r.params[v]), float(r.tstats[v])
        print(f"   {v:14s}{c:+9.1f}  t={t:+6.2f} {s}")
        if v in ref[era]:
            keep = abs(t) >= 2 and abs(c - ref[era][v]) \
                <= 0.5 * abs(ref[era][v])
            ok = ok and keep

st.select("cisin", "TR_DATE", "era", "p_sell", "p_neutral", "p_buy",
          "fstate", "farch", "state", "archetype").write_parquet(
    DRIVE / "phase2_filtered_states.parquet")
print("\nwrote phase2_filtered_states.parquet")
print("\n" + "=" * 70)
print("VERDICT:", "A2 PASS — CAUSAL FOUNDATION CONFIRMED; proceed to "
      "16B (calibration)" if ok else
      "A2 FAIL — filtered labels lose the economics; Phase II stops "
      "here and reports")
print("=" * 70)
