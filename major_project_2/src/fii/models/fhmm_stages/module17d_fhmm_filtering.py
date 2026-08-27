from fii.paths import VALIDATION_DATA, TRAINED_MODELS  # noqa: E402
# ============================================================================
# MODULE 17D · FACTORIAL HMM — CAUSAL FILTERING (16A protocol)
#
# 17A/17C labels are full-sequence Viterbi (the state at t sees data
# after t). This stage makes the FHMM causal: forward-filtered product
# posteriors P((d_t, c_t) | x_{1:t}) from the FROZEN 17A parameters —
# forward recursion only, no refitting — then chain marginals
#   p(d_t | x_{1:t}) = sum_c P((d,c)_t | x_{1:t})   (direction nowcast)
#   p(c_t | x_{1:t}) = sum_d P((d,c)_t | x_{1:t})   (concentration nowcast)
# Filtered labels = per-chain argmax; archetypes via the side-aware map
# (NO thresholds anywhere, as in 17A).
#
# D1 · fidelity: filtered vs smoothed agreement (both chains +
#     archetype), flip rates, onset lag (informational, no gate).
# A2F · PRE-REGISTERED GATE (economic invariance, referenced to the
#     SMOOTHED FHMM coefficients from 17C — not to the naive HMM, whose
#     comparison 17C already settled with verdict V3):
#       (a) D_SHARK_ACC keeps |t| >= 2 in BOTH eras and coefficient
#           within +/-50% of 17C smoothed (-101.7 / -109.7);
#       (b) D_SHARK_DIST TRAIN keeps |t| >= 2 and coefficient within
#           +/-50% of +115.7. (TEST SD was already sub-significant for
#           the smoothed FHMM, t=1.30, so no significance bar is set
#           there; its coefficient is reported.)
#     PASS -> "CAUSAL FHMM FOUNDATION CONFIRMED"; else the FHMM
#     Phase-II battery stops here and reports.
# Output: fhmm_filtered_states.parquet (chain posteriors + labels)
# ============================================================================
import json

import numpy as np
import polars as pl
from linearmodels.panel import PanelOLS

from fii.models.fhmm_stages.fhmm_core import FactorialGaussianHMM

DRIVE = VALIDATION_DATA
FEATS = ["F_persist", "F_block", "F_entity_s", "F_entity_buy_s"]

params = json.loads((TRAINED_MODELS / "fhmm_params.json").read_text())
model = FactorialGaussianHMM.from_chain_params(params)
D_NAME = {int(k): v for k, v in params["d_names"].items()}
C_NAME = {int(k): v for k, v in params["c_names"].items()}
KC = model.k_c
print("frozen FHMM loaded | D:", D_NAME, "| C:", C_NAME)

st = (pl.read_parquet(DRIVE / "states_v3.parquet")
        .sort(["cisin", "TR_DATE"]))
X = st.select(FEATS).to_numpy()

# per-observation product-state log-likelihoods
M = model.means_                                   # (9, 4)
SIG2 = model.sigma2                                # (4,) shared diag
const = -0.5 * (np.log(2 * np.pi * SIG2)).sum()
inv = 1.0 / SIG2
LL = (const - 0.5 * (((X[:, None, :] - M[None, :, :]) ** 2)
                     * inv[None, None, :]).sum(axis=2))   # (N, 9)
LOGA = np.log(model.transmat_ + 1e-300)
LOGPI = np.log(model.startprob_ + 1e-300)

# ---- forward filtering per stock (log-space; 16A recursion, 9 states) ------
cis = st["cisin"].to_numpy()
starts = np.flatnonzero(np.r_[True, cis[1:] != cis[:-1]])
ends = np.r_[starts[1:], len(cis)]
POST = np.empty((len(cis), model.n_components))
for s, e in zip(starts, ends):
    a = LOGPI + LL[s]
    a -= a.max()
    POST[s] = np.exp(a) / np.exp(a).sum()
    for t in range(s + 1, e):
        m = a[:, None] + LOGA
        mm = m.max(axis=0)
        a = mm + np.log(np.exp(m - mm).sum(axis=0)) + LL[t]
        a -= a.max()
        p = np.exp(a)
        POST[t] = p / p.sum()

# chain marginals
PD = POST.reshape(-1, model.k_d, KC).sum(axis=2)   # (N, k_d)
PC = POST.reshape(-1, model.k_d, KC).sum(axis=1)   # (N, k_c)
fstate = np.array([D_NAME[i] for i in PD.argmax(axis=1)])
fcstate = np.array([C_NAME[i] for i in PC.argmax(axis=1)])

def dcol(name):
    return [k for k, v in D_NAME.items() if v == name][0]
def ccol(name):
    return [k for k, v in C_NAME.items() if v == name][0]

st = st.with_columns(
    pl.Series("p_sell", PD[:, dcol("SELL_REGIME")]),
    pl.Series("p_neutral", PD[:, dcol("NEUTRAL")]),
    pl.Series("p_buy", PD[:, dcol("BUY_REGIME")]),
    pl.Series("p_disp", PC[:, ccol("DISPERSED")]),
    pl.Series("p_csell", PC[:, ccol("CONC_SELL")]),
    pl.Series("p_cbuy", PC[:, ccol("CONC_BUY")]),
    pl.Series("fstate", fstate),
    pl.Series("fcstate", fcstate))
st = st.with_columns(
    pl.when((pl.col("fstate") == "SELL_REGIME")
            & (pl.col("fcstate") == "DISPERSED")).then(pl.lit("HOSTAGE"))
     .when((pl.col("fstate") == "SELL_REGIME")
           & (pl.col("fcstate") == "CONC_SELL")).then(pl.lit("SHARK_DIST"))
     .when((pl.col("fstate") == "BUY_REGIME")
           & (pl.col("fcstate") == "CONC_BUY")).then(pl.lit("SHARK_ACC"))
     .when(pl.col("fstate") == "NEUTRAL").then(pl.lit("ROBOT"))
     .otherwise(pl.lit("UNTAGGED_DIRECTIONAL")).alias("farch"))

# ---- smoothed FHMM labels on states_v3 (17C decode, reproduced) --------------
sv = st.sort(["cisin", "era", "TR_DATE"])
xs = sv.select(FEATS).to_numpy()
lens = (sv.group_by("cisin", "era", maintain_order=True)
          .agg(pl.len())["len"].to_list())
prod = model.predict(xs, lens)
d, c = prod // KC, prod % KC
sv = sv.with_columns(
    pl.Series("sstate", [D_NAME[int(v)] for v in d]),
    pl.Series("scstate", [C_NAME[int(v)] for v in c]))
sv = sv.with_columns(
    pl.when((pl.col("sstate") == "SELL_REGIME")
            & (pl.col("scstate") == "DISPERSED")).then(pl.lit("HOSTAGE"))
     .when((pl.col("sstate") == "SELL_REGIME")
           & (pl.col("scstate") == "CONC_SELL")).then(pl.lit("SHARK_DIST"))
     .when((pl.col("sstate") == "BUY_REGIME")
           & (pl.col("scstate") == "CONC_BUY")).then(pl.lit("SHARK_ACC"))
     .when(pl.col("sstate") == "NEUTRAL").then(pl.lit("ROBOT"))
     .otherwise(pl.lit("UNTAGGED_DIRECTIONAL")).alias("sarch"))
st = sv.sort(["cisin", "TR_DATE"])

# ---- D1 · fidelity ------------------------------------------------------------
print("\n=== D1 · filtered vs smoothed (FHMM) ===")
for a, b, tag in (("fstate", "sstate", "chain D"),
                  ("fcstate", "scstate", "chain C"),
                  ("farch", "sarch", "archetype")):
    print(f"  {tag:9s} agreement {100*float((st[a]==st[b]).mean()):.1f}%")
for c_, tag in (("sarch", "smoothed"), ("farch", "filtered")):
    fl = st.with_columns(
        (pl.col(c_) != pl.col(c_).shift(1)).over("cisin").alias("_f"))
    print(f"  {tag:9s} archetype daily flip rate "
          f"{100*float(fl['_f'].drop_nulls().mean()):.2f}%")
sm = st.select("cisin", "TR_DATE", "sstate", "fstate").with_columns(
    ((pl.col("sstate") != pl.col("sstate").shift(1)).fill_null(True))
    .cum_sum().over("cisin").alias("_r"))
lag = (sm.with_columns(
        pl.int_range(pl.len()).over("cisin", "_r").alias("_i"),
        (pl.col("fstate") == pl.col("sstate")).alias("_ok"))
       .filter(pl.col("_ok"))
       .group_by("cisin", "_r").agg(pl.col("_i").min().alias("lag")))
print("  onset lag (chain D): median "
      f"{float(lag['lag'].median()):.0f}, "
      f"p90 {float(lag['lag'].quantile(0.9)):.0f}")
print(st.group_by("era", "farch").len()
        .with_columns((100 * pl.col("len")
                       / pl.col("len").sum().over("era")).round(2)
                      .alias("pct")).sort(["era", "farch"]))

# ---- A2F · Table-1 on filtered FHMM labels -------------------------------------
print("\n=== A2F · Table-1 R2 on FILTERED FHMM labels ===")
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

runs = st.with_columns(
    ((pl.col("farch") != pl.col("farch").shift(1)).fill_null(True))
    .cum_sum().over("cisin").alias("_r"))
ep = runs.group_by("cisin", "_r").agg(
    pl.col("farch").first().alias("arch"), pl.col("era").first(),
    pl.col("TR_DATE").last().alias("ed"), pl.len().alias("eplen"))
ev = ep.join(anch, left_on=["cisin", "ed"],
             right_on=["isin", "date"], how="inner")
ev = ev.with_columns(
    pl.col("ed").dt.strftime("%Y-%m").alias("month"),
    (1e4 * pl.col("pre20")).alias("pre20bp"),
    (1e4 * pl.col("mom")).alias("mombp"),
    (1e4 * pl.col("post20")).alias("y20"),
    pl.col("eplen").cast(pl.Float64).log().alias("logeplen"))
for a, dm in LBL.items():
    ev = ev.with_columns((pl.col("arch") == a).cast(pl.Float64).alias(dm))

# 17C smoothed-FHMM reference coefficients
REF = {"TRAIN": {"D_SHARK_DIST": 115.7, "D_SHARK_ACC": -101.7},
       "TEST": {"D_SHARK_DIST": 37.9, "D_SHARK_ACC": -109.7}}
res = {}
for era in ("TRAIN", "TEST"):
    need = ["y20", "cisin", "ed", "month"] + DUM + CTL
    dd = (ev.filter(pl.col("era") == era).drop_nulls(need)
            .select(need).to_pandas().set_index(["cisin", "ed"]))
    r = PanelOLS(dd["y20"], dd[DUM + CTL], entity_effects=True,
                 time_effects=True).fit(
        cov_type="clustered", cluster_entity=True, clusters=dd[["month"]])
    res[era] = r
    print(f"\n FILTERED FHMM {era} (n={r.nobs}):")
    for v in ("D_SHARK_DIST", "D_SHARK_ACC", "D_HOSTAGE"):
        pv = float(r.pvalues[v])
        s = "***" if pv < .01 else "**" if pv < .05 else \
            "*" if pv < .10 else ""
        cf, t = float(r.params[v]), float(r.tstats[v])
        print(f"   {v:14s}{cf:+9.1f}  t={t:+6.2f} {s} "
              f"(smoothed ref {REF[era].get(v, float('nan')):+.1f})")

# gate evaluation
def cell(era, v):
    r = res[era]
    return float(r.params[v]), float(r.tstats[v])

ok = True
for era in ("TRAIN", "TEST"):
    cf, t = cell(era, "D_SHARK_ACC")
    ok = ok and abs(t) >= 2 and abs(cf - REF[era]["D_SHARK_ACC"]) \
        <= 0.5 * abs(REF[era]["D_SHARK_ACC"])
cf, t = cell("TRAIN", "D_SHARK_DIST")
ok = ok and abs(t) >= 2 and abs(cf - REF["TRAIN"]["D_SHARK_DIST"]) \
    <= 0.5 * abs(REF["TRAIN"]["D_SHARK_DIST"])

st.select("cisin", "TR_DATE", "era",
          "p_sell", "p_neutral", "p_buy",
          "p_disp", "p_csell", "p_cbuy",
          "fstate", "fcstate", "farch",
          "sstate", "scstate", "sarch").write_parquet(
    DRIVE / "fhmm_filtered_states.parquet")
print("\nwrote fhmm_filtered_states.parquet")
print("\n" + "=" * 70)
print("VERDICT:", "A2F PASS — CAUSAL FHMM FOUNDATION CONFIRMED; "
      "proceed to 17E (calibration)" if ok else
      "A2F FAIL — filtered FHMM labels lose even the smoothed FHMM "
      "economics; the FHMM Phase-II battery stops here and reports")
print("=" * 70)
