from fii.paths import VALIDATION_DATA, TRAINED_MODELS  # noqa: E402
# ============================================================================
# MODULE 17C · FACTORIAL HMM — TABLE-1 ECONOMICS vs THE NAIVE HMM
#
# The decisive question: do the FHMM's END-TO-END archetypes (no
# thresholds anywhere) reproduce the transitory-permanent decomposition
# that the naive HMM needed frozen overlay thresholds to deliver?
#
# Protocol is Module 13A's, verbatim in structure: labels are computed
# ON states_v3 (canonical universe), then the paper's Table-1
# regression (R2 spec, PanelOLS stock+date FE, SE clustered stock x
# month) runs on BOTH label sets side by side.  FHMM labels come from
# a frozen-parameter Viterbi decode of states_v3's features per
# (cisin, era) sequence — the same features the naive labels carry.
#
# PRE-REGISTERED GATES AND VERDICTS (written before results):
#  C0 DECODE CONSISTENCY: per-era archetype census on states_v3 within
#     +/-20% RELATIVE of the 17A census for SHARK_DIST/SHARK_ACC/
#     HOSTAGE (the closure must not change what the model says).
#  Verdicts on the Table-1 comparison (SD = D_SHARK_DIST, SA =
#  D_SHARK_ACC, HO = D_HOSTAGE; "within band" = |t| >= 2 AND coef
#  within +/-50% of the naive-HMM coefficient, per era):
#   V1 "FHMM REPLACES THE THRESHOLDS": SD and SA within band in BOTH
#      eras AND HO stays null (|t| < 2) in both eras. -> the learned
#      concentration chain does the work of the frozen quantile cuts;
#      the hybrid becomes a fully generative model.
#   V2 "PARTIAL REPLACEMENT": SD within band in at least one era but
#      V1 not met. -> quantify the gap; thresholds stay for the
#      published spec, FHMM reported as structural evidence.
#   V3 "THRESHOLDS STAND": SD out of band in BOTH eras. -> the
#      factorial channel does not recover the economics; the Module-2
#      starvation finding extends to factorial form (negative result,
#      reported as designed).
# ============================================================================
import json

import numpy as np
import polars as pl

try:
    from linearmodels.panel import PanelOLS
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "linearmodels"])
    from linearmodels.panel import PanelOLS

from fii.models.fhmm_stages.fhmm_core import FactorialGaussianHMM

DRIVE = VALIDATION_DATA
FEATS = ["F_persist", "F_block", "F_entity_s", "F_entity_buy_s"]

# ---- load canonical states + frozen FHMM ------------------------------------
st = (pl.read_parquet(DRIVE / "states_v3.parquet")
        .sort(["cisin", "TR_DATE"]))
print("states_v3:", st.shape)

params = json.loads((TRAINED_MODELS / "fhmm_params.json").read_text())
model = FactorialGaussianHMM.from_chain_params(params)
D_NAME = {int(k): v for k, v in params["d_names"].items()}
C_NAME = {int(k): v for k, v in params["c_names"].items()}
print("frozen FHMM loaded | train loglik "
      f"{params['loglik']:,.0f} | cutoff {params['train_cutoff']}")

# ---- frozen Viterbi decode per (cisin, era) sequence --------------------------
st = st.sort(["cisin", "era", "TR_DATE"])
x = st.select(FEATS).to_numpy()
lens = (st.group_by("cisin", "era", maintain_order=True)
          .agg(pl.len())["len"].to_list())
prod = model.predict(x, lens)
d, c = prod // model.k_c, prod % model.k_c
st = st.with_columns(
    pl.Series("fstate", [D_NAME[int(v)] for v in d]),
    pl.Series("fcstate", [C_NAME[int(v)] for v in c]))
st = st.with_columns(
    pl.when((pl.col("fstate") == "SELL_REGIME")
            & (pl.col("fcstate") == "DISPERSED")).then(pl.lit("HOSTAGE"))
     .when((pl.col("fstate") == "SELL_REGIME")
           & (pl.col("fcstate") == "CONC_SELL"))
     .then(pl.lit("SHARK_DIST"))
     .when((pl.col("fstate") == "BUY_REGIME")
           & (pl.col("fcstate") == "CONC_BUY"))
     .then(pl.lit("SHARK_ACC"))
     .when(pl.col("fstate") == "NEUTRAL").then(pl.lit("ROBOT"))
     .otherwise(pl.lit("UNTAGGED_DIRECTIONAL")).alias("farch"))
st = st.sort(["cisin", "TR_DATE"])

# ---- C0: decode consistency vs 17A census --------------------------------------
from fii.paths import ISIN_MAPPING
fh17a = pl.read_parquet(ISIN_MAPPING / "stockday_states_fhmm.parquet")
print("\n=== C0 · census consistency: states_v3 decode vs 17A ===")
c0 = True
for era in ("TRAIN", "TEST"):
    a = fh17a.filter(pl.col("era") == era)
    b = st.filter(pl.col("era") == era)
    for arch in ("SHARK_DIST", "SHARK_ACC", "HOSTAGE"):
        sa = a.filter(pl.col("archetype") == arch).height / a.height
        sb = b.filter(pl.col("farch") == arch).height / b.height
        ok = sa > 0 and abs(sb - sa) <= 0.20 * sa
        c0 = c0 and ok
        print(f"  {era:5s} {arch:11s} 17A {100*sa:5.2f}% "
              f"v3 {100*sb:5.2f}% -> {'PASS' if ok else 'FAIL'}")
print("C0:", "PASS" if c0 else "FAIL")

# direct-join agreement (reported, not gated: closure remaps some keys)
jj = st.join(fh17a.select("cisin", "TR_DATE",
                          pl.col("archetype").alias("a17")),
             on=["cisin", "TR_DATE"], how="inner")
agr = float((jj["farch"] == jj["a17"]).mean())
print(f"direct-join label agreement (n={jj.height:,}): {100*agr:.1f}%")

# ---- Table-1 machinery (verbatim from Module 13A) --------------------------------
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

def episodes(label_col):
    runs = st.with_columns(
        ((pl.col(label_col) != pl.col(label_col).shift(1))
         .fill_null(True)).cum_sum().over("cisin").alias("_r"))
    runs = runs.group_by("cisin", "_r").agg(
        pl.col(label_col).first().alias("arch"),
        pl.col("era").first(),
        pl.col("TR_DATE").last().alias("ed"), pl.len().alias("eplen"))
    ev = runs.join(anch, left_on=["cisin", "ed"],
                   right_on=["isin", "date"], how="inner")
    ev = ev.with_columns(
        pl.col("ed").dt.strftime("%Y-%m").alias("month"),
        (1e4 * pl.col("pre20")).alias("pre20bp"),
        (1e4 * pl.col("mom")).alias("mombp"),
        (1e4 * pl.col("post20")).alias("y20"),
        pl.col("eplen").cast(pl.Float64).log().alias("logeplen"))
    for a in ("HOSTAGE", "SHARK_ACC", "SHARK_DIST", "ROBOT"):
        ev = ev.with_columns((pl.col("arch") == a).cast(pl.Float64)
                             .alias("D_" + a))
    return ev

def run_r2(ev, era, tag):
    need = ["y20", "cisin", "ed", "month"] + DUM + CTL
    d = (ev.filter(pl.col("era") == era).drop_nulls(need)
           .select(need).to_pandas().set_index(["cisin", "ed"]))
    res = PanelOLS(d["y20"], d[DUM + CTL], entity_effects=True,
                   time_effects=True).fit(
        cov_type="clustered", cluster_entity=True,
        clusters=d[["month"]])
    out = {}
    print(f"\n {tag} {era} (n={res.nobs})")
    for v in DUM:
        pv = float(res.pvalues[v])
        s = "***" if pv < .01 else "**" if pv < .05 else \
            "*" if pv < .10 else ""
        cf, t = float(res.params[v]), float(res.tstats[v])
        out[v] = (cf, t)
        print(f"   {v:14s}{cf:+9.1f}  t={t:+6.2f} {s}")
    return out

print("\n=== T1 · Table-1 regression: naive-HMM labels vs FHMM labels ===")
res = {}
for tag, col in (("HMM ", "archetype"), ("FHMM", "farch")):
    ev = episodes(col)
    for era in ("TRAIN", "TEST"):
        res[(tag, era)] = run_r2(ev, era, tag)

# ---- pre-registered verdict ---------------------------------------------------
print("\n" + "=" * 70)
print("PRE-REGISTERED VERDICT (V1/V2/V3 — see header)")
band = {}
for era in ("TRAIN", "TEST"):
    for v in ("D_SHARK_DIST", "D_SHARK_ACC"):
        ch, th = res[("HMM ", era)][v]
        cf, tf = res[("FHMM", era)][v]
        band[(era, v)] = abs(tf) >= 2 and abs(cf - ch) <= 0.5 * abs(ch)
        print(f"  {era:5s} {v:14s} HMM {ch:+7.1f}(t{th:+5.2f}) "
              f"FHMM {cf:+7.1f}(t{tf:+5.2f}) "
              f"{'WITHIN BAND' if band[(era, v)] else 'OUT OF BAND'}")
ho_null = all(abs(res[('FHMM', era)]['D_HOSTAGE'][1]) < 2
              for era in ("TRAIN", "TEST"))
print(f"  HOSTAGE null (FHMM, both eras): "
      f"{'HOLDS' if ho_null else 'VIOLATED'}")

sd_ok = [band[(e, "D_SHARK_DIST")] for e in ("TRAIN", "TEST")]
if all(band.values()) and ho_null:
    verdict = ("V1 — FHMM REPLACES THE THRESHOLDS: the learned "
               "concentration chain reproduces Table 1 end-to-end")
elif any(sd_ok):
    verdict = ("V2 — PARTIAL REPLACEMENT: gap quantified above; "
               "thresholds stay for the published spec")
else:
    verdict = ("V3 — THRESHOLDS STAND: factorial channel does not "
               "recover the economics (negative result, reported)")
print("\nVERDICT:", verdict)
print("=" * 70)
