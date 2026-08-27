# ============================================================================
# MODULE 16 · RESOLVING THE PIN CONFLICT  (audit follow-up, 2026-08-23)
#
# On the corrected concentration feature two results appear to contradict:
#
#   (a) the CAR20 panel regression says DISPERSED selling (HOSTAGE) is
#       followed by a REVERSAL -> transitory -> a liquidity mechanism;
#   (b) the FII-PIN regression says stock-years with a higher HOSTAGE share
#       have HIGHER PIN -> informed -> a permanent mechanism.
#
# Both cannot describe the same thing. Three tests separate them.
#
#  A · IS sh_host EVEN DIFFERENT FROM sh_sd?  In module 11 all three
#      archetype shares load positively and at similar magnitude
#      (TRAIN 0.133 / 0.101 / 0.121). If they are not distinguishable, the
#      loading is a common "FII archetype activity" component relative to
#      the omitted ROBOT/untagged base, not an information signal specific
#      to dispersed selling. Tested as a formal contrast.
#
#  B · BETWEEN-STOCK OR WITHIN-STOCK?  Module 11's T2 is pooled OLS with no
#      stock fixed effects, so it is dominated by cross-sectional variation:
#      small, illiquid, high-asymmetry NAMES may both carry high PIN and
#      attract dispersed FII selling. Adding stock effects asks whether a
#      stock's PIN moves with its own archetype mix over time -- the only
#      version of (b) that could speak to episode informativeness.
#
#  C · THE DECISIVE TEST.  (a) is an event-level, within-stock claim; (b) is
#      a stock-year-level attribute. They only conflict if the reversal is
#      concentrated in HIGH-PIN names. Interact the archetype dummy with a
#      pre-frozen hiPIN indicator inside the SAME panel specification the
#      reversal is measured in. If the reversal lives in LOW-PIN names, the
#      two results describe different populations and there is no conflict.
#
# PRE-COMMITTED READING (stated before running):
#   C is decisive. Interaction ~ 0        -> genuine conflict, unresolved.
#   Interaction < 0 (reversal in lo-PIN)  -> no conflict; the reversal is a
#                                            liquidity effect in low-
#                                            information names, and PIN's
#                                            loading is a stock-type effect.
#   Interaction > 0 (reversal in hi-PIN)  -> the liquidity reading is wrong
#                                            and (b) wins.
# ============================================================================
from fii.paths import VALIDATION_DATA, ISIN_MAPPING  # noqa: E402

import numpy as np
import polars as pl
import statsmodels.api as sm
from linearmodels.panel import PanelOLS
from scipy.stats import norm

DRIVE, MODELD = VALIDATION_DATA, ISIN_MAPPING
ERAS = ("TRAIN", "TEST")

# --------------------------------------------------------------- shared data
exec(open("src/fii/validation/_pin_conflict_panel.py").read())   # builds `ev`

pin = pl.read_parquet(DRIVE / "fii_pin_stockyear.parquet").filter("ok")
st = pl.read_parquet(DRIVE / "states_v3.parquet")

print("=" * 74)
print("A · ARE THE ARCHETYPE SHARES DISTINGUISHABLE IN THE PIN REGRESSION?")
print("=" * 74)
sh = (st.with_columns(pl.col("TR_DATE").dt.year().alias("yr"))
        .group_by("cisin", "yr", "era")
        .agg((pl.col("archetype") == "HOSTAGE").mean().alias("sh_host"),
             (pl.col("archetype") == "SHARK_DIST").mean().alias("sh_sd"),
             (pl.col("archetype") == "SHARK_ACC").mean().alias("sh_sa")))
toy = (ev.select(pl.col("cisin"), pl.col("ed").dt.year().alias("yr"),
                 pl.col("logto"))
         .group_by("cisin", "yr").agg(pl.col("logto").mean()))
t2 = pin.join(sh, on=["cisin", "yr"], how="inner").join(
    toy, on=["cisin", "yr"], how="left").drop_nulls("logto")

for era in ERAS:
    e = t2.filter(pl.col("era") == era).to_pandas()
    X = sm.add_constant(e[["sh_host", "sh_sd", "sh_sa", "logto"]])
    r = sm.OLS(e["pin"], X).fit(cov_type="cluster",
                                cov_kwds={"groups": e["cisin"]})
    d = r.params["sh_host"] - r.params["sh_sd"]
    V = r.cov_params()
    se = np.sqrt(V.loc["sh_host", "sh_host"] + V.loc["sh_sd", "sh_sd"]
                 - 2 * V.loc["sh_host", "sh_sd"])
    t = d / se
    print(f"\n {era} (n={int(r.nobs)} stock-years)")
    for v in ("sh_host", "sh_sd", "sh_sa"):
        print(f"   {v:9s} {r.params[v]:+.4f}  t={r.tvalues[v]:+5.2f}")
    print(f"   CONTRAST sh_host - sh_sd = {d:+.4f}  t={t:+5.2f}  "
          f"p={2*(1-norm.cdf(abs(t))):.4f}"
          f"   {'distinguishable' if 2*(1-norm.cdf(abs(t))) < .05 else 'NOT distinguishable'}")

print("\n" + "=" * 74)
print("B · BETWEEN-STOCK OR WITHIN-STOCK?")
print("=" * 74)
for era in ERAS:
    e = t2.filter(pl.col("era") == era).to_pandas()
    n_multi = int((e.groupby("cisin").size() > 1).sum())
    d = e.set_index(["cisin", "yr"])
    x = ["sh_host", "sh_sd", "sh_sa", "logto"]
    try:
        r = PanelOLS(d["pin"], d[x], entity_effects=True).fit(
            cov_type="clustered", cluster_entity=True)
        print(f"\n {era} stock fixed effects (n={int(r.nobs)}, "
              f"{n_multi} stocks with >1 year)")
        for v in x:
            print(f"   {v:9s} {float(r.params[v]):+.4f}  "
                  f"t={float(r.tstats[v]):+5.2f}  p={float(r.pvalues[v]):.4f}")
    except Exception as exc:                       # pragma: no cover
        print(f"  {era}: stock-FE fit failed ({exc})")

print("\n" + "=" * 74)
print("C · DECISIVE — DOES THE REVERSAL LIVE IN HIGH-PIN OR LOW-PIN NAMES?")
print("=" * 74)
# hiPIN frozen on the stock-year, split at the era median (module 11's rule)
pv = pin.select("cisin", "yr", "pin")
ee = (ev.with_columns(pl.col("ed").dt.year().alias("yr"))
        .join(pv, on=["cisin", "yr"], how="inner"))
ee = ee.with_columns(
    (pl.col("pin") > pl.col("pin").median().over("era")).cast(pl.Float64)
    .alias("hiPIN"))
DUM = ["D_HOSTAGE", "D_SHARK_ACC", "D_SHARK_DIST", "D_ROBOT"]
CTL = ["beta120", "mombp", "amihud", "logto", "vol20", "relvol",
       "logclose", "logeplen", "pre20bp"]
for a in ("HOSTAGE", "SHARK_DIST"):
    ee = ee.with_columns((pl.col("D_" + a) * pl.col("hiPIN"))
                         .alias(f"X_{a}_hiPIN"))
INT = ["X_HOSTAGE_hiPIN", "X_SHARK_DIST_hiPIN", "hiPIN"]

for era in ERAS:
    x = DUM + CTL + INT
    need = ["y20", "cisin", "ed", "month"] + x
    d = (ee.filter(pl.col("era") == era).drop_nulls(need)
           .select(need).to_pandas().set_index(["cisin", "ed"]))
    r = PanelOLS(d["y20"], d[x], entity_effects=True,
                 time_effects=True).fit(cov_type="clustered",
                                        cluster_entity=True,
                                        clusters=d[["month"]])
    print(f"\n {era} (n={int(r.nobs)} episodes)")
    for v in ("D_HOSTAGE", "X_HOSTAGE_hiPIN",
              "D_SHARK_DIST", "X_SHARK_DIST_hiPIN"):
        pvl = float(r.pvalues[v])
        s = "***" if pvl < .01 else "**" if pvl < .05 else "*" if pvl < .10 else ""
        print(f"   {v:20s}{float(r.params[v]):+8.1f}  "
              f"t={float(r.tstats[v]):+6.2f} {s}")
    lo = float(r.params["D_HOSTAGE"])
    hi = lo + float(r.params["X_HOSTAGE_hiPIN"])
    print(f"   -> HOSTAGE reversal  lo-PIN {lo:+.1f}bp   hi-PIN {hi:+.1f}bp")

print("\n" + "=" * 74)
print("READ: C's interaction is the decisive coefficient. See the")
print("pre-committed reading in this module's header.")
print("=" * 74)
