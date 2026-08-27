# [migrated from Colab: paths now come from fii.paths; see
#  scripts/migrate_colab_modules.py — research logic unchanged]
from fii.paths import VALIDATION_DATA, ISIN_MAPPING  # noqa: E402
# ============================================================================
# MODULE 11 · FII-PIN (Easley-O'Hara) — the informed-trading benchmark
#
# PIN = probability of informed trading, from the EHO mixture model on
# daily buy/sell COUNTS. Ours uses FII trade counts only -> "FII-PIN"
# (stated limitation: classical PIN uses all market orders).
# Estimated per STOCK-YEAR by MLE (Lin-Ke stabilized log-likelihood),
# params: alpha (news prob), delta (bad-news prob), mu (informed rate),
# eb/es (uninformed buy/sell rates). PIN = a*mu / (a*mu + eb + es).
# RUNTIME: several thousand MLEs -> expect ~10-20 min.
#
# PRE-REGISTERED predictions (from the transitory/permanent reframing):
#  T2 KEY: PIN should align with the PERMANENT axis — stock-years with
#     more HOSTAGE days (dispersed, permanent declines = information-
#     consistent) have HIGHER PIN; SHARK_DIST share should have no
#     (or weaker) PIN association (liquidity, not information).
#  T3: the SHARK_DIST reversal should NOT be concentrated in high-PIN
#     names (it is a liquidity effect): D_SHARK_DIST x hiPIN ~ 0.
#  T1 sanity: PIN levels ~0.1-0.35 (literature range); PIN higher in
#     small/illiquid names (negative corr with log turnover).
# ============================================================================
import numpy as np
import polars as pl
from pathlib import Path
from scipy.optimize import minimize
from scipy.special import logsumexp

try:
    from linearmodels.panel import PanelOLS
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "linearmodels"])
    from linearmodels.panel import PanelOLS

DRIVE = VALIDATION_DATA
MODELD = ISIN_MAPPING

# ---- data: FII buy/sell counts on model-universe stock-days ------------------
st = (pl.read_parquet(MODELD / "stockday_states_calibrated.parquet")
        .select("cisin", "TR_DATE", "era", "archetype"))
f = (pl.read_parquet(MODELD / "stockday_features_v2.parquet")
       .select("cisin", "TR_DATE", "n_buys", "n_sells"))
d = st.join(f, on=["cisin", "TR_DATE"], how="inner")
d = d.with_columns(pl.col("TR_DATE").dt.year().alias("yr"))
print("stock-days with counts:", d.height)

# ---- EHO likelihood (Lin-Ke stabilization via logsumexp) ---------------------
def negll(theta, B, S):
    a = 1 / (1 + np.exp(-theta[0]))
    dl = 1 / (1 + np.exp(-theta[1]))
    mu, eb, es = np.exp(theta[2:5])
    def pois(k, lam):                      # log Poisson w/o log(k!)
        return -lam + k * np.log(lam + 1e-12)
    c1 = np.log(1 - a + 1e-12) + pois(B, eb) + pois(S, es)
    c2 = np.log(a * dl + 1e-12) + pois(B, eb) + pois(S, es + mu)
    c3 = np.log(a * (1 - dl) + 1e-12) + pois(B, eb + mu) + pois(S, es)
    return -np.sum(logsumexp(np.vstack([c1, c2, c3]), axis=0))

def fit_pin(B, S):
    mu0 = max(np.mean(np.abs(B - S)), 1.0)
    eb0 = max(np.mean(B) * 0.75, 0.5)
    es0 = max(np.mean(S) * 0.75, 0.5)
    x0 = np.array([0.0, 0.0, np.log(mu0), np.log(eb0), np.log(es0)])
    try:
        r = minimize(negll, x0, args=(B, S), method="L-BFGS-B",
                     options={"maxiter": 300})
        a = 1 / (1 + np.exp(-r.x[0]))
        mu, eb, es = np.exp(r.x[2:5])
        pin = a * mu / (a * mu + eb + es)
        return pin, a, mu, eb + es, bool(r.success)
    except Exception:
        return np.nan, np.nan, np.nan, np.nan, False

print("estimating PIN per stock-year (min 60 days)... ~10-20 min")
rows = []
pdf = d.select("cisin", "yr", "n_buys", "n_sells").to_pandas()
for (cis, yr), g in pdf.groupby(["cisin", "yr"], sort=False):
    B = g["n_buys"].to_numpy(dtype=float)
    S = g["n_sells"].to_numpy(dtype=float)
    if len(B) < 60:
        continue
    pin, a, mu, eps, ok = fit_pin(B, S)
    rows.append((cis, int(yr), pin, a, mu, eps, ok, len(B)))
import pandas as pd
pin = pl.from_pandas(pd.DataFrame(
    rows, columns=["cisin", "yr", "pin", "alpha", "mu", "eps",
                   "ok", "ndays"]))
pin = pin.filter(pl.col("ok") & pl.col("pin").is_finite())
print("PIN estimated for", pin.height, "stock-years,",
      pin["cisin"].n_unique(), "stocks")

# ---- T1 sanity ----------------------------------------------------------------
print("\n=== T1 · sanity ===")
print("PIN distribution:")
print(pin["pin"].describe())
# liquidity link: mean turnover per stock-year from the panel
p = (pl.read_parquet(DRIVE / "returns_panel_v3.parquet")
       .select("isin", "date", "close", "volume"))
p = p.with_columns((pl.col("close") * pl.col("volume")).alias("to"),
                   pl.col("date").dt.year().alias("yr"))
toy = (p.group_by("isin", "yr")
         .agg((pl.col("to").mean() + 1.0).log().alias("logto")))
px = pin.join(toy, left_on=["cisin", "yr"], right_on=["isin", "yr"],
              how="inner")
from scipy.stats import spearmanr
rho = spearmanr(px["pin"].to_numpy(), px["logto"].to_numpy()).statistic
print("corr(PIN, log turnover):", round(float(rho), 3),
      "(expect NEGATIVE: informed trading prob higher in small names)")

# ---- T2 KEY: PIN vs archetype composition (stock-year level) -----------------
print("\n=== T2 · KEY: PIN vs archetype shares (stock-year) ===")
sh = (d.group_by("cisin", "yr", "era")
        .agg(pl.len().alias("n"),
             (pl.col("archetype") == "HOSTAGE").mean().alias("sh_host"),
             (pl.col("archetype") == "SHARK_DIST").mean().alias("sh_sd"),
             (pl.col("archetype") == "SHARK_ACC").mean().alias("sh_sa")))
t2 = pin.join(sh, on=["cisin", "yr"], how="inner")
t2 = t2.join(toy, left_on=["cisin", "yr"], right_on=["isin", "yr"],
             how="left")
import statsmodels.api as sm
for era in ("TRAIN", "TEST"):
    e = t2.filter((pl.col("era") == era)
                  & pl.col("logto").is_not_null()).to_pandas()
    X = sm.add_constant(e[["sh_host", "sh_sd", "sh_sa", "logto"]])
    res = sm.OLS(e["pin"], X).fit(cov_type="cluster",
                                  cov_kwds={"groups": e["cisin"]})
    print(f"\n {era}  (n={int(res.nobs)} stock-years) "
          "dep = PIN, SE clustered by stock")
    for v in ["sh_host", "sh_sd", "sh_sa", "logto"]:
        pv = res.pvalues[v]
        star = "***" if pv < .01 else "**" if pv < .05 else \
               "*" if pv < .10 else ""
        print("   " + v.ljust(9)
              + ("%.4f" % res.params[v]).rjust(10)
              + "  t=" + ("%.2f" % res.tvalues[v]).rjust(6)
              + "  p=" + ("%.3f" % pv) + " " + star)
print("\nread: prediction = sh_host coef POSITIVE (dispersed permanent")
print("declines are information-consistent -> higher PIN); sh_sd weaker/0.")

# ---- T3: does the reversal live in high-PIN names? (it should NOT) -----------
print("\n=== T3 · SHARK_DIST post20 by era-median PIN split ===")
# forward returns
p2 = (pl.read_parquet(DRIVE / "returns_panel_v3.parquet")
        .select("isin", "date", "ret_adj_mktadj").sort(["isin", "date"]))
p2 = p2.with_columns(
    pl.col("ret_adj_mktadj").clip(-0.5, 0.5).fill_null(0.0).alias("ar"))
p2 = p2.with_columns(pl.col("ar").cum_sum().over("isin").alias("cum"))
p2 = p2.with_columns(
    ((pl.coalesce(pl.col("cum").shift(-20).over("isin"),
                  pl.col("cum").last().over("isin"))
      - pl.col("cum")) * 1e4).alias("y20"))
runs = (st.sort(["cisin", "TR_DATE"])
          .with_columns(((pl.col("archetype")
                          != pl.col("archetype").shift(1)).fill_null(True))
                        .cum_sum().over("cisin").alias("_r")))
runs = runs.group_by("cisin", "_r").agg(
    pl.col("archetype").first(), pl.col("era").first(),
    pl.col("TR_DATE").last().alias("ed"))
runs = runs.with_columns(pl.col("ed").dt.year().alias("yr"))
ev = runs.join(p2.select("isin", "date", "y20"),
               left_on=["cisin", "ed"], right_on=["isin", "date"],
               how="inner")
ev = ev.join(pin.select("cisin", "yr", "pin"), on=["cisin", "yr"],
             how="left")
sd2 = ev.filter((pl.col("archetype") == "SHARK_DIST")
                & pl.col("pin").is_not_null())
sd2 = sd2.with_columns(
    (pl.col("pin") > pl.col("pin").median().over("era")).alias("hiPIN"))
print(sd2.group_by("era", "hiPIN").agg(
    pl.len().alias("n"), pl.col("y20").mean().round(0).alias("post20bp"))
    .sort(["era", "hiPIN"]))
hst = ev.filter((pl.col("archetype") == "HOSTAGE")
                & pl.col("pin").is_not_null())
hst = hst.with_columns(
    (pl.col("pin") > pl.col("pin").median().over("era")).alias("hiPIN"))
print("\nHOSTAGE for contrast:")
print(hst.group_by("era", "hiPIN").agg(
    pl.len().alias("n"), pl.col("y20").mean().round(0).alias("post20bp"))
    .sort(["era", "hiPIN"]))

pin.write_parquet(DRIVE / "fii_pin_stockyear.parquet")
print("\nwrote fii_pin_stockyear.parquet", pin.shape)
print("""
READ:
 T1: PIN in ~0.1-0.35 band + negative turnover corr = estimator sane.
 T2: sh_host > 0 (both eras) = PIN endorses the PERMANENT/information
     reading of dispersed selling — external benchmark for the reframing.
     sh_sd ~ 0 = concentrated selling is NOT informed (liquidity).
 T3: SHARK_DIST reversal similar in hi/lo PIN = reversal is not an
     informed-trading phenomenon (consistent with liquidity mechanism).
""")
