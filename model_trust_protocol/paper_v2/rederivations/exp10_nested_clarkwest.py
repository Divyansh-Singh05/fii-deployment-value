"""
EXP10 - The nested-model objection against the surviving application.

Charge (critics2, and the manuscript's own 6.1):
  Diebold-Mariano is invalid under the null for NESTED models, where the
  loss differential degenerates. M2_vix_flow = [VIX, NF+, NF-] strictly
  nests M1_vix = [VIX]. Parameters are fit on an EXPANDING window, so the
  Giacomini-White finite-window escape does not apply either.

Question: does the survivor's -2.22 hold once nesting is handled?

Three tests, same days, same tilt:
  (a) DM-NW(5)             as published            -> reproduction control
  (b) stationary block BS  no normal reference     -> distribution-free
  (c) Clark-West           nested-valid, on the    -> named correction
                           variance forecast
"""
import numpy as np, pandas as pd
from pathlib import Path

ROOT = Path("/Users/divyanshsingh/Desktop/Major Project 2")
V1 = ROOT / "outputs/validation/market_engine_scores.parquet"
V2 = ROOT / "outputs/validation/market_engine_v2_scores.parquet"
NW = 5
TRAIN_END = pd.Timestamp("2021-04-30")
RNG = np.random.default_rng(20260826)

def nw_var(d, lags=NW):
    d0 = d - d.mean(); n = len(d)
    g = (d0 * d0).mean()
    v = g
    for l in range(1, lags + 1):
        v += 2.0 * (1.0 - l / (lags + 1)) * (d0[l:] * d0[:-l]).mean()
    return max(v, 1e-30) / n

def dm(la, lb, lags=NW):
    """negative t = model A (first arg) better"""
    d = (la - lb); d = d[np.isfinite(d)]
    return d.mean() / np.sqrt(nw_var(d, lags)), len(d)

def block_boot_p(d, n_boot=20000, block=10):
    """stationary bootstrap p-value for H0: E[d]=0, one-sided d<0"""
    d = d[np.isfinite(d)]; n = len(d); obs = d.mean()
    dc = d - obs                      # impose the null
    p_geom = 1.0 / block
    means = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.empty(n, dtype=np.int64)
        i = RNG.integers(0, n); 
        for k in range(n):
            idx[k] = i
            if RNG.random() < p_geom: i = RNG.integers(0, n)
            else: i = (i + 1) % n
        means[b] = dc[idx].mean()
    return float((means <= obs).mean()), obs

def clark_west(y, f1, f2, lags=NW):
    """
    y  : realised target (ret^2)
    f1 : RESTRICTED (nested, smaller) variance forecast
    f2 : UNRESTRICTED (larger) variance forecast
    CW_t = (y-f1)^2 - (y-f2)^2 + (f1-f2)^2 ;  H1: E[CW]>0 = large better
    """
    m = np.isfinite(y) & np.isfinite(f1) & np.isfinite(f2)
    y, f1, f2 = y[m], f1[m], f2[m]
    cw = (y - f1) ** 2 - (y - f2) ** 2 + (f1 - f2) ** 2
    return cw.mean() / np.sqrt(nw_var(cw, lags)), len(cw), cw

def var_from_scale(scale, nu):
    """Student-t scale -> variance; nu>2 required"""
    v = np.where(nu > 2.0, scale ** 2 * nu / (nu - 2.0), np.nan)
    return v

# ---------------------------------------------------------------- load
a = pd.read_parquet(V1); b = pd.read_parquet(V2)
a["date"] = pd.to_datetime(a["date"]); b["date"] = pd.to_datetime(b["date"])
a["target_date"] = pd.to_datetime(a["target_date"])

print("=" * 74)
print("EXP10 - NESTED-MODEL CORRECTION FOR THE SURVIVING APPLICATION")
print("=" * 74)
print("survivor  : M2_vix_flow [VIX, NF+, NF-]  vs  M1_vix [VIX]")
print("nesting   : strict, 2 extra parameters")
print("estimation: expanding window, refit every 21 days")
print()

# scored days for the pair
ok = np.isfinite(a["crps_M2_vix_flow"]) & np.isfinite(a["crps_M1_vix"])
d1 = a.loc[ok].reset_index(drop=True)

# common-day restriction (the 2,235-day window shared with the GJR engine)
okb = np.isfinite(b["crps_G2a_vix_nf"]) & np.isfinite(b["crps_G1_vix"])
common = set(b.loc[okb, "date"]) & set(d1["date"])
d2 = d1[d1["date"].isin(common)].reset_index(drop=True)

for tag, d in (("OWN window", d1), ("COMMON window", d2)):
    print("-" * 74)
    print(f"{tag}   n = {len(d)}")
    print("-" * 74)
    la = d["crps_M2_vix_flow"].to_numpy()
    lb = d["crps_M1_vix"].to_numpy()
    dd = la - lb

    t, n = dm(la, lb)
    print(f"  (a) DM-NW({NW})             t = {t:+.3f}   n = {n}"
          f"   {'PASS' if t <= -2.0 else 'FAIL'}  (bar <= -2.0)")

    p, obs = block_boot_p(dd)
    print(f"  (b) stationary block BS   mean diff = {obs:+.3e}"
          f"   one-sided p = {p:.4f}   {'PASS' if p < 0.05 else 'FAIL'}")

    y = d["ret_next"].to_numpy() ** 2
    f1 = var_from_scale(d["scale_M1_vix"].to_numpy(), d["nu_M1_vix"].to_numpy())
    f2 = var_from_scale(d["scale_M2_vix_flow"].to_numpy(),
                        d["nu_M2_vix_flow"].to_numpy())
    cwt, ncw, cw = clark_west(y, f1, f2)
    from scipy import stats as st
    pcw = st.norm.sf(cwt)
    print(f"  (c) Clark-West            t = {cwt:+.3f}   n = {ncw}"
          f"   one-sided p = {pcw:.4f}   {'PASS' if pcw < 0.05 else 'FAIL'}")
    print(f"      (CW sign convention: POSITIVE = larger model better)")
    print()

# ---------------------------------------------------------------- era legs
print("-" * 74)
print("ERA LEGS on the common window (pre-registered leg 2)")
print("-" * 74)
tr = d2[d2["target_date"] <= TRAIN_END]
te = d2[d2["target_date"] > TRAIN_END]
for tag, d in (("TRAIN", tr), ("TEST", te)):
    adv = (d["crps_M2_vix_flow"] - d["crps_M1_vix"]).mean() * 100
    print(f"  {tag:5s} n={len(d):5d}  mean CRPS advantage x100 = {adv:+.4f}"
          f"   {'better' if adv < 0 else 'WORSE'}")
print()

print("=" * 74)
print("VERDICT")
print("=" * 74)
