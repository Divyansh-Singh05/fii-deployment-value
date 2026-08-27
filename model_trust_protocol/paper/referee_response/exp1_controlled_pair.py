"""EXP1. The controlled pair, done two ways the paper does not do it.

(A) Identical flow tilt on both bases.  The paper compares M2-vs-M1 (EWMA
    base, tilt = [nf+, nf-]) against G2-vs-G1 (GJR base, tilt = [nf+, nf-,
    nf5, breadth, gross-intensity]).  Those differ in the BASE *and* in the
    feature set, so the pair is not controlled.  G2a is the true analogue:
    GJR base, tilt = [nf+, nf-].

(B) Common scored days.  v1 scores 2739 days, v2 scores 2235.  Re-run both
    DMs on the intersection so the evaluation window is identical.

DM replicates module21.dm_test exactly: Newey-West, NW_LAGS = 5, Bartlett.
Negative t = first model better (lower CRPS).
"""
import datetime as dt
import numpy as np
import polars as pl

ROOT = "/Users/divyanshsingh/Desktop/Major Project 2/outputs/validation"
NW_LAGS = 5
TRAIN_END = dt.date(2021, 4, 30)
TEST_START = dt.date(2021, 7, 1)


def dm_test(la, lb):
    d = la - lb
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 30:
        return np.nan, n
    d0 = d - d.mean()
    g = [float((d0 * d0).mean())]
    for l in range(1, NW_LAGS + 1):
        g.append(float((d0[l:] * d0[:-l]).mean()))
    var = g[0] + 2.0 * sum((1.0 - l / (NW_LAGS + 1)) * g[l] for l in range(1, NW_LAGS + 1))
    return float(d.mean() / np.sqrt(max(var, 1e-30) / n)), n


v1 = pl.read_parquet(f"{ROOT}/market_engine_scores.parquet").select(
    "date", "target_date", "crps_M1_vix", "crps_M2_vix_flow", "crps_M0_ewma_t")
v2 = pl.read_parquet(f"{ROOT}/market_engine_v2_scores.parquet").select(
    "date", "crps_G1_vix", "crps_G2a_vix_nf", "crps_G2_vix_flow", "crps_G0_gjr_t")
j = v1.join(v2, on="date", how="inner")

tgt = j["target_date"].to_numpy()
eras = {
    "FULL": np.ones(len(tgt), bool),
    "TRAIN": tgt <= np.datetime64(TRAIN_END),
    "TEST": tgt >= np.datetime64(TEST_START),
}


def col(name):
    return j[name].to_numpy().astype(float)


def run(label, a, b, mask_extra=None):
    out = []
    for e, m in eras.items():
        mm = m if mask_extra is None else (m & mask_extra)
        t, n = dm_test(col(a)[mm], col(b)[mm])
        out.append(f"{e}: t={t:+6.2f} (n={n})")
    print(f"  {label:34s} " + "  ".join(out))


print("=" * 78)
print("EXP1 · THE CONTROLLED PAIR")
print("=" * 78)

print("\n-- (A) each engine on its OWN scored days (reproduces the logs) --")
run("M2 vs M1   [EWMA base, nf+-]", "crps_M2_vix_flow", "crps_M1_vix")
run("G2  vs G1  [GJR base, 5 feats]", "crps_G2_vix_flow", "crps_G1_vix")
run("G2a vs G1  [GJR base, nf+-]", "crps_G2a_vix_nf", "crps_G1_vix")

# common days: every series finite
common = np.all(np.column_stack([
    np.isfinite(col(c)) for c in
    ("crps_M1_vix", "crps_M2_vix_flow", "crps_G1_vix", "crps_G2a_vix_nf", "crps_G2_vix_flow")
]), axis=1)
print(f"\n-- (B) COMMON scored days only: n={int(common.sum())} "
      f"(v1 alone {int(np.isfinite(col('crps_M2_vix_flow')).sum())}, "
      f"v2 alone {int(np.isfinite(col('crps_G2a_vix_nf')).sum())}) --")
run("M2 vs M1   [EWMA base, nf+-]", "crps_M2_vix_flow", "crps_M1_vix", common)
run("G2  vs G1  [GJR base, 5 feats]", "crps_G2_vix_flow", "crps_G1_vix", common)
run("G2a vs G1  [GJR base, nf+-]", "crps_G2a_vix_nf", "crps_G1_vix", common)

print("\n-- (C) the tilt against the BARE base (no VIX in the comparison) --")
run("M2 vs M0   [EWMA base]", "crps_M2_vix_flow", "crps_M0_ewma_t")
run("G2a vs G0  [GJR base]", "crps_G2a_vix_nf", "crps_G0_gjr_t")
print("   (common days)")
run("M2 vs M0   [EWMA base]", "crps_M2_vix_flow", "crps_M0_ewma_t", common)
run("G2a vs G0  [GJR base]", "crps_G2a_vix_nf", "crps_G0_gjr_t", common)

print("\n" + "=" * 78)
print("VERDICT")
print("=" * 78)
tA, _ = dm_test(col("crps_M2_vix_flow")[common], col("crps_M1_vix")[common])
tB, _ = dm_test(col("crps_G2a_vix_nf")[common], col("crps_G1_vix")[common])
print(f"  On identical days with an identical tilt:")
print(f"     EWMA base  t = {tA:+.2f}   {'PASS' if tA <= -2.0 else 'FAIL'} (bar <= -2.0)")
print(f"     GJR  base  t = {tB:+.2f}   {'PASS' if tB <= -2.0 else 'FAIL'} (bar <= -2.0)")
print(f"  Controlled-pair claim: "
      f"{'SURVIVES' if (tA <= -2.0 and tB > -2.0) else 'DOES NOT SURVIVE'}")
