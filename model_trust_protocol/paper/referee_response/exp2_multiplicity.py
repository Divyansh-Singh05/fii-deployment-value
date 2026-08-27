"""EXP2. Price the engine-variant search that the paper says it did not price.

The manuscript's limitations concede: "Eight strategy books and four engine
variants were examined without a correction pricing that search."  For the
SURVIVING result the relevant family is the four pre-registered engine gates.
Each gate is one-sided (t <= -2.0), so p = Phi(t).

Reports Bonferroni and Holm across the family, on the survivor's own scored
days and on the days common to the two market-level engines.
"""
import datetime as dt
import numpy as np
import polars as pl
from scipy import stats

ROOT = "/Users/divyanshsingh/Desktop/Major Project 2/outputs/validation"
TRAIN_END = dt.date(2021, 4, 30)
TEST_START = dt.date(2021, 7, 1)


def dm_test(la, lb, nw):
    d = (la - lb)[np.isfinite(la - lb)]
    n = len(d)
    d0 = d - d.mean()
    g = [float((d0 * d0).mean())]
    for l in range(1, nw + 1):
        g.append(float((d0[l:] * d0[:-l]).mean()))
    var = g[0] + 2.0 * sum((1.0 - l / (nw + 1)) * g[l] for l in range(1, nw + 1))
    return float(d.mean() / np.sqrt(max(var, 1e-30) / n)), n


def load(f, *cols):
    return pl.read_parquet(f"{ROOT}/{f}.parquet").select("date", *cols)


gates = []
# v1 EWMA base  -- the survivor
v1 = load("market_engine_scores", "crps_M1_vix", "crps_M2_vix_flow")
t, n = dm_test(v1["crps_M2_vix_flow"].to_numpy().astype(float),
               v1["crps_M1_vix"].to_numpy().astype(float), 5)
gates.append(("v1 EWMA base   M2 vs M1", t, n))
# v2 GJR base
v2 = load("market_engine_v2_scores", "crps_G1_vix", "crps_G2_vix_flow")
t, n = dm_test(v2["crps_G2_vix_flow"].to_numpy().astype(float),
               v2["crps_G1_vix"].to_numpy().astype(float), 5)
gates.append(("v2 GJR base    G2 vs G1", t, n))
# v3 global forces
v3 = load("market_engine_v3_scores", "crps_H0_gjr_t", "crps_H1_global3")
t, n = dm_test(v3["crps_H1_global3"].to_numpy().astype(float),
               v3["crps_H0_gjr_t"].to_numpy().astype(float), 5)
gates.append(("v3 globals     H1 vs H0", t, n))
# v4 weekly (NW10 per its prereg)
v4 = load("market_engine_v4_weekly_scores", "crps_B0_gjr5", "crps_B1_spx5")
t, n = dm_test(v4["crps_B1_spx5"].to_numpy().astype(float),
               v4["crps_B0_gjr5"].to_numpy().astype(float), 10)
gates.append(("v4 weekly      B1 vs B0", t, n))

print("=" * 78)
print("EXP2 · MULTIPLICITY ACROSS THE FOUR PRE-REGISTERED ENGINE GATES")
print("=" * 78)
print(f"\n{'gate':26s}{'t':>8s}{'n':>7s}{'one-sided p':>14s}{'verdict':>10s}")
ps = []
for name, t, n in gates:
    p = float(stats.norm.cdf(t))
    ps.append(p)
    print(f"{name:26s}{t:+8.2f}{n:7d}{p:14.4f}{'PASS' if t <= -2.0 else 'FAIL':>10s}")

m = len(gates)
print(f"\nFamily size m = {m}.  Survivor's raw one-sided p = {ps[0]:.4f}")
print(f"  Bonferroni adjusted p = {min(ps[0]*m,1):.4f}   "
      f"{'survives' if ps[0]*m < 0.05 else 'FAILS'} at 5%")
order = np.argsort(ps)
holm = np.empty(m)
run = 0.0
for rank, idx in enumerate(order):
    run = max(run, ps[idx] * (m - rank))
    holm[idx] = min(run, 1.0)
print(f"  Holm adjusted p       = {holm[0]:.4f}   "
      f"{'survives' if holm[0] < 0.05 else 'FAILS'} at 5%")

# same, on the days common to v1 and v2
j = (pl.read_parquet(f"{ROOT}/market_engine_scores.parquet")
       .select("date", "crps_M1_vix", "crps_M2_vix_flow")
       .join(pl.read_parquet(f"{ROOT}/market_engine_v2_scores.parquet")
               .select("date", "crps_G1_vix", "crps_G2a_vix_nf"), on="date"))
mask = np.all(np.column_stack([np.isfinite(j[c].to_numpy().astype(float))
                               for c in j.columns if c != "date"]), axis=1)
tc, nc = dm_test(j["crps_M2_vix_flow"].to_numpy().astype(float)[mask],
                 j["crps_M1_vix"].to_numpy().astype(float)[mask], 5)
pc = float(stats.norm.cdf(tc))
print(f"\nOn the {nc} days common to the two market-level engines:")
print(f"  survivor t = {tc:+.2f}, raw p = {pc:.4f}, "
      f"Bonferroni x{m} = {min(pc*m,1):.4f}   "
      f"{'survives' if pc*m < 0.05 else 'FAILS'} at 5%")
