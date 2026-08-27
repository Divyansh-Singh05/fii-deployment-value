"""EXP6. Exact contents of the corrected Table 2: identical tilt, identical days."""
import datetime as dt
import numpy as np
import polars as pl

R = "/Users/divyanshsingh/Desktop/Major Project 2/outputs/validation"
TRAIN_END, TEST_START = dt.date(2021, 4, 30), dt.date(2021, 7, 1)


def dm(la, lb, nw=5):
    d = (la - lb)[np.isfinite(la - lb)]
    n = len(d); d0 = d - d.mean()
    g = [float((d0*d0).mean())]
    for l in range(1, nw+1):
        g.append(float((d0[l:]*d0[:-l]).mean()))
    v = g[0] + 2*sum((1-l/(nw+1))*g[l] for l in range(1, nw+1))
    return float(d.mean()/np.sqrt(max(v, 1e-30)/n)), n


j = (pl.read_parquet(f"{R}/market_engine_scores.parquet")
       .select("date", "target_date", "crps_M1_vix", "crps_M2_vix_flow",
               "hit5_M2_vix_flow", "hit1_M2_vix_flow")
       .join(pl.read_parquet(f"{R}/market_engine_v2_scores.parquet")
               .select("date", "crps_G1_vix", "crps_G2a_vix_nf",
                       "hit5_G2a_vix_nf", "hit1_G2a_vix_nf"), on="date"))
c = lambda n: j[n].to_numpy().astype(float)
common = np.all(np.column_stack([np.isfinite(c(x)) for x in
                ("crps_M1_vix", "crps_M2_vix_flow", "crps_G1_vix", "crps_G2a_vix_nf")]), axis=1)
tgt = j["target_date"].to_numpy()
eras = {"FULL": common,
        "TRAIN": common & (tgt <= np.datetime64(TRAIN_END)),
        "TEST": common & (tgt >= np.datetime64(TEST_START))}

print("=" * 74)
print("CORRECTED TABLE 2 — identical tilt [nf+, nf-], identical %d days" % common.sum())
print("=" * 74)
for lab, (a, b) in (("EWMA base", ("crps_M2_vix_flow", "crps_M1_vix")),
                    ("GJR  base", ("crps_G2a_vix_nf", "crps_G1_vix"))):
    t, n = dm(c(a)[eras["FULL"]], c(b)[eras["FULL"]])
    tr = (np.nanmean(c(a)[eras["TRAIN"]]) - np.nanmean(c(b)[eras["TRAIN"]]))*100
    te = (np.nanmean(c(a)[eras["TEST"]]) - np.nanmean(c(b)[eras["TEST"]]))*100
    print(f"\n{lab}")
    print(f"  score differential, full   : {t:+.2f}  ({'pass' if t<=-2 else 'fail'})   n={n}")
    print(f"  mean score advantage x100  : TRAIN {tr:+.4f}  TEST {te:+.4f}")

def kupiec(h, p):
    h = h[np.isfinite(h)]; n = len(h); x = h.sum(); ph = x/n
    ll = lambda q: (n-x)*np.log(max(1-q,1e-12)) + x*np.log(max(q,1e-12))
    from scipy import stats
    return float(stats.chi2.sf(-2*(ll(p)-ll(max(ph,1e-12))), 1))

for lab, h5, h1 in (("EWMA base", "hit5_M2_vix_flow", "hit1_M2_vix_flow"),
                    ("GJR  base", "hit5_G2a_vix_nf", "hit1_G2a_vix_nf")):
    print(f"  {lab} coverage p (5%,1%) on common days: "
          f"{kupiec(c(h5)[common],0.05):.3f}, {kupiec(c(h1)[common],0.01):.3f}")
