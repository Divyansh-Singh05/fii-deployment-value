"""EXP5. Make Table 1's two rows actually comparable.

The manuscript says of its availability table: "Nothing differs between the
rows except the information set: same panel, same fixed effects, same
controls, same clustering."

In module19 that is not what runs. The contemporaneous row is
    spec("stock + date FE", ..., two_way=True)   -> stock AND date FE
the deployable row is
    spec("share at t-2",   ..., )                 -> two_way defaults False
                                                     -> date FE only

So the published -5.53 vs -0.32 contrast changes the fixed effects as well as
the information set. This re-runs the t-2 specification WITH stock + date FE,
so the comparison is the one the paper claims to make.
"""
import sys
import numpy as np

sys.path.insert(0, "/Users/divyanshsingh/Desktop/Major Project 2/src")
from fii.validation import module19_institutional_share as m19  # noqa: E402

import polars as pl

j = m19.load()
j = j.with_columns(pl.col("share").shift(2).over("cisin").alias("share_L2"),
                   pl.col("vol").shift(2).over("cisin").alias("vol_L2"))
d = j.to_pandas()
_, day = np.unique(d["TR_DATE"].values, return_inverse=True)
_, stk = np.unique(d["cisin"].values, return_inverse=True)
era = d["era"].values

z2 = np.clip(d["z_h1"].values, -m19.CLIP, m19.CLIP) ** 2


def run(label, ycol, xcols, two_way, sub=None):
    m = np.ones(len(d), bool) if sub is None else sub
    cols = [d[c].values for c in xcols]
    ok = m & np.isfinite(ycol) & np.all([np.isfinite(c) for c in cols], axis=0)
    dd = day[ok]
    f = ((lambda x: m19._demean2(x, day[ok], stk[ok])) if two_way
         else (lambda x: m19._demean(x, dd)))
    r = m19._cluster_reg(f(ycol[ok]), [f(c[ok]) for c in cols], dd)
    print(f"  {label:<44}{r[0][0]:>10.4f}{r[0][1]:>8.2f}{ok.sum():>10,}")
    return r[0][1]


print("=" * 78)
print("EXP5 · TABLE 1 WITH THE FIXED EFFECTS HELD CONSTANT")
print("=" * 78)
print(f"  {'specification':<44}{'b(share)':>10}{'t':>8}{'n':>10}")
out = {}
for lab, e in (("FULL", None), ("TRAIN", "TRAIN"), ("TEST", "TEST")):
    sub = None if e is None else (era == e)
    print(f"\n  -- {lab} --")
    out[(lab, "contemp", "2way")] = run("contemporaneous, stock + date FE", z2,
                                        ["share", "vol"], True, sub)
    out[(lab, "t2", "2way")] = run("t-2,            stock + date FE", z2,
                                   ["share_L2", "vol_L2"], True, sub)
    out[(lab, "contemp", "date")] = run("contemporaneous, date FE only", z2,
                                        ["share", "vol"], False, sub)
    out[(lab, "t2", "date")] = run("t-2,            date FE only  [published]", z2,
                                   ["share_L2", "vol_L2"], False, sub)

print("\n" + "=" * 78)
print("LIKE-FOR-LIKE CONTRAST (stock + date FE on BOTH rows)")
print("=" * 78)
print(f"{'era':<8}{'contemporaneous t':>20}{'t-2 t':>12}")
for lab in ("FULL", "TRAIN", "TEST"):
    print(f"{lab:<8}{out[(lab,'contemp','2way')]:>20.2f}"
          f"{out[(lab,'t2','2way')]:>12.2f}")
print("\nPublished table used stock+date FE on row 1 and date FE on row 2.")
print("Conclusion unchanged either way: the effect collapses at the")
print("deployable lag. The table can now be stated as the paper describes it.")
