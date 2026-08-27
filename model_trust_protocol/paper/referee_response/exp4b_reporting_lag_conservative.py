"""EXP4b. Reporting lag with the honest denominator.

EXP4 computed cumulative shares inside the subset where BOTH dates fall on the
exchange calendar. That subset drops 0.84% of value whose report date is off
the calendar with a median gap of 30 calendar days -- i.e. genuinely late or
amended filings. Excluding them flatters the availability figure.

Denominator here = all value whose TRADE date is a trading day. Anything whose
report date is off-calendar is counted as NOT reported by any finite k.
"""
import glob
import polars as pl

ROOT = "/Users/divyanshsingh/Desktop/Major Project 2"
files = sorted(glob.glob(f"{ROOT}/data/ISIN_MAPPING/20[0-9][0-9].parquet"))
cal = (pl.scan_parquet(f"{ROOT}/data/VALIDATION_DATA/returns_panel_v3.parquet")
         .select("date").unique().collect().sort("date")).with_row_index("tidx")

b = (pl.scan_parquet(files).with_columns(pl.col("RFDE_INSTR_TYPE").cast(pl.Utf8))
     .filter(pl.col("TR_TYPE").is_in([1, 4]) & (pl.col("RATE") > 0)
             & (pl.col("RFDE_INSTR_TYPE") == "REG_DL_INSTR_EQ"))
     .select("TR_DATE", "RFDE_RPT_DT", "VALUE_INR"))

tr = cal.rename({"date": "TR_DATE", "tidx": "t_tr"}).lazy()
rp = cal.rename({"date": "RFDE_RPT_DT", "tidx": "t_rp"}).lazy()

# denominator: trade date on the calendar
d = b.join(tr, on="TR_DATE", how="inner").join(rp, on="RFDE_RPT_DT", how="left").collect()
den = float(d["VALUE_INR"].sum())
d = d.with_columns(lag=(pl.col("t_rp") - pl.col("t_tr")))
off = float(d.filter(pl.col("lag").is_null())["VALUE_INR"].sum())

print("=" * 72)
print("REPORTING LAG, CONSERVATIVE DENOMINATOR")
print("=" * 72)
print(f"value with trade date on the exchange calendar : {den:,.0f}")
print(f"  of which report date is OFF-calendar (late)  : {100*off/den:.3f}%"
      "  <- counted as never reported by t+k")
print(f"\n{'k':>4s}{'cumulative % of value reported by close of t+k':>50s}")
for k in (0, 1, 2, 3, 5):
    c = float(d.filter(pl.col("lag") <= k)["VALUE_INR"].sum()) / den * 100
    print(f"{'t+'+str(k):>4s}{c:>50.2f}")

print("\nera split (cumulative % by t+k):")
print(f"{'era':<24s}{'t+0':>8s}{'t+1':>8s}{'t+2':>8s}{'t+3':>8s}")
for lab, sub in (("TRAIN (<= 2021-04-30)", d.filter(pl.col("TR_DATE") <= pl.date(2021, 4, 30))),
                 ("TEST  (>= 2021-07-01)", d.filter(pl.col("TR_DATE") >= pl.date(2021, 7, 1)))):
    dd = float(sub["VALUE_INR"].sum())
    row = f"{lab:<24s}"
    for k in (0, 1, 2, 3):
        row += f"{100*float(sub.filter(pl.col('lag') <= k)['VALUE_INR'].sum())/dd:8.2f}"
    print(row)

c1 = float(d.filter(pl.col("lag") <= 1)["VALUE_INR"].sum()) / den * 100
c2 = float(d.filter(pl.col("lag") <= 2)["VALUE_INR"].sum()) / den * 100
c0 = float(d.filter(pl.col("lag") <= 0)["VALUE_INR"].sum()) / den * 100
print("\n" + "=" * 72)
print(f"  contemporaneous (t)  specification assumes {c0:5.2f}% availability")
print(f"  t-1                  specification assumes {c1:5.2f}%")
print(f"  t-2  (the one used)  specification assumes {c2:5.2f}%")
print("\n  -> the contemporaneous spec in the availability application uses")
print(f"     information of which {c0:.2f}% is actually in hand. The t-2 choice")
print("     is the conservative one and is empirically grounded.")
