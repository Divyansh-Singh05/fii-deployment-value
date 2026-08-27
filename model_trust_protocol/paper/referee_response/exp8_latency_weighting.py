"""EXP8. Is the latency curve an artefact of the weighting?

Round-2 review 2.2: "A single aggregate latency curve is not enough if a small
number of large transactions systematically arrive later." Reports the curve
value-weighted AND trade-count-weighted, and the mean trade size by lag, so the
direction of any size/latency relation is visible rather than assumed.
"""
import glob
import numpy as np
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
d = (b.join(tr, on="TR_DATE", how="inner").join(rp, on="RFDE_RPT_DT", how="left")
      .collect().with_columns(lag=(pl.col("t_rp") - pl.col("t_tr"))))

V = float(d["VALUE_INR"].sum()); N = d.height
print("=" * 74)
print("EXP8 · LATENCY BY WEIGHTING SCHEME")
print("=" * 74)
print(f"{'by close of':>12}{'% of VALUE':>14}{'% of TRADES':>14}")
for k in (0, 1, 2, 3):
    s = d.filter(pl.col("lag") <= k)
    print(f"{'t+'+str(k):>12}{100*float(s['VALUE_INR'].sum())/V:14.2f}"
          f"{100*s.height/N:14.2f}")

print(f"\n{'lag':>6}{'mean trade size (INR)':>24}{'share of value':>17}{'share of trades':>17}")
agg = (d.group_by("lag").agg(v=pl.col("VALUE_INR").sum(), n=pl.len(),
                             mean=pl.col("VALUE_INR").mean()).sort("lag"))
for r in agg.head(6).iter_rows(named=True):
    lg = "late/off-cal" if r["lag"] is None else str(r["lag"])
    print(f"{lg:>6}{r['mean']:24,.0f}{100*r['v']/V:17.2f}{100*r['n']/N:17.2f}")
late = d.filter(pl.col("lag").is_null())
if late.height:
    print(f"{'off-cal':>6}{float(late['VALUE_INR'].mean()):24,.0f}"
          f"{100*float(late['VALUE_INR'].sum())/V:17.2f}{100*late.height/N:17.2f}")

on_time = d.filter(pl.col("lag") <= 1)["VALUE_INR"]
slow = d.filter((pl.col("lag") > 1) | pl.col("lag").is_null())["VALUE_INR"]
print("\n" + "=" * 74)
print(f"mean trade size reported by t+1      : {float(on_time.mean()):>16,.0f}")
print(f"mean trade size reported after t+1   : {float(slow.mean()):>16,.0f}")
print(f"ratio (slow / fast)                  : {float(slow.mean())/float(on_time.mean()):>16.2f}")
print("\n-> " + ("LARGE trades arrive later; the value-weighted curve is the"
                 " binding one." if float(slow.mean()) > float(on_time.mean())
                 else "Late trades are SMALLER than average, so value-weighting is"
                      "\n   the conservative choice and count-weighting would look worse."))
