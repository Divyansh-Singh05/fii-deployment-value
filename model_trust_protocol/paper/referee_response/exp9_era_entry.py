"""EXP9. How much of the modelling universe rests on within-era look-ahead?

module3a builds each era by counting an instrument's qualifying days over the
WHOLE era and then keeping ALL of that instrument's rows. So a stock reaching
its 60th qualifying day in 2015 has its 2011 rows retained.

A strictly causal rule would admit an instrument only from its 60th qualifying
day onward. The difference is exactly the first 59 rows of each
(instrument, era) group. This bounds the exposure.
"""
import polars as pl

ROOT = "/Users/divyanshsingh/Desktop/Major Project 2"
st = pl.read_parquet(f"{ROOT}/data/VALIDATION_DATA/states_v3.parquet",
                     columns=["cisin", "TR_DATE", "era"]).sort(["era", "cisin", "TR_DATE"])
MIN_SEQ = 60
st = st.with_columns(pl.int_range(pl.len()).over(["era", "cisin"]).alias("k"))
tot = st.height
groups = st.select(["era", "cisin"]).unique().height

print("=" * 70)
print("EXP9 · EXPOSURE OF THE ERA-ENTRY RULE")
print("=" * 70)
print(f"modelling universe            : {tot:,} instrument-days, "
      f"{st['cisin'].n_unique():,} instruments")
print(f"(instrument, era) groups      : {groups:,}")
pre = st.filter(pl.col("k") < MIN_SEQ - 1)
print(f"\nrows a strictly causal rule would exclude "
      f"(the first {MIN_SEQ-1} of each group):")
print(f"  {pre.height:,} of {tot:,} instrument-days = {100*pre.height/tot:.2f}%")
for e in ("TRAIN", "TEST"):
    a = st.filter(pl.col("era") == e); b = pre.filter(pl.col("era") == e)
    print(f"  {e:5s}: {b.height:>7,} of {a.height:>8,} = {100*b.height/a.height:5.2f}%")

print("\nno instrument is added or removed by the change; only the first")
print("weeks of each instrument-era are affected, and every instrument in")
print("the universe qualifies on its own era total by construction.")
