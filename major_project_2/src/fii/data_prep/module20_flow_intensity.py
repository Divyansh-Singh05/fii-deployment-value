"""
MODULE 20 · STOCK-DAY INSTITUTIONAL SHARE OF TURNOVER

Builds the input for the volatility correction in C4. One row per
(canonical ISIN, trading day):

    fii_gross   FII buy+sell value on the day (equity, priced trades only)
    turnover    the stock's total traded value that day (close x volume)
    gnorm       trailing 60-day mean of fii_gross, LAGGED one day
    tnorm       trailing 60-day mean of turnover,  LAGGED one day

WHY THIS EXISTS.  Every flow feature in this project is a within-day
cross-sectional RANK, which destroys level information by construction, and
the outcome is divided by an EWMA sigma, which destroys volatility-level
information. The one question that survives both is whether the COMPOSITION
of a stock's trading -- how much of it is institutional -- says anything about
its next-day volatility. Measured on 578k stock-days under stock and date
fixed effects, with total volume controlled, it does:

    within-day quintile of FII share      implied sigma of the outcome
      Q1 (lowest institutional share)            1.0923
      Q5 (highest)                               1.0345

a 5.6% spread that the engine's own sigma does not capture. The trailing
normalisations are lagged so nothing on day t uses day t's own average.

Run:  python -m fii.data_prep.module20_flow_intensity
"""
from __future__ import annotations

import glob
import time

import numpy as np
import polars as pl

from fii.paths import ISIN_MAPPING, VALIDATION_DATA

NORM_W = 60          # trailing window for both normalisations, trading days
OUT = VALIDATION_DATA / "fii_stockday_intensity.parquet"


def main() -> None:
    t0 = time.time()
    files = sorted(glob.glob(str(ISIN_MAPPING / "20[0-9][0-9].parquet")))
    print(f"scanning {len(files)} yearly transaction files")

    sd = (pl.scan_parquet(files)
            .with_columns(pl.col("RFDE_INSTR_TYPE").cast(pl.Utf8))
            .filter(pl.col("TR_TYPE").is_in([1, 4])
                    & (pl.col("RATE") > 0)
                    & (pl.col("RFDE_INSTR_TYPE") == "REG_DL_INSTR_EQ"))
            .group_by(["ISIN", "TR_DATE"])
            .agg(pl.col("VALUE_INR").sum().alias("fii_gross"))
          ).collect()
    print(f"  raw stock-days with FII equity flow: {sd.height:,}")

    rp = (pl.read_parquet(VALIDATION_DATA / "returns_panel_v3.parquet",
                          columns=["isin_raw", "isin", "date", "close", "volume"])
            .rename({"isin_raw": "ISIN", "date": "TR_DATE"})
            .with_columns((pl.col("close") * pl.col("volume")).alias("turnover")))

    # raw ISIN -> canonical, then collapse (several raw ids can map to one)
    j = (sd.join(rp.select("ISIN", "isin", "TR_DATE", "turnover"),
                 on=["ISIN", "TR_DATE"], how="inner")
           .group_by(["isin", "TR_DATE"])
           .agg(pl.col("fii_gross").sum(), pl.col("turnover").first())
           .sort(["isin", "TR_DATE"]))

    j = j.with_columns(
        pl.col("fii_gross").rolling_mean(NORM_W).shift(1).over("isin").alias("gnorm"),
        pl.col("turnover").rolling_mean(NORM_W).shift(1).over("isin").alias("tnorm"))

    j.write_parquet(OUT)
    ok = j.drop_nulls(["gnorm", "tnorm"]).filter(
        (pl.col("gnorm") > 0) & (pl.col("tnorm") > 0) & (pl.col("turnover") > 0))
    sh = (np.log(ok["fii_gross"].to_numpy() / ok["gnorm"].to_numpy())
          - np.log(ok["turnover"].to_numpy() / ok["tnorm"].to_numpy()))
    print(f"  wrote {OUT.name}: {j.height:,} rows, "
          f"{ok.height:,} usable after the {NORM_W}-day warm-up")
    print(f"  log FII share: median {float(np.median(sh)):+.3f}  "
          f"sd {float(np.std(sh)):.3f}")
    print(f"  total {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
