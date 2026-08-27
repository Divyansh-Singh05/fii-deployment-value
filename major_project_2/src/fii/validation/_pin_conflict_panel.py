# Auto-extracted from module15_skeptic_tests.py (panel construction only).
# Builds `ev`: one row per archetype episode END, with characteristics.
# Kept as a shared include so module16 measures the reversal in EXACTLY
# the specification module15 does -- no second construction to drift.
# ---- panel characteristics (module7b construction, + delayed window) ---------
p = (pl.read_parquet(DRIVE / "returns_panel_v3.parquet")
       .select("isin", "date", "close", "volume", "ret_adj",
               "ret_adj_mktadj", "nifty50_ret").sort(["isin", "date"]))
p = p.with_columns(pl.col("nifty50_ret").fill_null(0.0))
p = p.with_columns(
    pl.col("ret_adj_mktadj").clip(-.5, .5).fill_null(0.0).alias("ar"),
    pl.col("ret_adj").clip(-.5, .5).fill_null(0.0).alias("r"),
    (pl.col("close") * pl.col("volume")).alias("to"))
p = p.with_columns(pl.col("ar").cum_sum().over("isin").alias("cum"))
W = 120
p = p.with_columns(
    (pl.col("r") * pl.col("nifty50_ret")).alias("_xy"),
    (pl.col("nifty50_ret") ** 2).alias("_y2"))
p = p.with_columns(
    pl.col("_xy").rolling_mean(window_size=W).over("isin").alias("_mxy"),
    pl.col("r").rolling_mean(window_size=W).over("isin").alias("_mx"),
    pl.col("nifty50_ret").rolling_mean(window_size=W).over("isin")
      .alias("_my"),
    pl.col("_y2").rolling_mean(window_size=W).over("isin").alias("_my2"))
p = p.with_columns(
    ((pl.col("_mxy") - pl.col("_mx") * pl.col("_my"))
     / (pl.col("_my2") - pl.col("_my") ** 2))
    .shift(1).over("isin").alias("beta120"))
p = p.with_columns(
    (pl.col("cum").shift(21).over("isin")
     - pl.col("cum").shift(127).over("isin")).alias("mom"),
    (pl.col("cum").shift(1).over("isin")
     - pl.col("cum").shift(21).over("isin")).alias("pre20"),
    pl.col("ar").rolling_std(window_size=20).over("isin")
      .shift(1).alias("vol20"),
    (pl.col("ar").abs() / (pl.col("to") + 1.0)).alias("_ilq"),
    pl.col("to").rolling_mean(window_size=20).over("isin").alias("_toma"))
last = pl.col("cum").last().over("isin")
p = p.with_columns(
    (pl.coalesce(pl.col("cum").shift(-20).over("isin"), last)
     - pl.col("cum")).alias("post20"),
    (pl.coalesce(pl.col("cum").shift(-22).over("isin"), last)
     - pl.coalesce(pl.col("cum").shift(-2).over("isin"), last))
    .alias("post20d"))                       # t+3 .. t+22 (bounce-free)
p = p.with_columns(
    (pl.col("_ilq").rolling_mean(window_size=20).over("isin").shift(1)
     * 1e9 + 1e-9).log().alias("amihud"),
    (pl.col("_toma").shift(1).over("isin") + 1.0).log().alias("logto"),
    (pl.col("volume") * pl.col("close")
     / pl.col("_toma").shift(1).over("isin")).alias("relvol"),
    pl.col("close").log().alias("logclose"))
anch = p.select("isin", "date", "pre20", "post20", "post20d", "vol20",
                "relvol", "logclose", "beta120", "mom", "amihud",
                "logto")

st = (pl.read_parquet(DRIVE / "states_v3.parquet")
        .sort(["cisin", "TR_DATE"]))
runs = st.with_columns(
    ((pl.col("archetype") != pl.col("archetype").shift(1))
     .fill_null(True)).cum_sum().over("cisin").alias("_r"))
runs = runs.group_by("cisin", "_r").agg(
    pl.col("archetype").first().alias("arch"), pl.col("era").first(),
    pl.col("TR_DATE").last().alias("ed"), pl.len().alias("eplen"))
ev = runs.join(anch, left_on=["cisin", "ed"],
               right_on=["isin", "date"], how="inner")
ev = ev.with_columns(
    pl.col("ed").dt.strftime("%Y-%m").alias("month"),
    (1e4 * pl.col("pre20")).alias("pre20bp"),
    (1e4 * pl.col("mom")).alias("mombp"),
    (1e4 * pl.col("post20")).alias("y20"),
    (1e4 * pl.col("post20d")).alias("y20d"),
    pl.col("eplen").cast(pl.Float64).log().alias("logeplen"))
for a in ("HOSTAGE", "SHARK_ACC", "SHARK_DIST", "ROBOT"):
    ev = ev.with_columns((pl.col("arch") == a).cast(pl.Float64)
                         .alias("D_" + a))
# public volume-conditioned-reversal dummy (era-frozen 20th pct of pre20)
q20 = {e: float(ev.filter(pl.col("era") == e)["pre20bp"].quantile(0.2))
       for e in ("TRAIN", "TEST")}
ev = ev.with_columns(
    pl.when(((pl.col("era") == "TRAIN")
             & (pl.col("pre20bp") < q20["TRAIN"])
             | (pl.col("era") == "TEST")
             & (pl.col("pre20bp") < q20["TEST"]))
            & (pl.col("relvol") > 1.1)).then(1.0).otherwise(0.0)
    .alias("D_VCR"))

