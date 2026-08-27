"""ONE-TIME ingestion of the dataset into data/.

THIS IS THE ONLY MODULE IN THE PACKAGE THAT READS A PATH OUTSIDE IT.

Everything else - every model, every level runner, every reproduction script -
resolves its inputs through `config/paths.yaml`, which contains only
package-relative paths. Once this has run, the package is self-contained and
the source trees can be removed entirely.

Disclosure classes applied here:

  PUBLIC      corporate actions, instrument map. Derived from exchange filings
              that are already public.
  DERIVED     panels and scored outputs. Normalised ratios and scores; no
              participant identifier and no reconstructable trade.
  RESTRICTED  the raw transaction record. NOT copied. Its schema and the
              digest of the upstream artifact are recorded instead.

Run:  python -m src.data.ingest
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import polars as pl
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.logging import banner, get_logger, verdict
from src.common.utilities import PKG_ROOT, sha256

DATA = PKG_ROOT / "data"
MANIFEST: list[dict] = []


def _cfg() -> dict:
    with (PKG_ROOT / "config" / "ingestion.yaml").open() as fh:
        return yaml.safe_load(fh)


def _src(tree: str, key: str) -> Path:
    c = _cfg()["source_trees"][tree]
    root = Path(os.environ.get({"tree_a": "FIHMM_SOURCE_A",
                                "tree_b": "FIHMM_SOURCE_B"}[tree], c["root"]))
    return root / c["artifacts"][key]


def _emit(df, path: Path, klass: str, source: str, log) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        (df.write_parquet(path) if isinstance(df, pl.DataFrame)
         else df.to_parquet(path, index=False))
    else:
        df.to_csv(path, index=False)
    n = df.height if isinstance(df, pl.DataFrame) else len(df)
    MANIFEST.append({"file": str(path.relative_to(PKG_ROOT)), "rows": n,
                     "class": klass, "ingested_from": source,
                     "sha256": sha256(path)})
    verdict(log, path.name, "PASS", f"{n:,} rows  ({klass})")


# ------------------------------------------------------------------ public
def ingest_corporate_actions(log) -> None:
    fac = pd.read_parquet(_src("tree_a", "ca_factors"))

    ca = (fac[["symbol", "ex_date", "kind", "factor", "n_events", "purpose"]]
          .sort_values(["ex_date", "symbol"]).reset_index(drop=True))
    _emit(ca, DATA / "corporate_actions/corporate_actions.csv",
          "PUBLIC", "ca_adjustment_factors.parquet", log)

    rows = []
    for exch, key in (("NSE", "nse_actions"), ("BSE", "bse_actions")):
        p = _src("tree_a", key)
        n = sum(1 for _ in p.open(encoding="utf-8", errors="replace")) - 1
        rows.append({"exchange": exch, "filings_scanned": n})
    match = pd.DataFrame(rows)
    match["actions_parsed"] = len(ca)
    match["parse_rate_of_filings"] = (match["actions_parsed"]
                                      / match["filings_scanned"]).round(5)
    match["note"] = ("most filings are dividends, which need no price "
                     "adjustment; only splits and bonuses yield a factor")
    _emit(match, DATA / "corporate_actions/ca_matches.csv",
          "PUBLIC", "nse/bse_corporate_actions.csv", log)

    val = fac[["symbol", "ex_date", "kind", "factor", "obs_ratio", "confirmed"]].copy()
    val["ratio_error"] = (val["obs_ratio"] / val["factor"] - 1.0).abs().round(5)
    val["status"] = val["confirmed"].map({True: "CONFIRMED", False: "MISMATCH"})
    val["status"] = val["status"].fillna("UNVERIFIABLE")
    val = val.sort_values(["status", "ex_date"]).reset_index(drop=True)
    _emit(val, DATA / "corporate_actions/ca_validation.csv",
          "PUBLIC", "ca_adjustment_factors.parquet", log)
    log.info("      " + "  ".join(f"{k}={v}" for k, v
                                  in val["status"].value_counts().items()))


def ingest_instrument_mapping(log) -> None:
    pit = pd.read_parquet(_src("tree_a", "isin_pit_map"))
    _emit(pit.sort_values(["old_isin", "effective_from"]).reset_index(drop=True),
          DATA / "instrument_mapping/instrument_map.csv",
          "PUBLIC", "isin_pit_map.parquet", log)

    chains = pd.read_csv(_src("tree_a", "isin_chains"))
    depth = (chains.groupby("chain_depth").size().reset_index(name="n_chains")
             .assign(metric="chain_depth").rename(columns={"chain_depth": "value"}))
    conf = (chains.groupby(["confidence", "chain_status"]).size()
            .reset_index(name="n_chains"))
    conf = conf.assign(metric="confidence_x_status",
                       value=conf["confidence"] + "/" + conf["chain_status"])
    val = pd.concat([depth[["metric", "value", "n_chains"]],
                     conf[["metric", "value", "n_chains"]]], ignore_index=True)
    ov = (pit.groupby("old_isin").apply(
        lambda g: (g.sort_values("effective_from")["effective_to"].shift(1)
                   > g.sort_values("effective_from")["effective_from"]).sum(),
        include_groups=False))
    val.loc[len(val)] = ["overlapping_intervals", "count", int(ov.sum())]
    _emit(val, DATA / "instrument_mapping/mapping_validation.csv",
          "PUBLIC", "isin_mapping_final.csv + isin_pit_map.parquet", log)


# ----------------------------------------------------------------- derived
def ingest_panels(log) -> None:
    root = _src("tree_a", "panel_module")
    sys.path.insert(0, str(root))
    from fii.validation import module19_institutional_share as m19  # noqa: PLC0415
    from fii.validation import module21_market_flow_engine as m21   # noqa: PLC0415

    # equation (1)'s panel, with the within-instrument lagged columns
    panel = m19.load().with_columns(
        pl.col("share").shift(2).over("cisin").alias("share_L2"),
        pl.col("vol").shift(2).over("cisin").alias("vol_L2"),
    ).select(["cisin", "TR_DATE", "era", "z_h1", "share", "vol",
              "share_L2", "vol_L2"])
    _emit(panel, DATA / "derived/panel_analysis.parquet",
          "DERIVED", "module19.load()", log)
    # the winsorisation constant travels with the panel, not with the code
    (DATA / "derived/panel_analysis.meta.json").write_text(
        json.dumps({"clip": m19.CLIP,
                    "dependent": "clip(z_h1, -CLIP, CLIP) ** 2",
                    "fixed_effects": ["cisin", "TR_DATE"],
                    "controls": ["vol = log(turnover / trailing-60d mean)"],
                    "cluster": "TR_DATE"}, indent=2))

    _emit(m21.build_frame(), DATA / "derived/daily_features.parquet",
          "DERIVED", "module21.build_frame()", log)


def ingest_scores(log) -> None:
    v1 = pd.read_parquet(_src("tree_a", "market_engine_v1"))
    v2 = pd.read_parquet(_src("tree_a", "market_engine_v2"))
    keep1 = ["date", "target_date", "ret_next", "nf_t2",
             "crps_M0_ewma_t", "crps_M1_vix", "crps_M2_vix_flow",
             "scale_M1_vix", "scale_M2_vix_flow", "nu_M1_vix", "nu_M2_vix_flow",
             "hit5_M2_vix_flow", "hit1_M2_vix_flow"]
    keep2 = ["date", "crps_G1_vix", "crps_G2a_vix_nf",
             "hit5_G2a_vix_nf", "hit1_G2a_vix_nf"]
    panel = v1[keep1].merge(v2[keep2], on="date", how="left")
    panel = panel[panel["crps_M2_vix_flow"].notna()].reset_index(drop=True)
    _emit(panel, DATA / "derived/analysis_dataset.parquet",
          "DERIVED", "market_engine v1 + v2 scores", log)

    w = pd.read_parquet(_src("tree_a", "market_engine_v4"))
    w = w[["date", "ret5_next", "crps_B0_gjr5", "crps_B1_spx5"]]
    w = w[w["crps_B1_spx5"].notna()].reset_index(drop=True)
    _emit(w, DATA / "derived/weekly_control_scores.parquet",
          "DERIVED", "market_engine_v4_weekly_scores.parquet", log)

    h = pd.read_parquet(_src("tree_a", "hazard_preds"))
    _emit(h, DATA / "derived/hazard_predictions.parquet",
          "DERIVED", "phase2_hazard_preds.parquet", log)

    m = pd.read_csv(_src("tree_a", "backtest_metrics"))
    _emit(m, DATA / "derived/backtest_metrics.csv",
          "DERIVED", "outputs/metrics/backtest_metrics.csv", log)


def record_restricted(log) -> None:
    src = _src("tree_a", "ca_raw")
    schema = pl.read_parquet_schema(src)
    note = {
        "status": "RESTRICTED - not distributed",
        "reason": ("proprietary depository settlement records; not a commercial "
                   "data product and not redistributable under any licence here"),
        "upstream_artifact": str(src),
        "upstream_sha256": sha256(src),
        "rows_upstream": pl.scan_parquet(src).select(pl.len()).collect().item(),
        "schema": {k: str(v) for k, v in schema.items()},
        "note": ("the derived panels in data/derived carry everything the "
                 "reproduction path needs; the raw record is required only to "
                 "REBUILD them from scratch"),
    }
    (DATA / "derived/adjusted_transactions.RESTRICTED.json").write_text(
        json.dumps(note, indent=2))
    verdict(log, "adjusted_transactions.parquet", "SKIP",
            "RESTRICTED - schema and upstream digest recorded instead")


def main() -> int:
    log = get_logger("ingest")
    banner(log, "ONE-TIME INGESTION - the only stage that reads outside the package")
    for fn in (ingest_corporate_actions, ingest_instrument_mapping,
               ingest_panels, ingest_scores, record_restricted):
        log.info("")
        fn(log)
    pd.DataFrame(MANIFEST).to_csv(DATA / "MANIFEST.csv", index=False)
    log.info("")
    verdict(log, "MANIFEST.csv", "PASS", f"{len(MANIFEST)} files, digests recorded")
    log.info("  the package is now self-contained; source trees may be removed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
