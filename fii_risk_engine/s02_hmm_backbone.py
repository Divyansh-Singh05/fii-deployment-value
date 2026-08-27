# [migrated from Colab: paths now come from paths; see
#  scripts/migrate_colab_modules.py — research logic unchanged]
from paths import VALIDATION_DATA, ISIN_MAPPING  # noqa: E402
# ============================================================================
# MODULE 3A · THE MODEL — TEMPORAL SPLIT + UNBIASED REFIT + FROZEN OOS DECODE
#
# One universal model file: train/test split around the masked May-Jun 2021
# gap, HMM refit on TRAIN ONLY with an unbiased fit-cap (random contiguous
# block per stock — fixes v1's recency-biased "most recent 400 rows"),
# frozen decode of both eras, and the out-of-sample replication checks
# (state signatures, census, empirical transitions).
#
# States only (SELL/NEUTRAL/BUY). Archetype overlay happens in 3B with the
# calibrated threshold; statistics battery in 3C.
#
# Input : stockday_features_v2.parquet
# Output: stockday_states_split.parquet  (era, state, features — input to 3B)
# ============================================================================
import numpy as np
import polars as pl
import datetime as dt
from pathlib import Path
from scipy.special import ndtri
from hmmlearn.hmm import GaussianHMM

# ------------------------------------------------------------- CONFIG -------
parquet_path = ISIN_MAPPING
PANEL_IN  = str(parquet_path / "stockday_features_v2.parquet")
OUT_SPLIT = str(parquet_path / "stockday_states_split.parquet")

TRAIN_END  = dt.date(2021, 4, 30)    # train era ends at the masked gap
TEST_START = dt.date(2021, 7, 1)     # test era starts after it
FEATS      = ["F_persist", "F_block", "F_entity_s", "F_entity_buy_s"]
# carried through, never fitted on — the renamed pre-audit measure (item 1)
AUX        = ["F_bookfocus_s", "F_bookfocus_buy_s"]
MIN_SEQ    = 60
FIT_CAP    = 400
SMOOTH     = 5
N_INITS    = 5
SEED       = 42
rng = np.random.default_rng(SEED)

# ------------------------------------------------------------- LOAD + SMOOTHED ENTITY FEATURES
# (v2 parquet stores only the raw daily HHIs; recompute the 5d-smoothed,
#  re-ranked, probit versions here and carry them forward in the output)
#
# AUDIT ITEM 1. The source columns are now `entity_conc_raw` /
# `entity_conc_buy_raw` — the participant Herfindahl taken WITHIN the
# stock-side-day, which is the measure docs/02_feature_engineering.md:45
# always specified. The pre-audit columns (`entity_hhi_raw`) held a different
# quantity: the value-weighted mean of each participant's own across-stock
# book concentration. The two rank-correlate -0.25 on the full panel.
#
# The downstream NAMES are unchanged (`F_entity_s`, `F_entity_buy_s`) because
# they now mean what they have always claimed to mean, so every consumer from
# module3b through C1-C8 keeps working with correct semantics. The renamed
# pre-audit measure is smoothed and carried alongside as `F_bookfocus_s` for
# robustness work; it does NOT enter FEATS and does not touch the model.
panel = pl.read_parquet(PANEL_IN).sort(["cisin", "TR_DATE"])
panel = panel.with_columns(
    pl.col("entity_conc_raw").rolling_mean(SMOOTH, min_samples=3).over("cisin").alias("hhi_s_raw"),
    pl.col("entity_conc_buy_raw").rolling_mean(SMOOTH, min_samples=3).over("cisin").alias("hhi_bs_raw"),
    pl.col("bookfocus_raw").rolling_mean(SMOOTH, min_samples=3).over("cisin").alias("bf_s_raw"),
    pl.col("bookfocus_buy_raw").rolling_mean(SMOOTH, min_samples=3).over("cisin").alias("bf_bs_raw"),
)
for raw, feat in {"hhi_s_raw": "F_entity_s", "hhi_bs_raw": "F_entity_buy_s",
                  "bf_s_raw": "F_bookfocus_s", "bf_bs_raw": "F_bookfocus_buy_s"}.items():
    masked = pl.when(pl.col("eligible")).then(pl.col(raw)).otherwise(None)
    panel = panel.with_columns(masked.alias("_m"))
    n_valid = pl.col("_m").is_not_null().sum().over("TR_DATE")
    rnk     = pl.col("_m").rank(method="average").over("TR_DATE")
    panel = panel.with_columns((rnk / (n_valid + 1)).alias("_p"))
    p = panel["_p"].to_numpy()
    pr = np.full(p.shape, np.nan); mm = ~np.isnan(p); pr[mm] = ndtri(p[mm])
    panel = panel.with_columns(pl.Series(feat, pr).fill_nan(None)).drop(["_m", "_p"])

# ------------------------------------------------------------- SPLIT + SEQUENCES
cc = (panel.filter(pl.all_horizontal([pl.col(f).is_not_null() for f in FEATS]))
           .select(["cisin", "TR_DATE"] + FEATS + AUX).sort(["cisin", "TR_DATE"]))

def build_era(df, lo=None, hi=None):
    e = df
    if lo: e = e.filter(pl.col("TR_DATE") >= lo)
    if hi: e = e.filter(pl.col("TR_DATE") <= hi)
    keep = e.group_by("cisin").agg(pl.len().alias("n")).filter(pl.col("n") >= MIN_SEQ)
    e = e.filter(pl.col("cisin").is_in(keep["cisin"].implode())).sort(["cisin", "TR_DATE"])
    X = e.select(FEATS).to_numpy()
    L = e.group_by("cisin", maintain_order=True).agg(pl.len())["len"].to_list()
    return e, X, L

train_df, X_tr, L_tr = build_era(cc, hi=TRAIN_END)
test_df,  X_te, L_te = build_era(cc, lo=TEST_START)
print(f"TRAIN: {train_df.height:,} stock-days | {len(L_tr):,} stocks | ≤ {TRAIN_END}")
print(f"TEST : {test_df.height:,} stock-days | {len(L_te):,} stocks | ≥ {TEST_START}")

# unbiased fit-cap: RANDOM contiguous block of ≤FIT_CAP days per stock
blocks, lengths, pos = [], [], 0
for n in L_tr:
    seg = X_tr[pos:pos + n]
    if n > FIT_CAP:
        s = rng.integers(0, n - FIT_CAP + 1)
        seg = seg[s:s + FIT_CAP]
    blocks.append(seg); lengths.append(len(seg)); pos += n
X_fit = np.vstack(blocks)
print(f"Fit sample: {X_fit.shape[0]:,} rows (random contiguous blocks ≤{FIT_CAP}/stock)")

# ------------------------------------------------------------- FIT (TRAIN ONLY) + FREEZE
best, best_ll = None, -np.inf
for k in range(N_INITS):
    m = GaussianHMM(n_components=3, covariance_type="diag",
                    n_iter=200, tol=1e-4, random_state=SEED + k)
    m.fit(X_fit, lengths)
    ll = m.score(X_fit, lengths)
    print(f"  init {k}: loglik = {ll:,.0f}{'  <- best' if ll > best_ll else ''}")
    if ll > best_ll: best, best_ll = m, ll
model = best

means = model.means_
order = np.argsort(means[:, 0])                      # by F_persist
sname = {int(order[0]): "SELL_REGIME", int(order[1]): "NEUTRAL", int(order[2]): "BUY_REGIME"}
print("\n=== TRAIN state signatures (fitted) ===")
print(f"{'state':<12}" + "".join(f"{f:>16}" for f in FEATS) + f"{'dwell':>8}")
for s in order:
    dw = 1 / (1 - model.transmat_[s, s])
    print(f"{sname[int(s)]:<12}" + "".join(f"{means[s, i]:>16.3f}" for i in range(len(FEATS))) + f"{dw:>8.1f}")

# frozen decode of BOTH eras
train_df = train_df.with_columns(pl.Series("state", [sname[s] for s in model.predict(X_tr, L_tr)]),
                                 pl.lit("TRAIN").alias("era"))
test_df  = test_df.with_columns(pl.Series("state", [sname[s] for s in model.predict(X_te, L_te)]),
                                pl.lit("TEST").alias("era"))
both = pl.concat([train_df, test_df])

# ------------------------------------------------------------- OOS REPLICATION CHECKS
print("\n=== STATE SIGNATURES: TRAIN vs TEST (frozen model — should replicate) ===")
print(both.group_by("era", "state").agg([pl.col(f).mean().round(3) for f in FEATS])
          .sort(["state", "era"]))

print("\n=== STATE CENSUS: TRAIN vs TEST ===")
print(both.group_by("era", "state").agg(pl.len().alias("n"))
          .with_columns((pl.col("n") / pl.col("n").sum().over("era")).round(4).alias("share"))
          .sort(["state", "era"]))

print("\n=== EMPIRICAL TRANSITIONS (from decoded states), per era ===")
for era in ("TRAIN", "TEST"):
    e = both.filter(pl.col("era") == era).sort(["cisin", "TR_DATE"])
    t = (e.with_columns(pl.col("state").shift(-1).over("cisin").alias("nxt"))
          .drop_nulls("nxt").group_by("state", "nxt").agg(pl.len().alias("n"))
          .with_columns((pl.col("n") / pl.col("n").sum().over("state")).round(3).alias("p"))
          .pivot(values="p", index="state", on="nxt").sort("state"))
    print(f"\n{era}:"); print(t)

both.write_parquet(OUT_SPLIT)
print(f"\nSaved → {OUT_SPLIT}   {both.shape}")

# [additive artifact export — research logic untouched: persist the frozen
#  backbone parameters so outputs/trained_models holds the actual model]
import json
from paths import TRAINED_MODELS
TRAINED_MODELS.mkdir(parents=True, exist_ok=True)
mp = {"features": FEATS,
      "state_labels": {str(k): v for k, v in sname.items()},
      "means": model.means_.tolist(),
      "covars_diag": model.covars_.tolist(),
      "transmat": model.transmat_.tolist(),
      "startprob": model.startprob_.tolist(),
      "loglik": float(best_ll), "seed": SEED,
      "train_cutoff": "2021-04-30"}
(TRAINED_MODELS / "hmm_backbone_params.json").write_text(
    json.dumps(mp, indent=2))
print(f"Saved → {TRAINED_MODELS / 'hmm_backbone_params.json'}")
print("Next: module3b_threshold_calibration.py")
