from fii.paths import ISIN_MAPPING, TRAINED_MODELS  # noqa: E402
# ============================================================================
# MODULE 17A · FACTORIAL HMM — TRAIN ON FROZEN SPLIT + FROZEN OOS DECODE
#
# The Module-2 finding was that a FLAT HMM cannot form concentration
# states: the persistent direction axis captures the likelihood mass and
# single-day concentration snapshots stay HMM-invisible, which forced the
# threshold-overlay hybrid (3B).  This module tests the structural fix:
# a FACTORIAL HMM (Ghahramani & Jordan 1997) with TWO independent chains
#   chain D (direction, k=3)      — owns the persistence axis
#   chain C (concentration, k=3)  — owns the entity-concentration axis
# emitting additively: x_t ~ N(mu_D[d_t] + mu_C[c_t], diag).  Estimation
# is exact EM on the 9-state product space via a constrained
# hmmlearn.GaussianHMM (same certified library as the naive backbone;
# only the M-step is constrained — see fhmm_stages/fhmm_core.py).
#
# Protocol mirrors 3A EXACTLY: same features, same 5d-smoothed re-ranked
# entity features, same frozen split (train <= 2021-04-30, May-Jun 2021
# masked, test >= 2021-07-01), same MIN_SEQ/FIT_CAP random-contiguous-
# block subsample (same seed), frozen decode of both eras.  Archetypes
# come END-TO-END from the two chains — NO overlay thresholds.
#
# Chain-C state naming is SIDE-AWARE (labeling correction, recorded in
# the research log: the first run's low/mid/high ordering on the mean
# entity loading mislabeled the states, because the fitted chain is
# side-specific — one state concentrates SELLING, another BUYING.
# Corrected AFTER the 17A signatures, BEFORE any 17C economics):
#   DISPERSED = lowest mean entity loading;
#   CONC_SELL = highest F_entity_s among the rest;
#   CONC_BUY  = the remaining state (check: max F_entity_buy_s).
# Archetypes:
#   HOSTAGE    = (SELL, DISPERSED)     SHARK_DIST = (SELL, CONC_SELL)
#   SHARK_ACC  = (BUY, CONC_BUY)       ROBOT      = (NEUTRAL, any)
#   UNTAGGED_DIRECTIONAL = everything else
#
# PRE-REGISTERED GATES (written before results; G1/G2 are implementation
# gates and HARD-FAIL the stage; G3/G4 are scientific verdicts, recorded
# and carried to 17B/17C either way):
#  G1 EM MONOTONE     : loglik non-decreasing (rel dips < 1e-6).
#  G2 EXACTNESS       : hmmlearn loglik == independent pure-numpy
#                       forward pass on identical product params
#                       (rel diff < 1e-8) — catches silent M-step bugs.
#  G3 CONCENTRATION CHANNEL EARNS STATES (the experiment):
#       (a) every chain-C state holds >= 5% of TRAIN stock-days;
#       (b) chain-C loading spread on F_entity_s >= 0.20 probit units.
#     PASS -> "CONCENTRATION CHANNEL FORMED";
#     FAIL -> "CONCENTRATION CHANNEL STARVED" (negative result: the
#             likelihood-starvation story extends to factorial form).
#  G4 OOS REPLICATION : frozen-decode signature drift TRAIN vs TEST
#       <= 0.15 on F_persist for chain D (naive-backbone bar), and
#       <= 0.25 on F_entity_s for chain C (rarer states, wider bar).
#
# Input : stockday_features_v2.parquet
# Output: stockday_states_fhmm.parquet, trained_models/fhmm_params.json
# ============================================================================
import datetime as dt
import json

import numpy as np
import polars as pl
from scipy.special import ndtri

from fii.models.fhmm_stages.fhmm_core import (
    FactorialGaussianHMM, numpy_product_loglik)

PANEL_IN = str(ISIN_MAPPING / "stockday_features_v2.parquet")
OUT_FHMM = str(ISIN_MAPPING / "stockday_states_fhmm.parquet")

TRAIN_END  = dt.date(2021, 4, 30)
TEST_START = dt.date(2021, 7, 1)
FEATS      = ["F_persist", "F_block", "F_entity_s", "F_entity_buy_s"]
DIR_DIMS   = [0, 1]          # F_persist, F_block
CONC_DIMS  = [2, 3]          # F_entity_s, F_entity_buy_s
MIN_SEQ    = 60
FIT_CAP    = 400
SMOOTH     = 5
N_INITS    = 5
SEED       = 42
rng = np.random.default_rng(SEED)

# ---- load + smoothed entity features (identical recipe to 3A) -------------
panel = pl.read_parquet(PANEL_IN).sort(["cisin", "TR_DATE"])
panel = panel.with_columns(
    pl.col("entity_conc_raw").rolling_mean(SMOOTH, min_samples=3)
      .over("cisin").alias("hhi_s_raw"),
    pl.col("entity_conc_buy_raw").rolling_mean(SMOOTH, min_samples=3)
      .over("cisin").alias("hhi_bs_raw"),
)
for raw, feat in {"hhi_s_raw": "F_entity_s",
                  "hhi_bs_raw": "F_entity_buy_s"}.items():
    masked = pl.when(pl.col("eligible")).then(pl.col(raw)).otherwise(None)
    panel = panel.with_columns(masked.alias("_m"))
    n_valid = pl.col("_m").is_not_null().sum().over("TR_DATE")
    rnk = pl.col("_m").rank(method="average").over("TR_DATE")
    panel = panel.with_columns((rnk / (n_valid + 1)).alias("_p"))
    p = panel["_p"].to_numpy()
    pr = np.full(p.shape, np.nan)
    mm = ~np.isnan(p)
    pr[mm] = ndtri(p[mm])
    panel = panel.with_columns(
        pl.Series(feat, pr).fill_nan(None)).drop(["_m", "_p"])

# ---- split + sequences (identical to 3A) -----------------------------------
cc = (panel.filter(pl.all_horizontal(
            [pl.col(f).is_not_null() for f in FEATS]))
          .select(["cisin", "TR_DATE"] + FEATS)
          .sort(["cisin", "TR_DATE"]))

def build_era(df, lo=None, hi=None):
    e = df
    if lo: e = e.filter(pl.col("TR_DATE") >= lo)
    if hi: e = e.filter(pl.col("TR_DATE") <= hi)
    keep = (e.group_by("cisin").agg(pl.len().alias("n"))
             .filter(pl.col("n") >= MIN_SEQ))
    e = (e.filter(pl.col("cisin").is_in(keep["cisin"].implode()))
          .sort(["cisin", "TR_DATE"]))
    x = e.select(FEATS).to_numpy()
    lens = e.group_by("cisin", maintain_order=True).agg(
        pl.len())["len"].to_list()
    return e, x, lens

train_df, x_tr, l_tr = build_era(cc, hi=TRAIN_END)
test_df,  x_te, l_te = build_era(cc, lo=TEST_START)
print(f"TRAIN: {train_df.height:,} stock-days | {len(l_tr):,} stocks "
      f"| <= {TRAIN_END}")
print(f"TEST : {test_df.height:,} stock-days | {len(l_te):,} stocks "
      f"| >= {TEST_START}")

blocks, lengths, pos = [], [], 0
for n in l_tr:
    seg = x_tr[pos:pos + n]
    if n > FIT_CAP:
        s = rng.integers(0, n - FIT_CAP + 1)
        seg = seg[s:s + FIT_CAP]
    blocks.append(seg)
    lengths.append(len(seg))
    pos += n
x_fit = np.vstack(blocks)
print(f"Fit sample: {x_fit.shape[0]:,} rows "
      f"(random contiguous blocks <= {FIT_CAP}/stock)")

# ---- fit (TRAIN only) + freeze ----------------------------------------------
best, best_ll, best_hist = None, -np.inf, None
for k in range(N_INITS):
    m = FactorialGaussianHMM(k_d=3, k_c=3, n_iter=200, tol=1e-4,
                             random_state=SEED + k)
    m.seed_init(x_fit, DIR_DIMS, CONC_DIMS, jitter_seed=SEED + k)
    m.fit(x_fit, lengths)
    ll = m.monitor_.full_history[-1]
    print(f"  init {k}: loglik = {ll:,.0f} "
          f"({len(m.monitor_.full_history)} iters)"
          f"{'  <- best' if ll > best_ll else ''}")
    if ll > best_ll:
        best, best_ll = m, ll
        best_hist = list(m.monitor_.full_history)
model = best

# ---- G1: EM monotonicity ------------------------------------------------------
h = np.array(best_hist)
dips = np.diff(h) < -1e-6 * np.abs(h[:-1])
g1 = not dips.any()
print(f"\nG1 EM MONOTONE: {'PASS' if g1 else 'FAIL'} "
      f"({dips.sum()} dips / {len(h)} iters)")

# ---- G2: exactness vs independent numpy forward pass ---------------------------
n_chk = min(len(lengths), 40)
x_chk = x_fit[:sum(lengths[:n_chk])]
ll_lib = model.score(x_chk, lengths[:n_chk])
ll_np = numpy_product_loglik(model.startprob_, model.transmat_,
                             model.means_, model._covars_,
                             x_chk, lengths[:n_chk])
rel = abs(ll_lib - ll_np) / max(abs(ll_lib), 1e-12)
g2 = rel < 1e-8
print(f"G2 EXACTNESS: {'PASS' if g2 else 'FAIL'} "
      f"(hmmlearn {ll_lib:,.2f} vs numpy {ll_np:,.2f}, rel {rel:.2e})")
if not (g1 and g2):
    raise SystemExit("implementation gate failed — halting 17A")

# ---- chain state naming ---------------------------------------------------------
d_ord = np.argsort(model.mu_d[:, 0])                     # by F_persist
D_NAME = {int(d_ord[0]): "SELL_REGIME", int(d_ord[1]): "NEUTRAL",
          int(d_ord[2]): "BUY_REGIME"}
# side-aware naming (see header): the chain is side-specific
ent = model.mu_c[:, CONC_DIMS]                           # (k_c, 2)
c_disp = int(np.argmin(ent.mean(axis=1)))
rem = [i for i in range(model.k_c) if i != c_disp]
c_sell = rem[int(np.argmax(model.mu_c[rem, CONC_DIMS[0]]))]
c_buy = [i for i in rem if i != c_sell][0]
C_NAME = {c_disp: "DISPERSED", c_sell: "CONC_SELL", c_buy: "CONC_BUY"}
c_ord = [c_disp, c_sell, c_buy]
side_ok = model.mu_c[c_buy, CONC_DIMS[1]] == model.mu_c[
    rem, CONC_DIMS[1]].max()
print(f"\nchain C naming: DISPERSED=s{c_disp} CONC_SELL=s{c_sell} "
      f"CONC_BUY=s{c_buy} | buy-side check "
      f"{'consistent' if side_ok else 'INCONSISTENT — inspect means'}")

print("\n=== chain D (direction) fitted means ===")
print(f"{'state':<12}" + "".join(f"{f:>16}" for f in FEATS) + f"{'dwell':>8}")
for s in d_ord:
    dw = 1 / (1 - model.a_d[s, s])
    print(f"{D_NAME[int(s)]:<12}"
          + "".join(f"{model.mu_d[s, i]:>16.3f}" for i in range(4))
          + f"{dw:>8.1f}")
print("\n=== chain C (concentration) fitted means (centered) ===")
print(f"{'state':<14}" + "".join(f"{f:>16}" for f in FEATS) + f"{'dwell':>8}")
for s in c_ord:
    dw = 1 / (1 - model.a_c[s, s])
    print(f"{C_NAME[int(s)]:<14}"
          + "".join(f"{model.mu_c[s, i]:>16.3f}" for i in range(4))
          + f"{dw:>8.1f}")

# ---- frozen decode of BOTH eras -------------------------------------------------
def decode(df, x, lens, era):
    prod = model.predict(x, lens)
    d, c = prod // model.k_c, prod % model.k_c
    return df.with_columns(
        pl.Series("state", [D_NAME[int(v)] for v in d]),
        pl.Series("cstate", [C_NAME[int(v)] for v in c]),
        pl.lit(era).alias("era"))

train_df = decode(train_df, x_tr, l_tr, "TRAIN")
test_df = decode(test_df, x_te, l_te, "TEST")
both = pl.concat([train_df, test_df])

both = both.with_columns(
    pl.when((pl.col("state") == "SELL_REGIME")
            & (pl.col("cstate") == "DISPERSED")).then(pl.lit("HOSTAGE"))
     .when((pl.col("state") == "SELL_REGIME")
           & (pl.col("cstate") == "CONC_SELL")).then(pl.lit("SHARK_DIST"))
     .when((pl.col("state") == "BUY_REGIME")
           & (pl.col("cstate") == "CONC_BUY")).then(pl.lit("SHARK_ACC"))
     .when(pl.col("state") == "NEUTRAL").then(pl.lit("ROBOT"))
     .otherwise(pl.lit("UNTAGGED_DIRECTIONAL")).alias("archetype"))

# ---- G3: did the concentration channel earn states? ------------------------------
tr = both.filter(pl.col("era") == "TRAIN")
census_c = (tr.group_by("cstate").agg(pl.len().alias("n"))
              .with_columns((pl.col("n") / pl.col("n").sum())
                            .alias("share")))
print("\n=== chain C census (TRAIN) ===")
print(census_c.sort("cstate"))
min_share = float(census_c["share"].min())
spread = float(model.mu_c[:, 2].max() - model.mu_c[:, 2].min())
g3a, g3b = min_share >= 0.05, spread >= 0.20
g3 = g3a and g3b
print(f"\nG3a census floor : min share {min_share:.3f} >= 0.05 -> "
      f"{'PASS' if g3a else 'FAIL'}")
print(f"G3b loading sprd : F_entity_s spread {spread:.3f} >= 0.20 -> "
      f"{'PASS' if g3b else 'FAIL'}")
print("G3 VERDICT:", "CONCENTRATION CHANNEL FORMED" if g3
      else "CONCENTRATION CHANNEL STARVED (negative result)")

# ---- G4: OOS replication (frozen decode) ------------------------------------------
print("\n=== state signatures TRAIN vs TEST (frozen — should replicate) ===")
sig = (both.group_by("era", "state")
           .agg([pl.col(f).mean().round(3) for f in FEATS])
           .sort(["state", "era"]))
print(sig)
sigc = (both.group_by("era", "cstate")
            .agg([pl.col(f).mean().round(3) for f in FEATS])
            .sort(["cstate", "era"]))
print(sigc)

def drift(df, col, feat):
    w = (df.group_by("era", col).agg(pl.col(feat).mean().alias("m"))
           .pivot(values="m", index=col, on="era"))
    return float((w["TRAIN"] - w["TEST"]).abs().max())

dr_d = drift(both, "state", "F_persist")
dr_c = drift(both, "cstate", "F_entity_s")
g4a, g4b = dr_d <= 0.15, dr_c <= 0.25
print(f"\nG4a chain D drift (F_persist)  : {dr_d:.3f} <= 0.15 -> "
      f"{'PASS' if g4a else 'FAIL'}")
print(f"G4b chain C drift (F_entity_s) : {dr_c:.3f} <= 0.25 -> "
      f"{'PASS' if g4b else 'FAIL'}")

print("\n=== archetype census (end-to-end, NO thresholds): TRAIN vs TEST ===")
print(both.group_by("era", "archetype").agg(pl.len().alias("n"))
          .with_columns((pl.col("n") / pl.col("n").sum().over("era"))
                        .round(4).alias("share"))
          .sort(["archetype", "era"]))

# ---- persist ------------------------------------------------------------------------
both.write_parquet(OUT_FHMM)
print(f"\nSaved -> {OUT_FHMM}   {both.shape}")

TRAINED_MODELS.mkdir(parents=True, exist_ok=True)
payload = model.chain_params()
payload.update({
    "features": FEATS, "dir_dims": DIR_DIMS, "conc_dims": CONC_DIMS,
    "d_names": {str(k): v for k, v in D_NAME.items()},
    "c_names": {str(k): v for k, v in C_NAME.items()},
    "loglik": float(best_ll), "seed": SEED,
    "train_cutoff": str(TRAIN_END),
    "gates": {"G1_monotone": bool(g1), "G2_exactness": bool(g2),
              "G3_channel_formed": bool(g3),
              "G3_min_share": min_share, "G3_spread": spread,
              "G4_drift_d": dr_d, "G4_drift_c": dr_c,
              "G4a": bool(g4a), "G4b": bool(g4b)}})
(TRAINED_MODELS / "fhmm_params.json").write_text(
    json.dumps(payload, indent=2))
print(f"Saved -> {TRAINED_MODELS / 'fhmm_params.json'}")
print("Next: module17b_fhmm_descriptives.py")
