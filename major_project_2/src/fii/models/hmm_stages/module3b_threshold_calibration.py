# [migrated from Colab: paths now come from fii.paths; see
#  scripts/migrate_colab_modules.py — research logic unchanged]
from fii.paths import VALIDATION_DATA, ISIN_MAPPING  # noqa: E402
# ============================================================================
# MODULE 3B · OVERLAY-THRESHOLD CALIBRATION (replaces the arbitrary ±0.5)
#
# Derives the archetype thresholds statistically, TRAIN era only:
#   - GMM (1/2/3 components) on F_entity_s within the SELL regime, BIC-selected.
#     Threshold = posterior-0.5 boundary of the tail component (equivalent to
#     the intersection of the mixing-weight-scaled PDFs — correctly demands
#     more evidence before assigning the rare class than a raw PDF crossing).
#   - Built-in FALSIFICATION: if BIC prefers 1 component, there is NO natural
#     boundary — the cut becomes an explicit quantile DESIGN CHOICE, and the
#     doc must say so.
#   - k-means k=3 cross-check (should land near the GMM boundary).
#   - STABILITY check: re-derive on TEST era; a real boundary shouldn't move.
# Then applies the calibrated overlay to both eras and saves.
#
# Input : stockday_states_split.parquet  (from 3A)
# Output: stockday_states_calibrated.parquet  (input to 3C)
# ============================================================================
import numpy as np
import polars as pl
from pathlib import Path
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

parquet_path = ISIN_MAPPING
IN_SPLIT  = str(parquet_path / "stockday_states_split.parquet")
OUT_CALIB = str(parquet_path / "stockday_states_calibrated.parquet")
SEED       = 42
Q_FALLBACK = 0.25    # explicit design-choice quantile if GMM falsifies

both = pl.read_parquet(IN_SPLIT)
train = both.filter(pl.col("era") == "TRAIN")
test  = both.filter(pl.col("era") == "TEST")

def gmm_threshold(vals, tail, label):
    """BIC-selected 1-D GMM; posterior-0.5 boundary of the tail component.
    tail='low' → dispersed (Hostage); 'high' → concentrated (Shark)."""
    v = vals.reshape(-1, 1)
    gms, bics = {}, {}
    for k in (1, 2, 3):
        gm = GaussianMixture(k, covariance_type="full", random_state=SEED, n_init=3).fit(v)
        gms[k], bics[k] = gm, gm.bic(v)
    k_best = min(bics, key=bics.get)
    print(f"\n[{label}] BIC: " + "  ".join(f"k={k}: {bics[k]:,.0f}" for k in (1, 2, 3))
          + f"   → k_best={k_best}")
    if k_best == 1:
        print(f"[{label}] FALSIFICATION: BIC prefers ONE component — no natural "
              f"boundary in this feature. Quantile design-choice fallback engaged.")
        return None
    gm = gms[k_best]
    comp = int(np.argmin(gm.means_)) if tail == "low" else int(np.argmax(gm.means_))
    grid = np.linspace(-3, 3, 1201).reshape(-1, 1)
    post = gm.predict_proba(grid)[:, comp]
    inside = grid.ravel()[post >= 0.5]
    if inside.size == 0:
        print(f"[{label}] tail component never reaches posterior 0.5 — too weak; quantile fallback.")
        return None
    thr = float(inside.max() if tail == "low" else inside.min())
    # [numpy>=2 compat: .item() instead of float(1-elem array); logic unchanged]
    w, mu = gm.weights_[comp], gm.means_[comp].item()
    print(f"[{label}] tail component: weight={w:.2f}, mean={mu:+.2f} → posterior-0.5 boundary = {thr:+.3f}")
    return thr

# ── derive on TRAIN only ───────────────────────────────────────────────────
sell_tr = train.filter(pl.col("state") == "SELL_REGIME")["F_entity_s"].to_numpy()
buy_tr  = train.filter(pl.col("state") == "BUY_REGIME")["F_entity_buy_s"].to_numpy()

thr_h = gmm_threshold(sell_tr, "low",  "HOSTAGE   | F_entity_s in SELL (TRAIN)")
thr_s = gmm_threshold(buy_tr,  "high", "SHARK_ACC | F_entity_buy_s in BUY (TRAIN)")
thr_sd = gmm_threshold(sell_tr, "high", "SHARK_DIST| F_entity_s in SELL (TRAIN)")

if thr_h  is None: thr_h  = float(np.quantile(sell_tr, Q_FALLBACK));     print(f"HOSTAGE fallback (q{Q_FALLBACK}) → {thr_h:+.3f}")
if thr_s  is None: thr_s  = float(np.quantile(buy_tr, 1 - Q_FALLBACK));  print(f"SHARK_ACC fallback (q{1-Q_FALLBACK}) → {thr_s:+.3f}")
if thr_sd is None: thr_sd = float(np.quantile(sell_tr, 1 - Q_FALLBACK)); print(f"SHARK_DIST fallback (q{1-Q_FALLBACK}) → {thr_sd:+.3f}")

# ── k-means cross-check (sell side) ────────────────────────────────────────
km = KMeans(3, n_init=10, random_state=SEED).fit(sell_tr.reshape(-1, 1))
c = np.sort(km.cluster_centers_.ravel())
print(f"\nk-means cross-check (sell): centers={np.round(c,2)} → "
      f"low boundary ≈ {(c[0]+c[1])/2:+.3f} (vs GMM {thr_h:+.3f}) | "
      f"high boundary ≈ {(c[1]+c[2])/2:+.3f} (vs GMM {thr_sd:+.3f})")

# ── stability: re-derive on TEST — a real boundary shouldn't move ──────────
sell_te = test.filter(pl.col("state") == "SELL_REGIME")["F_entity_s"].to_numpy()
thr_h_te = gmm_threshold(sell_te, "low", "STABILITY | F_entity_s in SELL (TEST)")
if thr_h_te is not None:
    d = abs(thr_h - thr_h_te)
    print(f"STABILITY: train {thr_h:+.3f} vs test {thr_h_te:+.3f}  (Δ={d:.3f} — "
          f"{'STABLE' if d < 0.25 else 'UNSTABLE — prefer a quantile rule'})")
    # [Module-3 decision made BINDING: the original run observed this same
    #  falsification (49/51 coin-flip GMM, Δ=0.665) and adopted the quantile
    #  rule for ALL thresholds — the frozen published thresholds
    #  (-0.513 / +0.877 / +0.795). The Colab session applied it manually;
    #  this makes the script self-contained. See FII_Module3_validation_log.]
    if d >= 0.25:
        thr_h = float(np.quantile(sell_tr, Q_FALLBACK))
        thr_sd = float(np.quantile(sell_tr, 1 - Q_FALLBACK))
        thr_s = float(np.quantile(buy_tr, 1 - Q_FALLBACK))
        print(f"GMM FALSIFIED → quantile rule adopted: HOSTAGE {thr_h:+.3f} "
              f"| SHARK_DIST {thr_sd:+.3f} | SHARK_ACC {thr_s:+.3f}")

# ── apply calibrated overlay to BOTH eras (thresholds frozen from TRAIN) ───
both = both.with_columns(
    pl.when((pl.col("state") == "SELL_REGIME") & (pl.col("F_entity_s") < thr_h)).then(pl.lit("HOSTAGE"))
     .when((pl.col("state") == "SELL_REGIME") & (pl.col("F_entity_s") > thr_sd)).then(pl.lit("SHARK_DIST"))
     .when((pl.col("state") == "BUY_REGIME") & (pl.col("F_entity_buy_s") > thr_s)).then(pl.lit("SHARK_ACC"))
     .when(pl.col("state") == "NEUTRAL").then(pl.lit("ROBOT"))
     .otherwise(pl.lit("UNTAGGED_DIRECTIONAL")).alias("archetype")
)

print("\n=== CALIBRATED ARCHETYPE CENSUS: TRAIN vs TEST ===")
print(both.group_by("era", "archetype").agg(pl.len().alias("n"))
          .with_columns((pl.col("n") / pl.col("n").sum().over("era")).round(4).alias("share"))
          .sort(["archetype", "era"]))

both.write_parquet(OUT_CALIB)

# [additive artifact export — the frozen overlay thresholds ARE half the
#  model; persist next to the backbone params]
import json
from fii.paths import TRAINED_MODELS
TRAINED_MODELS.mkdir(parents=True, exist_ok=True)
(TRAINED_MODELS / "overlay_thresholds.json").write_text(json.dumps(
    {"hostage_f_entity_s_max": thr_h,
     "shark_dist_f_entity_s_min": thr_sd,
     "shark_acc_f_entity_buy_s_min": thr_s,
     "rule": "TRAIN quantile (q25/q75) after GMM stability falsification",
     "train_cutoff": "2021-04-30"}, indent=2))
print(f"\nThresholds → HOSTAGE: F_entity_s < {thr_h:+.3f} | SHARK_DIST: > {thr_sd:+.3f} "
      f"| SHARK_ACC: F_entity_buy_s > {thr_s:+.3f}")
print(f"Saved → {OUT_CALIB}   {both.shape}")
print("Next: module3c_descriptive_stats.py")
