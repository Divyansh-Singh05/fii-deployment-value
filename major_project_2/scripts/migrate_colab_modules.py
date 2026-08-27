"""One-time migration of the Colab research scripts into the repo.

Policy (documented in README): research stage code is preserved
byte-for-byte except for TWO mechanical substitutions, applied here and
auditable in this file:

  1. hardcoded Colab Drive paths -> imports from fii.paths
  2. the backtest scripts' `assert "backtest" in globals()` session
     coupling -> an explicit `from fii.backtest.engine import *`

Originals are archived verbatim under legacy/colab_modules/.
Any remaining '/content' or 'google.colab' reference is REPORTED so it
can be reviewed by hand — never silently rewritten.
"""
from __future__ import annotations

import shutil
from pathlib import Path

SRC = Path("/Users/divyanshsingh/Desktop/temp")
REPO = Path(__file__).resolve().parents[1]
LEGACY = REPO / "legacy" / "colab_modules"
PKG = REPO / "src" / "fii"

HEADER = (
    "# [migrated from Colab: paths now come from fii.paths; see\n"
    "#  scripts/migrate_colab_modules.py — research logic unchanged]\n"
    "from fii.paths import VALIDATION_DATA, ISIN_MAPPING  # noqa: E402\n"
)

REPLACEMENTS = [
    ('Path("/content/drive/MyDrive/VALIDATION_DATA")', "VALIDATION_DATA"),
    ("Path('/content/drive/MyDrive/VALIDATION_DATA')", "VALIDATION_DATA"),
    ('Path("/content/drive/MyDrive/ISIN_MAPPING")', "ISIN_MAPPING"),
    ("Path('/content/drive/MyDrive/ISIN_MAPPING')", "ISIN_MAPPING"),
    ('assert "backtest" in globals(), "run module12a_bt_engine.py first"',
     "from fii.backtest.engine import *  # noqa: F401,F403"),
]

# stage script -> package destination (relative to src/fii)
DEST = {
    # data preparation
    "module5a_price_panel.py": "data_prep",
    "module5b1_ca_factors.py": "data_prep",
    "module5b2_apply_adjustment.py": "data_prep",
    "module5j_canonical_panel.py": "data_prep",
    # feature engineering
    "module1_feature_store_v2.py": "features",
    # model build / calibration / description
    "module2_v4_final_hybrid_overlay.py": "models/hmm_stages",
    "module3a_model_split_oos.py": "models/hmm_stages",
    "module3b_threshold_calibration.py": "models/hmm_stages",
    "module3c_descriptive_stats.py": "models/hmm_stages",
    # economic validation battery
    "module5b4_car_diff.py": "validation",
    "module6_deal_corroboration.py": "validation",
    "module6b_liquidity_shock_profile.py": "validation",
    "module7_panel_regression.py": "validation",
    "module7b_robustness.py": "validation",
    "module8_gbt_shap.py": "validation",
    "module8b_demeaning_check.py": "validation",
    "module9_net_innov.py": "validation",
    "module10_vix_lambda.py": "validation",
    "module11_pin.py": "validation",
    # read-only audits / diagnostics (optional phase)
    "module4c_data_audit.py": "validation/audits",
    "module5b2d_diagnose.py": "validation/audits",
    "module5b3_car_start.py": "validation/audits",
    "module5c_data_lineage.py": "validation/audits",
    "module5d_attrition_diagnostic.py": "validation/audits",
    "module5f_universe_audit.py": "validation/audits",
    "module5g_isin_provenance.py": "validation/audits",
    "module5h_cisin_validation.py": "validation/audits",
    "module5i_tape_canonicalization.py": "validation/audits",
    "module5k_isin_accounting.py": "validation/audits",
    "module5l_noisin_probe.py": "validation/audits",
    "module5m_universe_integrity.py": "validation/audits",
    # backtests (12a handled separately -> fii/backtest/engine.py)
    "module12b_strategies_base.py": "backtest",
    "module12c_strategies_hmm.py": "backtest",
    "module12d_gross_diagnosis.py": "backtest",
    "module12e_style_switch.py": "backtest",
}

# preserved in legacy only (Colab-specific / superseded / exploratory)
LEGACY_ONLY = [
    "module4_data_collection.py",   # one-time Colab downloads (done)
    "module5_bootstrap.py",         # Colab session preload (obsolete)
    "module5b_forward_returns.py",  # superseded by 5b1/5b2/5b4
    "module2_v1_hmm_first_fit.py",  # exploratory HMM iterations
    "module2_v2_dissection_bic_sweep.py",
    "module2_v3_autocorr_smoothing_refit.py",
    "module12a_bt_engine.py",       # refactored into fii/backtest/engine.py
    "hmm_regime_switching.py",      # pre-project prototype
    "isin_consolidate_final.py",    # ISIN workstream (pre-pipeline)
    "isin_final_tables_v2.py",
    "isin_restructure_diagnostic.py",
    "robust_isin_mapping_2.py",
]


def main() -> None:
    LEGACY.mkdir(parents=True, exist_ok=True)
    flagged: list[str] = []
    for name in sorted(set(DEST) | set(LEGACY_ONLY)):
        src = SRC / name
        if not src.exists():
            print(f"!! missing source: {name}")
            continue
        shutil.copy2(src, LEGACY / name)          # verbatim archive
        if name not in DEST:
            print(f"legacy-only: {name}")
            continue
        text = src.read_text()
        for old, new in REPLACEMENTS:
            text = text.replace(old, new)
        dest_dir = PKG / DEST[name]
        dest_dir.mkdir(parents=True, exist_ok=True)
        (dest_dir / "__init__.py").touch()
        (dest_dir / name).write_text(HEADER + text)
        leftovers = [ln.strip() for ln in text.splitlines()
                     if "/content" in ln or "google.colab" in ln]
        status = "OK " if not leftovers else "FLAG"
        print(f"{status} {name} -> src/fii/{DEST[name]}/")
        if leftovers:
            flagged.append(name)
            for ln in leftovers[:4]:
                print("      " + ln[:74])
    print("\nflagged for manual review:", flagged or "none")


if __name__ == "__main__":
    main()
