"""Ordered manifest of every pipeline stage.

Phases run in the order given by PHASE_ORDER; stages run in the order
listed within each phase.  The ordering encodes real data dependencies:

  data_prep     bhavcopy tape -> repaired panel -> CA factors -> v2 panel
  features      raw FII trades -> stockday feature store
  model         features -> HMM backbone (frozen split) -> calibrated states
  canonical     states + v2 panel -> issuer-bounded closure -> v3 panel
  validation    the full economic-validation battery (Modules 5B4-11)
  backtest      engine gates -> baselines -> HMM twins -> diagnosis -> S4
  audit         read-only diagnostics; optional, safe to run any time
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fii.paths import REPO_ROOT

_PKG = REPO_ROOT / "src" / "fii"


@dataclass(frozen=True)
class Stage:
    name: str
    phase: str
    script: Path
    desc: str


PHASE_ORDER = ["data_prep", "features", "model", "canonical",
               "validation", "backtest", "exhibits", "phase2", "fhmm",
               "audit"]

# fmt: off
STAGES: list[Stage] = [
    # ---- data preparation ---------------------------------------------------
    Stage("price_panel", "data_prep",
          _PKG / "data_prep/module5a_price_panel.py",
          "Bhavcopy -> repaired price panel (year-0020 fix, 2011 ISIN "
          "backfill, series dedupe, macro tz fix; gates G1/G2)"),
    Stage("ca_factors", "data_prep",
          _PKG / "data_prep/module5b1_ca_factors.py",
          "Parse split/bonus corporate actions -> adjustment factors, "
          "tape-verified (99.1% confirmed)"),
    Stage("apply_adjustment", "data_prep",
          _PKG / "data_prep/module5b2_apply_adjustment.py",
          "Apply CA factors -> returns_panel_v2 (ex-day gate A: median "
          "|ret| 0.508 -> 0.038; migration guard)"),
    # ---- feature engineering ------------------------------------------------
    Stage("feature_store", "features",
          _PKG / "features/module1_feature_store_v2.py",
          "Raw FII trades -> 10 probit-ranked stock-day flow features "
          "(strictly backward windows, May-Jun 2021 masked)"),
    # ---- model --------------------------------------------------------------
    # NOTE: module2_v4 (hybrid-overlay design exploration) is preserved in
    # models/hmm_stages/ and legacy/ but is NOT a runnable stage: it
    # continues the Module-2 notebook session (needs v1-v3 objects in
    # memory). The production chain is module3a -> module3b.
    Stage("hmm_train_oos", "model",
          _PKG / "models/hmm_stages/module3a_model_split_oos.py",
          "Frozen temporal split (train<=2021-04, test>=2021-07); "
          "fit backbone, decode both eras, OOS replication checks"),
    Stage("threshold_calibration", "model",
          _PKG / "models/hmm_stages/module3b_threshold_calibration.py",
          "Overlay thresholds via TRAIN-era quantile rule (GMM "
          "falsification documented) -> stockday_states_calibrated"),
    Stage("model_descriptives", "model",
          _PKG / "models/hmm_stages/module3c_descriptive_stats.py",
          "Archetype signatures, census, transitions, era replication"),
    # ---- canonical identity -------------------------------------------------
    Stage("canonical_panel", "canonical",
          _PKG / "data_prep/module5j_canonical_panel.py",
          "Issuer-bounded ISIN closure on tape AND states -> "
          "returns_panel_v3 + states_v3 (coverage 90.4% -> 98.5%)"),
    # ---- economic validation battery ----------------------------------------
    Stage("event_study", "validation",
          _PKG / "validation/module5b4_car_diff.py",
          "Excess-CAR difference-in-differences, START & END anchors"),
    Stage("deal_corroboration", "validation",
          _PKG / "validation/module6_deal_corroboration.py",
          "Block/bulk/short-deal coincidence test (negative result, "
          "kept: visibility-threshold lesson)"),
    Stage("liquidity_shock", "validation",
          _PKG / "validation/module6b_liquidity_shock_profile.py",
          "Event arc: pressure + volume climax + reversal (mechanism)"),
    Stage("panel_regression", "validation",
          _PKG / "validation/module7_panel_regression.py",
          "PanelOLS stock+date FE, two-way clustered; specs R0/R1/R2"),
    Stage("robustness", "validation",
          _PKG / "validation/module7b_robustness.py",
          "Non-overlap, horizons 10-60, dose-response, ROBOT placebo "
          "decomposition, beta-null fix"),
    Stage("gbt_challenger", "validation",
          _PKG / "validation/module8_gbt_shap.py",
          "LightGBM challenger vs regime baseline + SHAP attribution"),
    Stage("demeaning_check", "validation",
          _PKG / "validation/module8b_demeaning_check.py",
          "Characteristics-vs-dynamics decomposition (pre-registered "
          "LSTM gate: not met)"),
    Stage("flow_innovation", "validation",
          _PKG / "validation/module9_net_innov.py",
          "NET_INNOV AR(5) yardstick: concentration != flow surprise; "
          "surprise-reversion regularity"),
    Stage("vix_lambda", "validation",
          _PKG / "validation/module10_vix_lambda.py",
          "State-dependence: VIX interaction + FII-flow Kyle lambda "
          "(mixed/null, reported honestly)"),
    Stage("pin_model", "validation",
          _PKG / "validation/module11_pin.py",
          "Easley-O'Hara FII-PIN MLE per stock-year; independent "
          "endorsement of the transitory/permanent reading"),
    Stage("flow_intensity", "data_prep",
          _PKG / "data_prep/module20_flow_intensity.py",
          "Stock-day FII gross flow and turnover with lagged 60d "
          "normalisations — input for module19 and the C4 share correction"),
    Stage("institutional_share", "validation",
          _PKG / "validation/module19_institutional_share.py",
          "Institutional share of turnover -> next-day volatility: "
          "b=-0.030 (t=-5.53) under stock+date FE, both eras, volume "
          "controlled; null at t-2 so economic, not tradeable"),
    Stage("market_flow_engine", "validation",
          _PKG / "validation/module21_market_flow_engine.py",
          "Walk-forward Nifty density engine: EWMA x Student-t base with "
          "VIX and aggregate-FII-flow tilts (PREREG_MARKET_ENGINE.md); "
          "gate = flow beyond VIX on CRPS, DM full + both eras"),
    Stage("market_flow_engine_v2", "validation",
          _PKG / "validation/module22_market_flow_engine_v2.py",
          "v2 on GJR-GARCH base + 5-feature aggregate flow set "
          "(PREREG_MARKET_ENGINE_V2.md). Gate FAILED - reported as the "
          "power boundary of the aggregate flow signal"),
    Stage("market_flow_engine_v3", "validation",
          _PKG / "validation/module23_market_flow_engine_v3.py",
          "v3 global-forces engine: GJR base + USDINR/S&P500/DXY/rate "
          "tilts (PREREG_MARKET_ENGINE_V3.md). Gate FAILED at t=-1.79 "
          "vs -2.0 bar; calibration repair + TEST-era gains recorded"),
    Stage("weekly_engine", "validation",
          _PKG / "validation/module24_weekly_engine.py",
          "h=5 engine (PREREG_WEEKLY_ENGINE.md): screen passed on S&P "
          "spillover (t=-3.5/-2.7/-4.4) but the density conversion "
          "FAILED - completes the market-level engine boundary"),
    Stage("pin_conflict", "validation",
          _PKG / "validation/module16_pin_conflict.py",
          "Audit follow-up: does PIN's stock-year loading on dispersed "
          "selling contradict the episode-level reversal? (it does not "
          "arbitrate — no episode-level bite)"),
    Stage("backbone_ablation", "validation",
          _PKG / "validation/module13a_backbone_ablation.py",
          "Referee test: census-matched rule backbone reproduces Table 1 "
          "(V1: HMM not necessary — contribution is the measure)"),
    Stage("incremental_value", "validation",
          _PKG / "validation/module13b_incremental_value.py",
          "Referee test: composition vs flow-magnitude controls + "
          "GBT with/without composition block (the +28% rel. IC claim is WITHDRAWN on the corrected feature: dIC +0.0012, t=0.67)"),
    Stage("recon_exclusion", "validation",
          _PKG / "validation/module13c_recon_exclusion.py",
          "Referee test: index-reconstitution window exclusion "
          "(passive mechanics excluded)"),
    Stage("skeptic_tests", "validation",
          _PKG / "validation/module15_skeptic_tests.py",
          "Self-audit: bounce-free windows (repriced -11/-24%), direct "
          "SD-HO contrast, public VCR head-to-head, PUBLIC->FLOW->COMP "
          "GBT ladder (all pre-registered; all pass)"),
    # ---- backtests -----------------------------------------------------------
    Stage("engine_gates", "backtest",
          _PKG / "backtest/engine_gates.py",
          "Backtest-engine correctness gates G1/G2/G3 (must PASS "
          "before any strategy runs)"),
    Stage("bt_baselines", "backtest",
          _PKG / "backtest/module12b_strategies_base.py",
          "Three no-model baselines (REV20, FLOW10, PROXY), frozen"),
    Stage("bt_hmm_twins", "backtest",
          _PKG / "backtest/module12c_strategies_hmm.py",
          "HMM-conditioned twins + pre-registered dSharpe verdicts"),
    Stage("bt_gross_diagnosis", "backtest",
          _PKG / "backtest/module12d_gross_diagnosis.py",
          "Signal-vs-implementation: gross dSharpe, breakeven costs"),
    Stage("bt_style_switch", "backtest",
          _PKG / "backtest/module12e_style_switch.py",
          "S4 style-switch (trend/reversion/hold by regime) pair"),
    # ---- paper exhibits --------------------------------------------------------
    Stage("paper_exhibits", "exhibits",
          _PKG / "reporting/make_exhibits.py",
          "Export every paper table (CSV+LaTeX) and figure (PNG) to "
          "outputs/tables and outputs/figures"),
    Stage("collect_outputs", "exhibits",
          _PKG / "reporting/collect_outputs.py",
          "Populate outputs/: predictions, trained_models, validation "
          "reports, descriptive stats, metrics, run summary"),
    # ---- Phase II (charter: docs/PHASE2_PLAN.md; run explicitly) -------------
    Stage("phase2_filtering", "phase2",
          _PKG / "phase2/module16a_causal_filtering.py",
          "16A: forward-filtered posteriors from frozen params; gate A2 "
          "= Table 1 survives on fully causal labels"),
    Stage("phase2_calibration", "phase2",
          _PKG / "phase2/module16b_calibration.py",
          "16B: nowcast calibration — Brier/log-loss/ECE vs census "
          "prior and label-Markov persistence baselines"),
    Stage("phase2_hazard", "phase2",
          _PKG / "phase2/module16c_hazard.py",
          "16C: episode-END hazard model, walk-forward yearly refits; "
          "bar = beat age-only KM baseline (k=1,3) at paired t>=2"),
    Stage("phase2_decision", "phase2",
          _PKG / "phase2/module16d_decision.py",
          "16D: anticipation vs confirmation, paired within episode, "
          "cost-neutral; theta walk-forward; bar = TEST gain CI>0 AND "
          "beats KM anticipator (t>=2)"),
    # ---- factorial HMM challenger (Module 17; separate code from the naive
    #      HMM chain by design — run explicitly, not part of --all) -----------
    Stage("fhmm_train_oos", "fhmm",
          _PKG / "models/fhmm_stages/module17a_fhmm_train_oos.py",
          "17A: factorial HMM (direction x concentration chains), frozen "
          "split, end-to-end archetypes, gates G1-G4"),
    Stage("fhmm_descriptives", "fhmm",
          _PKG / "models/fhmm_stages/module17b_fhmm_descriptives.py",
          "17B: chain census/dwell/transitions + agreement vs the "
          "calibrated naive-HMM archetypes (kappa, overlap)"),
    Stage("fhmm_table1", "fhmm",
          _PKG / "models/fhmm_stages/module17c_fhmm_table1.py",
          "17C: Table-1 PanelOLS on FHMM labels vs naive-HMM labels, "
          "pre-registered verdicts V1/V2/V3"),
    Stage("fhmm_filtering", "fhmm",
          _PKG / "models/fhmm_stages/module17d_fhmm_filtering.py",
          "17D: causal forward filtering on the product space (16A "
          "protocol); gate A2F = smoothed-FHMM economics survive"),
    Stage("fhmm_calibration", "fhmm",
          _PKG / "models/fhmm_stages/module17e_fhmm_calibration.py",
          "17E: nowcast calibration for BOTH chains vs causal "
          "baselines (16B protocol, amended baseline inherited)"),
    Stage("fhmm_hazard", "fhmm",
          _PKG / "models/fhmm_stages/module17f_fhmm_hazard.py",
          "17F: episode-END hazard, walk-forward yearly refits, AUC; "
          "bar = beat age-only KM at paired t>=2 (16C protocol)"),
    Stage("fhmm_decision", "fhmm",
          _PKG / "models/fhmm_stages/module17g_fhmm_decision.py",
          "17G: anticipation vs confirmation, theta walk-forward, "
          "KM-anticipator control (16D protocol)"),
    # ---- tail-probe extension study (Module 14; run in order a->b->c) --------
    Stage("tail_census", "audit",
          _PKG / "validation/module14a_tail_census.py",
          "Tail probe 14A: feasibility census of the attributable "
          "illiquid tail (gates G1-G3)"),
    Stage("tail_labels", "audit",
          _PKG / "validation/module14b_tail_labels.py",
          "Tail probe 14B: method-frozen rule labels on the tail "
          "(census/clustering/power gates)"),
    Stage("tail_economics", "audit",
          _PKG / "validation/module14c_tail_economics.py",
          "Tail probe 14C: verdict NO TAIL EFFECT — dislocation "
          "transfers, reversal does not; friction bar ~100bp"),
    # ---- read-only audits (optional) -----------------------------------------
    Stage("data_audit", "audit",
          _PKG / "validation/audits/module4c_data_audit.py",
          "Raw collected-data audit (dates, ISIN nulls, tz shifts)"),
    Stage("adjustment_diagnose", "audit",
          _PKG / "validation/audits/module5b2d_diagnose.py",
          "Read-only diagnosis of the stored v2 panel returns"),
    Stage("car_start_legacy", "audit",
          _PKG / "validation/audits/module5b3_car_start.py",
          "START-anchor CARs (inference superseded by event_study)"),
    Stage("data_lineage", "audit",
          _PKG / "validation/audits/module5c_data_lineage.py",
          "Table heads along the full derivation chain"),
    Stage("attrition", "audit",
          _PKG / "validation/audits/module5d_attrition_diagnostic.py",
          "Join-attrition decomposition (model vs tape)"),
    Stage("universe_audit", "audit",
          _PKG / "validation/audits/module5f_universe_audit.py",
          "Model-universe provenance and count integrity"),
    Stage("isin_provenance", "audit",
          _PKG / "validation/audits/module5g_isin_provenance.py",
          "5,960 -> 3,812 -> 946 ISIN funnel accounting"),
    Stage("cisin_validation", "audit",
          _PKG / "validation/audits/module5h_cisin_validation.py",
          "Canonical-ISIN validity checks"),
    Stage("tape_canonicalization", "audit",
          _PKG / "validation/audits/module5i_tape_canonicalization.py",
          "Coverage recovery measurement (90.4% -> 98.2%)"),
    Stage("isin_accounting", "audit",
          _PKG / "validation/audits/module5k_isin_accounting.py",
          "Active/inactive closure accounting; entity-boundary rule"),
    Stage("noisin_probe", "audit",
          _PKG / "validation/audits/module5l_noisin_probe.py",
          "Degenerate-key (NOISIN/null) audit: 0 model impact"),
    Stage("universe_integrity", "audit",
          _PKG / "validation/audits/module5m_universe_integrity.py",
          "Is 946 the right company count? (939 issuers, fragments)"),
]
# fmt: on


def by_name(name: str) -> Stage:
    for s in STAGES:
        if s.name == name:
            return s
    raise KeyError(f"unknown stage '{name}' — try: pipeline.py --list")


def phase_stages(phase: str) -> list[Stage]:
    return [s for s in STAGES if s.phase == phase]
