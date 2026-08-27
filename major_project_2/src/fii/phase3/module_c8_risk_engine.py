"""
MODULE C8 · FLOWSENSE RISK ENGINE  —  query interface  (Phase III, final)

Ask for any stock over any date window and get, for every day in that window,
the risk numbers the walk-forward system would have produced ON that day using
only information available at the time.

    python -m fii.phase3.module_c8_risk_engine --stock HDFCBANK \
        --start 2020-01-01 --end 2020-06-30 --horizon 1

    from fii.phase3.module_c8_risk_engine import RiskEngine
    eng = RiskEngine()
    res = eng.query("DHFL", "2018-08-01", "2018-10-31", horizon=1)

HOW A DAY IS PRICED.  For a trading day t:

  1. VINTAGE.  Pick the latest monthly parameter vintage whose `asof` is
     strictly before t (C1). That fixes the HMM emission means, covariances,
     transition matrix and the four archetype thresholds — the "params we
     deduced via HMM" for that month. Nothing fitted after t can touch it.
  2. STATE.  Forward-filter the flow features to P(S_t | x_1..t), propagating
     across trading gaps with A^k (C2). Never Viterbi, never smoothed.
  3. ARCHETYPE.  Convert the three state probabilities into seven archetype
     probabilities using the vintage's thresholds and their bootstrap standard
     errors (C3).
  4. DENSITY.  Mix that vintage's seven past-only outcome densities, each
     built under the s + h <= asof embargo (C4), with those probabilities.
  5. RISK.  Read VaR and ES off the mixture and rescale by the stock's
     past-only EWMA volatility (C5).

Everything above is precomputed and stored. This module is a query layer, and
it VERIFIES that by rebuilding the mixture from the stored densities and
checking it reproduces the stored VaR/ES to 1e-6 (gate K1). If that gate ever
fails, the engine is out of step with the validated pipeline and says so.

BENCHMARK.  The parametric EWMA-Normal engine shares the same volatility model
and the same dates; in standardised space it is exactly N(0,1), so its
VaR/ES are closed form and carry no estimation error. The only difference
between the two engines is the assumed shape of the standardised distribution.

VALIDATION ON THE QUERIED WINDOW.  Kupiec unconditional coverage at 1% and 5%,
ES coverage (realised mean shortfall / predicted ES), mean CRPS, and PIT
summary — for both engines, on exactly the rows returned. Small windows are
reported with an explicit low-power warning rather than a bare p-value.

LINEAGE — the modules this engine is built on, in order:
    src/fii/features/module1_feature_store_v2.py     flow features
    src/fii/models/hmm_stages/module3a_model_split_oos.py   HMM backbone
    src/fii/models/hmm_stages/module3b_threshold_calibration.py  thresholds
    src/fii/phase2/module16a_causal_filtering.py     causal filter (frozen)
    src/fii/phase3/module_c1_refit_harness.py        106 monthly vintages
    src/fii/phase3/module_c2_daily_filter.py         walk-forward posteriors
    src/fii/phase3/module_c3_archetype_probs.py      soft archetype overlay
    src/fii/phase3/module_c4_outcome_densities.py    past-only densities
    src/fii/phase3/module_c5_predictive_mixture.py   mixture + VaR/ES/CRPS
    src/fii/phase3/module_c6_validation.py           pre-registered tests
    src/fii/phase3/module_c7_stock_risk_profiles.py  comparative profiles
"""
from __future__ import annotations

import argparse
import datetime as dt
import json

import numpy as np
import polars as pl
from scipy.stats import chi2, norm

from fii.paths import OUTPUTS, VALIDATION_DATA

ARCH = ["HOSTAGE", "SELL_MID", "SHARK_DIST", "ROBOT",
        "DISPERSED_ACC", "BUY_MID", "SHARK_ACC"]

# AUDIT ITEM 8. h=20 is retired from the public interface. It breaches 5.67%
# at the 5% level and 1.36% at 1% (Kupiec p_adj 0.003), and item 4 ruled out
# thin tail support as the cause, so the failure is structural: consecutive
# 20-day windows overlap by 19 days, leaving roughly a twentieth of the
# independent observations the row count implies. It remains available as an
# internal diagnostic via allow_internal=True and is never returned by a
# default query.
PUBLIC_HORIZONS = (1, 5)
INTERNAL_HORIZONS = (20,)
HORIZONS = PUBLIC_HORIZONS + INTERNAL_HORIZONS
LEVELS = ((0.05, "05"), (0.01, "01"))

# AUDIT ITEM 5. Policy for a stock the in-effect vintage never saw. `safest`
# withholds the empirical VaR/ES entirely; the parametric benchmark, which
# needs no fitted parameters for that stock, is still returned and labelled.
OUT_OF_WINDOW_POLICY = "suppress"      # "suppress" | "flag_only"


def _breach(z, var):
    """Breach indicator that abstains instead of guessing.

    `z < NaN` is False in NumPy, which silently converts "no forecast" into
    "no breach" — the item-4 defect. Returns None where either side is absent.
    """
    ok = np.isfinite(z) & np.isfinite(var)
    return [bool(z[i] < var[i]) if ok[i] else None for i in range(len(z))]


def _kupiec(x, n, a):
    if n <= 0 or x <= 0 or x >= n:
        return np.nan
    pi = x / n
    lr = -2.0 * ((n - x) * np.log(1 - a) + x * np.log(a)
                 - (n - x) * np.log(1 - pi) - x * np.log(pi))
    return float(1.0 - chi2.cdf(lr, 1))


class RiskEngine:
    """Query layer over the validated Phase III walk-forward pipeline."""

    def __init__(self, verify: bool = True):
        p3 = OUTPUTS / "phase3"
        self.pred = pl.read_parquet(p3 / "predictive.parquet")
        self.arch = pl.read_parquet(p3 / "archetype_probs.parquet")
        self.vt = pl.read_parquet(p3 / "vintages.parquet").sort("asof")
        self.D = np.load(p3 / "outcome_densities.npz", allow_pickle=True)
        self.grid = self.D["grid"]
        self.mid = 0.5 * (self.grid[1:] + self.grid[:-1])

        rp = (pl.read_parquet(VALIDATION_DATA / "returns_panel_v3.parquet")
                .select(["isin", "symbol"]).drop_nulls()
                .unique(subset=["isin"], keep="last"))
        self.sym2isin = {}
        for i, s in zip(rp["isin"].to_list(), rp["symbol"].to_list()):
            self.sym2isin.setdefault(str(s).upper(), i)
        self.isin2sym = dict(zip(rp["isin"].to_list(), rp["symbol"].to_list()))

        self.nvar = {a: float(norm.ppf(a)) for a, _ in LEVELS}
        self.nes = {a: float(-norm.pdf(norm.ppf(a)) / a) for a, _ in LEVELS}
        self.cov_lo = self.pred["TR_DATE"].min()
        self.cov_hi = self.pred["TR_DATE"].max()
        self.verify_ok = self._verify() if verify else None

    # ---------------------------------------------------------------- utils
    def resolve(self, stock: str) -> str:
        s = str(stock).strip()
        if s in self.isin2sym:
            return s
        u = s.upper()
        if u in self.sym2isin:
            return self.sym2isin[u]
        known = self.pred["cisin"].unique().to_list()
        cands = [k for k in known
                 if u in str(self.isin2sym.get(k, "")).upper()]
        if len(cands) == 1:
            return cands[0]
        raise KeyError(
            f"'{stock}' not recognised. Give an NSE symbol (e.g. HDFCBANK) or "
            f"a 12-character ISIN. {len(known)} stocks are covered; "
            f"{'candidates: ' + ', '.join(self.isin2sym.get(c, c) for c in cands[:8]) if cands else 'no near matches'}.")

    def _verify(self, n_probe: int = 300, horizon: int = 5) -> dict:
        """K1 — rebuild the mixture from stored densities and check it
        reproduces the stored VaR/ES. Proves this engine is in step with the
        pipeline that C6 and C7 validated."""
        rng = np.random.default_rng(7)
        j = self.pred.with_row_index("_i").filter(
            pl.col(f"soft_roll_h{horizon}_var05").is_finite())
        take = rng.choice(j.height, size=min(n_probe, j.height), replace=False)
        sub = j[take]
        W = self.arch[sub["_i"].to_numpy()].select(
            [f"pa_{a}" for a in ARCH]).to_numpy()
        C = self.D[f"cdf__{horizon}__soft__roll5y"]
        worst_v = worst_e = 0.0
        for k in range(sub.height):
            vi = int(sub["vintage_id"][k])
            M = W[k] @ C[vi]
            pdf = np.clip(np.diff(M), 0, None)
            pdf = pdf / max(pdf.sum(), 1e-300)
            cp = np.cumsum(pdf)
            cm = np.cumsum(pdf * self.mid)
            for a, tag in LEVELS:
                idx = int(np.clip((cp < a).sum(), 0, len(self.mid) - 1))
                worst_v = max(worst_v, abs(self.mid[idx]
                                           - float(sub[f"soft_roll_h{horizon}_var{tag}"][k])))
                worst_e = max(worst_e, abs(cm[idx] / a
                                           - float(sub[f"soft_roll_h{horizon}_es{tag}"][k])))
        return {"n": sub.height, "max_var_err": worst_v, "max_es_err": worst_e,
                "pass": bool(worst_v < 1e-6 and worst_e < 1e-6)}

    # ---------------------------------------------------------------- query
    def query(self, stock, start=None, end=None, horizon: int = 1,
              allow_internal: bool = False):
        if horizon not in PUBLIC_HORIZONS and not allow_internal:
            if horizon in INTERNAL_HORIZONS:
                raise ValueError(
                    f"horizon {horizon} is retired from the public interface "
                    f"(audit item 8): it is not calibrated — 5.67% / 1.36% "
                    f"breach against 5% / 1% targets, Kupiec p_adj 0.003. "
                    f"Use {PUBLIC_HORIZONS}, or pass allow_internal=True to "
                    f"obtain it as a diagnostic.")
            raise ValueError(f"horizon must be one of {PUBLIC_HORIZONS}")
        isin = self.resolve(stock)
        lo = dt.date.fromisoformat(str(start)) if start else self.cov_lo
        hi = dt.date.fromisoformat(str(end)) if end else self.cov_hi

        idx = (self.pred.with_row_index("_i")
               .filter((pl.col("cisin") == isin)
                       & (pl.col("TR_DATE") >= lo)
                       & (pl.col("TR_DATE") <= hi))
               .sort("TR_DATE"))
        notes = []
        if lo < self.cov_lo:
            notes.append(f"window starts before coverage; first scored day is "
                         f"{self.cov_lo} (2011-2016 builds the first vintage)")
        if hi > self.cov_hi:
            notes.append(f"window ends after coverage; last scored day is "
                         f"{self.cov_hi}")
        if idx.height == 0:
            return {"isin": isin, "symbol": self.isin2sym.get(isin, "?"),
                    "horizon": horizon, "n": 0, "notes": notes + [
                        "no FII-active scored days for this stock in this "
                        "window (the panel only carries days with FII flow)"],
                    "table": None}

        rows = idx["_i"].to_numpy()
        A = self.arch[rows]
        vid = idx["vintage_id"].to_numpy()
        # the horizon-specific scale the outcome was standardised by
        sig = idx[f"sigma_adj_h{horizon}"].to_numpy()
        z = idx[f"z_h{horizon}"].to_numpy()
        rt = np.sqrt(horizon)

        out = idx.select(["TR_DATE", "era", "vintage_id", "p_max", "pa_max",
                          "arch_soft", "arch_hard", "sigma_ewma"])
        # AUDIT ITEM 5 — the out-of-window flag now reaches the caller and,
        # under the default policy, actually suppresses the empirical forecast.
        # Previously it was computed in C2, printed once, carried inert through
        # C3/C4/C5 and never selected here, so it changed nothing operationally.
        oow = idx["stock_out_of_window"].to_numpy().astype(bool)
        out = out.with_columns(
            pl.Series("param_age_days", A["param_age_days"].to_numpy()),
            pl.Series("vintage_asof", self.vt["asof"].to_numpy()[vid]),
            pl.Series("stock_out_of_window", oow),
            *[pl.Series(f"pa_{a}", A[f"pa_{a}"].to_numpy()) for a in ARCH])
        if oow.any():
            notes.append(
                f"{int(oow.sum())} of {len(oow)} days fall outside the fitting "
                f"window of the vintage in force (the stock was not in that "
                f"refit)" + ("; Flowsense VaR/ES withheld on those days, the "
                             "parametric benchmark is still shown"
                             if OUT_OF_WINDOW_POLICY == "suppress" else ""))
        for a, tag in LEVELS:
            fv = idx[f"soft_roll_h{horizon}_var{tag}"].to_numpy().copy()
            fe = idx[f"soft_roll_h{horizon}_es{tag}"].to_numpy().copy()
            if OUT_OF_WINDOW_POLICY == "suppress" and oow.any():
                fv[oow] = np.nan
                fe[oow] = np.nan
            out = out.with_columns(
                pl.Series(f"fs_var{tag}_z", fv),
                pl.Series(f"fs_es{tag}_z", fe),
                pl.Series(f"nm_var{tag}_z", np.full(len(fv), self.nvar[a])),
                pl.Series(f"nm_es{tag}_z", np.full(len(fv), self.nes[a])),
                pl.Series(f"fs_var{tag}_pct",
                          100 * np.expm1(fv * sig * rt)),
                pl.Series(f"fs_es{tag}_pct", 100 * np.expm1(fe * sig * rt)),
                pl.Series(f"nm_var{tag}_pct",
                          100 * np.expm1(self.nvar[a] * sig * rt)),
                pl.Series(f"nm_es{tag}_pct",
                          100 * np.expm1(self.nes[a] * sig * rt)))
        out = out.with_columns(
            pl.Series(f"realised_z", z),
            pl.Series("realised_pct", 100 * np.expm1(z * sig * rt)),
            # NaN-safe: an abstained forecast yields a null flag, never False.
            # _validate excludes those rows from both numerator and denominator.
            pl.Series("fs_breach05", _breach(z, out["fs_var05_z"].to_numpy())),
            pl.Series("fs_breach01", _breach(z, out["fs_var01_z"].to_numpy())),
            pl.Series("nm_breach05", _breach(z, np.full(len(z), self.nvar[0.05]))),
            pl.Series("nm_breach01", _breach(z, np.full(len(z), self.nvar[0.01]))),
            pl.Series("fs_crps", idx[f"soft_roll_h{horizon}_crps"].to_numpy()),
            pl.Series("nm_crps", idx[f"normal_h{horizon}_crps"].to_numpy()),
            pl.Series("fs_pit", idx[f"soft_roll_h{horizon}_pit"].to_numpy()),
            pl.Series("nm_pit", idx[f"normal_h{horizon}_pit"].to_numpy()))

        val = self._validate(out, horizon)
        params = self._params(int(vid[-1]))
        return {"isin": isin, "symbol": self.isin2sym.get(isin, "?"),
                "horizon": horizon, "n": out.height, "notes": notes,
                "table": out, "validation": val, "params_last": params,
                "window": (str(out["TR_DATE"][0]), str(out["TR_DATE"][-1]))}

    def _validate(self, t, horizon):
        """AUDIT ITEM 4 — abstention is accounted for, not silently absorbed.

        The pre-audit mask was `ok = isfinite(realised_z)` alone, and breach
        flags come from `z < var`. In NumPy `z < NaN` is False, so every row
        whose VaR had been suppressed (J6 thin-tail floor, or pre-frontier)
        was counted as a NON-BREACH and left in the denominator. Measured on
        the full panel that understated Flowsense's 1% breach rate by 5.9% to
        6.8% relative, and the bias was one-sided: the parametric engine's VaR
        is closed form and never NaN, so the benchmark could not benefit.

        The mask is now per (engine, level) and requires a finite forecast.
        Availability is reported separately, because "we abstained" and "we
        were right" are different statements and must not be summed.
        """
        z = t["realised_z"].to_numpy()
        has_z = np.isfinite(z)
        v = {"n_scored": t.height, "n_with_outcome": int(has_z.sum()),
             "horizon": horizon}
        if not has_z.any():
            v["warning"] = ("no realised outcomes in this window — the last "
                            f"{horizon} trading days never mature, and rows "
                            "before the C4 warm-up frontier are unscored")
            return v

        # forecast availability, independent of whether it was any good
        for a, tag in LEVELS:
            fin = np.isfinite(t[f"fs_var{tag}_z"].to_numpy())
            v[f"forecast_available_{tag}"] = float(fin.mean())
            v[f"forecast_available_{tag}_n"] = int(fin.sum())
            v[f"abstained_{tag}_n"] = int((~fin).sum())

        for eng, pre in (("flowsense", "fs"), ("normal", "nm")):
            for a, tag in LEVELS:
                var = t[f"{pre}_var{tag}_z"].to_numpy()
                ok = has_z & np.isfinite(var)          # <- the fix
                n = int(ok.sum())
                v[f"{eng}_n{tag}"] = n
                if n == 0:
                    v[f"{eng}_breach{tag}"] = np.nan
                    continue
                br = (z[ok] < var[ok])
                x = int(br.sum())
                v[f"{eng}_breach{tag}"] = x / n
                v[f"{eng}_breach{tag}_count"] = x
                v[f"{eng}_kupiec_p{tag}"] = _kupiec(x, n, a)
                if x:
                    pe = float(np.nanmean(t[f"{pre}_es{tag}_z"].to_numpy()[ok][br]))
                    re = float(np.nanmean(z[ok][br]))
                    v[f"{eng}_es{tag}_pred"] = pe
                    v[f"{eng}_es{tag}_real"] = re
                    v[f"{eng}_es{tag}_coverage"] = re / pe if pe else np.nan
            c = t[f"{pre}_crps"].to_numpy()
            v[f"{eng}_crps"] = float(np.nanmean(c)) if np.isfinite(c).any() else np.nan
            u = t[f"{pre}_pit"].to_numpy()
            u = u[np.isfinite(u)]
            v[f"{eng}_pit_mean"] = float(u.mean()) if len(u) else np.nan

        # the head-to-head is scored on rows where BOTH engines produced a
        # number, so a Flowsense abstention cannot flatter its own average
        cf = t["fs_crps"].to_numpy()
        cn = t["nm_crps"].to_numpy()
        both = np.isfinite(cf) & np.isfinite(cn)
        v["n_paired_crps"] = int(both.sum())
        if both.any():
            v["crps_gain_pct"] = float(
                100 * (1 - cf[both].mean() / cn[both].mean()))

        n1 = v.get("flowsense_n01", 0)
        exp1 = 0.01 * n1
        v["low_power"] = n1 < 250 or exp1 < 5
        if v["low_power"]:
            v["warning"] = (f"only {n1} observations carry both an outcome and "
                            f"a 1% forecast ({exp1:.1f} breaches expected) — "
                            "the Kupiec p-values are descriptive only")
        return v

    def _params(self, vi):
        r = self.vt[vi]
        mu = np.array(json.loads(r["means"][0]))
        A = np.array(json.loads(r["transmat"][0]))
        return {"vintage_id": vi, "asof": str(r["asof"][0]),
                "window_start": str(r["window_start"][0]),
                "n_stocks": int(r["n_stocks"][0]),
                "n_fit_rows": int(r["n_fit_rows"][0]),
                "self_transitions": [float(A[i, i]) for i in range(3)],
                "means_F_persist": [float(x) for x in mu[:, 0]],
                "th_hostage": float(r["th_hostage"][0]),
                "th_shark_dist": float(r["th_shark_dist"][0]),
                "th_dispersed_acc": float(r["th_dispersed_acc"][0]),
                "th_shark_acc": float(r["th_shark_acc"][0])}


# --------------------------------------------------------------- reporting
def render(res, max_rows=14):
    L = []
    sym, isin = res["symbol"], res["isin"]
    L.append("=" * 100)
    L.append(f"FLOWSENSE RISK ENGINE — {sym}  ({isin})   horizon h={res.get('horizon','?')}")
    L.append("=" * 100)
    for n in res["notes"]:
        L.append(f"  NOTE: {n}")
    if res["n"] == 0:
        L.append("  no scored days in this window.")
        return "\n".join(L)
    t = res["table"]
    L.append(f"  window {res['window'][0]} to {res['window'][1]}  |  "
             f"{res['n']} FII-active scored days")
    p = res["params_last"]
    L.append("")
    L.append(f"  HMM PARAMETERS IN EFFECT AT THE END OF THE WINDOW "
             f"(vintage {p['vintage_id']}, asof {p['asof']})")
    L.append(f"    fitted on {p['window_start']} to {p['asof']} — "
             f"{p['n_fit_rows']:,} stock-days, {p['n_stocks']} stocks")
    L.append(f"    self-transitions  SELL {p['self_transitions'][0]:.3f}  "
             f"NEUT {p['self_transitions'][1]:.3f}  "
             f"BUY {p['self_transitions'][2]:.3f}")
    L.append(f"    F_persist means   SELL {p['means_F_persist'][0]:+.3f}  "
             f"NEUT {p['means_F_persist'][1]:+.3f}  "
             f"BUY {p['means_F_persist'][2]:+.3f}")
    L.append(f"    thresholds        HOSTAGE<{p['th_hostage']:+.3f}  "
             f"SHARK_DIST>{p['th_shark_dist']:+.3f}  "
             f"DISPERSED_ACC<{p['th_dispersed_acc']:+.3f}  "
             f"SHARK_ACC>{p['th_shark_acc']:+.3f}")

    L.append("")
    L.append("  DAILY RISK  (VaR/ES as % of price; 'real' is the realised "
             "forward return; * marks a breach)")
    L.append(f"    {'date':>11}{'vint':>6}{'age':>5}{'archetype':>15}"
             f"{'pa':>6}{'FS VaR1%':>10}{'FS ES1%':>9}{'N VaR1%':>9}"
             f"{'N ES1%':>9}{'real %':>9}  brch")
    rows = t.to_dicts()
    show = rows if len(rows) <= max_rows else (
        rows[:max_rows // 2] + [None] + rows[-max_rows // 2:])
    for r in show:
        if r is None:
            L.append(f"    {'...':>11}   ({len(rows) - max_rows} rows hidden)")
            continue
        rp = r["realised_pct"]
        bf = "FS" if r["fs_breach01"] else "  "
        bn = "N" if r["nm_breach01"] else " "
        L.append(f"    {str(r['TR_DATE']):>11}{r['vintage_id']:>6}"
                 f"{r['param_age_days']:>5}{r['arch_soft'][:14]:>15}"
                 f"{r['pa_max']:>6.2f}{r['fs_var01_pct']:>10.2f}"
                 f"{r['fs_es01_pct']:>9.2f}{r['nm_var01_pct']:>9.2f}"
                 f"{r['nm_es01_pct']:>9.2f}"
                 f"{(f'{rp:.2f}' if np.isfinite(rp) else 'pending'):>9}"
                 f"  {bf}{bn}")

    v = res["validation"]
    L.append("")
    L.append("  VALIDATION ON THIS WINDOW")
    if v.get("warning"):
        L.append(f"    ! {v['warning']}")
    if v["n_with_outcome"] == 0:
        return "\n".join(L)
    L.append(f"    matured observations: {v['n_with_outcome']} of "
             f"{v['n_scored']} scored")
    L.append(f"    {'engine':11s}{'breach5%':>10}{'breach1%':>10}"
             f"{'Kupiec p(1%)':>14}{'ES1% pred':>11}{'ES1% real':>11}"
             f"{'ES cover':>10}{'CRPS':>9}")
    for eng in ("flowsense", "normal"):
        b5 = v.get(f"{eng}_breach05", np.nan)
        b1 = v.get(f"{eng}_breach01", np.nan)
        kp = v.get(f"{eng}_kupiec_p01", np.nan)
        ep = v.get(f"{eng}_es01_pred", np.nan)
        er = v.get(f"{eng}_es01_real", np.nan)
        ec = v.get(f"{eng}_es01_coverage", np.nan)
        cr = v.get(f"{eng}_crps", np.nan)

        def g(x, n=4):
            return f"{x:.{n}f}" if x is not None and np.isfinite(x) else "n/a"
        L.append(f"    {eng:11s}{g(b5):>10}{g(b1):>10}{g(kp):>14}"
                 f"{g(ep,3):>11}{g(er,3):>11}{g(ec,3):>10}{g(cr,5):>9}")
    L.append(f"    targets    {0.05:>10.4f}{0.01:>10.4f}{'>0.05':>14}"
             f"{'':>11}{'':>11}{1.0:>10.3f}")
    if np.isfinite(v.get("crps_gain_pct", np.nan)):
        L.append(f"    CRPS improvement of Flowsense over EWMA-Normal: "
                 f"{v['crps_gain_pct']:+.3f}%")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(
        description="Flowsense walk-forward risk engine (Phase III)")
    ap.add_argument("--stock", required=True, help="NSE symbol or ISIN")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--horizon", type=int, default=1, choices=HORIZONS)
    ap.add_argument("--csv", default=None, help="write the daily table here")
    ap.add_argument("--rows", type=int, default=14)
    a = ap.parse_args()

    try:
        eng = RiskEngine()
    except FileNotFoundError as e:
        raise SystemExit(f"missing pipeline output: {e}. Run C1-C5 first.")
    if eng.verify_ok is not None:
        k = eng.verify_ok
        print(f"[K1] engine reproduces the stored C5 mixture on {k['n']} "
              f"probe rows: max VaR err {k['max_var_err']:.2e}, "
              f"max ES err {k['max_es_err']:.2e}  "
              f"{'PASS' if k['pass'] else 'FAIL'}")
        if not k["pass"]:
            raise SystemExit("engine is out of step with the validated pipeline")
    try:
        res = eng.query(a.stock, a.start, a.end, a.horizon)
    except (KeyError, ValueError) as e:
        raise SystemExit(str(e).strip('"'))
    print(render(res, a.rows))
    if a.csv and res["table"] is not None:
        res["table"].write_csv(a.csv)
        print(f"\nwrote {a.csv}")


if __name__ == "__main__":
    main()
