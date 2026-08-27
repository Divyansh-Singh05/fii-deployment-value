"""
MODULE C9 · LOOK-AHEAD AUDIT  —  Phase III

The decisive test that the risk engine contains no look-ahead bias.

THE PROBLEM WITH THE GATES SO FAR.  C2's G2 showed that randomising the
future leaves past posteriors unchanged, and C4's I0 showed the density
embargo holds. Both are single-module checks. Neither rules out a leak that
enters through a quantity computed ONCE over the whole sample and then used
at every date — a full-sample calendar, a full-sample stock universe, a
warm-up frontier chosen by scanning all vintages. Those are invisible to a
per-module gate because no module ever "looks forward"; the leak is baked
into a constant.

THE TEST.  Truncate the world at a date T and re-run the ACTUAL production
modules (C2 -> C3 -> C4 -> C5) against inputs that stop at T, then compare
every row dated on or before T against the full-sample run.

    If any full-sample quantity influences a pre-T row, the two runs differ.
    If the pipeline is genuinely causal, they are bit-identical.

This is stronger than any per-module assertion because it tests the composed
system, including every constant, and it makes no assumption about where a
leak might be.

WHAT IS ALLOWED TO DIFFER.  Rows in the final h trading days before T have no
matured outcome in the truncated world, so their CRPS and PIT are NaN there
while the full run has values. That is correct behaviour, not leakage — the
outcome genuinely had not happened yet at T. VaR and ES need no outcome and
must match on every row without exception.

Gates
    L1  posteriors   p_sell/p_neutral/p_buy identical on every pre-T row
    L2  archetypes   all seven archetype probabilities identical
    L3  risk         VaR and ES identical at every level and horizon
    L4  scores       CRPS and PIT identical wherever both runs have an outcome
    L5  coverage     the truncated run scores exactly the pre-T rows it should

Run:  python -m fii.phase3.module_c9_lookahead_audit [--asof 2021-06-30]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

from fii.paths import OUTPUTS

P3 = OUTPUTS / "phase3"
STAGES = ["fii.phase3.module_c2_daily_filter", "fii.phase3.module_c3_archetype_probs",
          "fii.phase3.module_c4_outcome_densities", "fii.phase3.module_c5_predictive_mixture"]
HORIZONS = [1, 5, 20]
SUF = "_trunc"


def run_truncated(asof: str, rerun: bool = True):
    if not rerun:
        return
    env = dict(os.environ, P3_TRUNC=asof, P3_SUFFIX=SUF,
               PYTHONPATH=str(Path(__file__).resolve().parents[2]))
    for m in STAGES:
        t = time.time()
        r = subprocess.run([sys.executable, "-m", m],
                           env=env, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-3000:])
            print(r.stderr[-3000:])
            raise SystemExit(f"truncated run failed at {m}")
        print(f"  {m:34s} ok  ({time.time() - t:.0f}s)")


def cmp_frames(full, trunc, keys, cols, label):
    """Inner-join on keys and return the worst disagreement AND every change
    in availability.

    Comparing only where BOTH runs are finite is a blind spot: a value that
    goes finite -> NaN (or NaN -> finite) between the two runs is a change in
    forecast ELIGIBILITY, and it would sail through a numeric-equality gate
    that skips non-finite cells. Both directions are counted and returned:

        only_full   finite in the full run, NaN in the truncated run
        only_trunc  NaN in the full run, finite in the truncated run

    `only_trunc` is ALWAYS a failure — a world that stops earlier cannot know
    more. `only_full` is a failure everywhere except the CRPS/PIT gate, where
    it is the expected unmatured tail and is checked separately against the
    last h trading days.
    """
    j = full.select(keys + cols).join(
        trunc.select(keys + cols), on=keys, how="inner", suffix="__t")
    worst, worst_col, n_bad = 0.0, None, 0
    only_full, only_trunc = 0, 0
    of_mask = np.zeros(j.height, dtype=bool)
    for c in cols:
        a = j[c].to_numpy().astype(float)
        b = j[f"{c}__t"].to_numpy().astype(float)
        fa, fb = np.isfinite(a), np.isfinite(b)
        only_full += int((fa & ~fb).sum())
        only_trunc += int((~fa & fb).sum())
        of_mask |= (fa & ~fb)
        both = fa & fb
        if not both.any():
            continue
        dmax = float(np.abs(a[both] - b[both]).max())
        n_bad += int((a[both] != b[both]).sum())
        if dmax > worst:
            worst, worst_col = dmax, c
    return dict(n=j.height, worst=worst, col=worst_col, n_bad=n_bad,
                only_full=only_full, only_trunc=only_trunc,
                dates=j["TR_DATE"].to_numpy(), of_mask=of_mask)


def verdict(r, allow_only_full=False):
    """A gate passes only if the values match AND availability is unchanged."""
    return (r["worst"] == 0.0 and r["only_trunc"] == 0
            and (allow_only_full or r["only_full"] == 0))


def fmt(tag, name, r, allow_only_full=False):
    ok = verdict(r, allow_only_full)
    return (f"[{tag}] {name} — {r['n']:,} rows, max|diff| {r['worst']:.3e} "
            f"({r['col']}), rows differing {r['n_bad']}, "
            f"availability full-only {r['only_full']:,} / trunc-only "
            f"{r['only_trunc']:,}  {'PASS' if ok else 'FAIL'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asof", default="2021-06-30",
                    help="truncate the world here")
    ap.add_argument("--no-rerun", action="store_true",
                    help="reuse an existing truncated run")
    # C1 is audited BY DEFAULT. It used to be opt-in behind --with-c1, and
    # that path crashed (it unpacked four values from a five-value
    # window_rows), so in practice the fitting stage was never audited and
    # the headline verdict silently covered only C2-C5.
    ap.add_argument("--skip-c1", action="store_true",
                    help="skip the C1 fitting-stage gate (L0). The verdict is "
                         "then explicitly scoped to the downstream chain.")
    a = ap.parse_args()
    T = np.datetime64(a.asof)
    with_c1, a_asof = (not a.skip_c1), a.asof

    print("=" * 78)
    print(f"LOOK-AHEAD AUDIT — truncating the world at {a.asof}")
    print("=" * 78)
    print("\nre-running the production pipeline against a world that stops "
          f"at {a.asof}:")
    run_truncated(a.asof, rerun=not a.no_rerun)

    dp_f = pl.read_parquet(P3 / "daily_posteriors.parquet")
    dp_t = pl.read_parquet(P3 / f"daily_posteriors{SUF}.parquet")
    ap_f = pl.read_parquet(P3 / "archetype_probs.parquet")
    ap_t = pl.read_parquet(P3 / f"archetype_probs{SUF}.parquet")
    pr_f = pl.read_parquet(P3 / "predictive.parquet")
    pr_t = pl.read_parquet(P3 / f"predictive{SUF}.parquet")

    keys = ["cisin", "TR_DATE"]
    pre = pl.col("TR_DATE") <= np.datetime64(a.asof).astype("datetime64[D]").item()
    dp_f2, ap_f2, pr_f2 = dp_f.filter(pre), ap_f.filter(pre), pr_f.filter(pre)

    print(f"\nrows on or before {a.asof}:")
    print(f"  full-sample run : {pr_f2.height:,}")
    print(f"  truncated run   : {pr_t.height:,}")

    ok = True

    # ---- L0 · C1 refitting ---------------------------------------------------
    # C2-C5 consume vintages.parquet as given, so the audit above never tested
    # the FITTING stage. L0 closes that: for a sample of vintages, check the
    # data the fit consumes is bitwise identical when the world stops at that
    # vintage's own asof. Parameters are compared separately in C1's own M2b
    # against a measured non-determinism floor, because EM on this panel is not
    # bitwise reproducible (BLAS reduction order, ~3e-09 in log-likelihood).
    if with_c1:
        from fii.phase3.module_c1_refit_harness import load_states, window_rows, FEATS
        vt_all = pl.read_parquet(P3 / "vintages.parquet").sort("asof")
        cand = [i for i in range(vt_all.height)
                if vt_all["asof"][i] <= np.datetime64(a_asof).astype(
                    "datetime64[D]").item()]
        probe = cand[:: max(1, len(cand) // 6)][:6]
        st_full = load_states()
        l0 = True
        print(f"\n[L0] C1 refit inputs — {len(probe)} vintages probed of "
              f"{len(cand)} available at T")
        for vid in probe:
            asof_v = vt_all["asof"][vid]
            # window_rows returns five values since the gap-aware refit
            # (it now also yields the elapsed trading-day gaps). Unpacking
            # four raised ValueError, so this audit could not run at all.
            loF, nsF, nrF, wF, gF = window_rows(st_full, asof_v)
            loT, nsT, nrT, wT, gT = window_rows(load_states(asof_v), asof_v)
            XF, XT = wF.select(FEATS).to_numpy(), wT.select(FEATS).to_numpy()
            LF = wF.group_by("cisin", maintain_order=True).len()["len"].to_numpy()
            LT = wT.group_by("cisin", maintain_order=True).len()["len"].to_numpy()
            # The gaps are a first-class input, not bookkeeping: the estimator
            # links observations k trading days apart by A^k, so a gap vector
            # that differed between the two worlds would change the fitted
            # transition matrix while X and lengths matched. They were being
            # unpacked and then ignored.
            same = (np.array_equal(XF, XT) and np.array_equal(LF, LT)
                    and np.array_equal(gF, gT)
                    and wF["cisin"].to_list() == wT["cisin"].to_list()
                    and (loF, nsF, nrF) == (loT, nsT, nrT))
            l0 &= same
            print(f"     v{vid:<3} asof {asof_v}  {wF.height:>7} rows  "
                  f"{len(gF):>7} gaps  "
                  f"bitwise identical: {'YES' if same else 'NO'}")
        print(f"     {'PASS' if l0 else 'FAIL'}")
        ok &= l0

    # ---- L5 · coverage ------------------------------------------------------
    kf = set(zip(dp_f2["cisin"].to_list(), dp_f2["TR_DATE"].to_list()))
    kt = set(zip(dp_t["cisin"].to_list(), dp_t["TR_DATE"].to_list()))
    only_f, only_t = kf - kt, kt - kf
    print(f"\n[L5] coverage — scored rows present in one run only: "
          f"full-only {len(only_f):,}, truncated-only {len(only_t):,}  "
          f"{'PASS' if not only_f and not only_t else 'FAIL'}")
    ok &= not only_f and not only_t
    # L5 checks that the ROWS match. Whether each row actually carries a SCORE
    # is a separate question, and is what the availability counters in L1-L4
    # below answer.

    # ---- L1 · posteriors ----------------------------------------------------
    r = cmp_frames(dp_f2, dp_t, keys,
                   ["p_sell", "p_neutral", "p_buy"], "posteriors")
    print(fmt("L1", "filtered posteriors", r))
    ok &= verdict(r)

    # ---- L2 · archetype probabilities --------------------------------------
    acols = [c for c in ap_f.columns if c.startswith("pa_") and c != "pa_max"]
    r = cmp_frames(ap_f2, ap_t, keys, acols + ["pa_max"], "archetypes")
    print(fmt("L2", "archetype probabilities", r))
    ok &= verdict(r)

    # ---- L3 · VaR and ES ----------------------------------------------------
    rcols = [f"soft_roll_h{h}_{m}{lv}" for h in HORIZONS
             for m in ("var", "es") for lv in ("01", "05")]
    rcols = [c for c in rcols if c in pr_f.columns]
    r = cmp_frames(pr_f2, pr_t, keys, rcols, "risk")
    print(fmt("L3", f"VaR and ES ({len(rcols)} cols)", r))
    ok &= verdict(r)

    # ---- L4 · CRPS and PIT --------------------------------------------------
    scols = [f"{s}_h{h}_{m}" for h in HORIZONS
             for s in ("soft_roll", "hard_roll", "clim_roll", "normal")
             for m in ("crps", "pit")]
    scols = [c for c in scols if c in pr_f.columns]
    r = cmp_frames(pr_f2, pr_t, keys, scols, "scores")
    print(fmt("L4", f"CRPS and PIT ({len(scols)} cols)", r, allow_only_full=True))
    ok &= verdict(r, allow_only_full=True)

    # L4b · the tolerated full-only cells must lie inside the unmatured tail.
    # Anything earlier would be a genuine eligibility change dressed up as a
    # maturity effect. The calendar is the distinct pre-T dates themselves.
    cal = np.sort(np.unique(r["dates"]))
    edge = set(cal[-max(HORIZONS):].tolist())
    bad_dates = sorted({d for d, f in zip(r["dates"].tolist(), r["of_mask"])
                        if f and d not in edge})
    print(f"[L4b] tolerated full-only cells confined to the last "
          f"{max(HORIZONS)} trading days before T: "
          f"{len(bad_dates)} earlier dates affected  "
          f"{'PASS' if not bad_dates else 'FAIL'}")
    if bad_dates:
        print(f"      first offenders: {bad_dates[:5]}")
    ok &= not bad_dates

    # ---- expected, benign difference: unmatured tail -----------------------
    print("\n  expected NaN-only differences near the truncation edge "
          "(outcomes that had not happened yet at T):")
    j = pr_f2.select(keys + [f"z_h{h}" for h in HORIZONS]).join(
        pr_t.select(keys + [f"z_h{h}" for h in HORIZONS]), on=keys,
        how="inner", suffix="__t")
    for h in HORIZONS:
        a_ = np.isfinite(j[f"z_h{h}"].to_numpy())
        b_ = np.isfinite(j[f"z_h{h}__t"].to_numpy())
        print(f"    h={h:>2}: matured in full run {a_.sum():,}, in truncated "
              f"{b_.sum():,}, unmatured at T {int((a_ & ~b_).sum()):,}")

    # ---- vintage-level check ------------------------------------------------
    vt = pl.read_parquet(P3 / "vintages.parquet").sort("asof")
    nv_t = int(pr_t["vintage_id"].max()) + 1
    print(f"\n  vintages available at T: {nv_t} of {vt.height} "
          f"(last asof {vt['asof'][nv_t - 1]})")

    print("\n" + "=" * 78)
    if ok:
        scope = ("C1 fitting inputs, and C2-C5 scoring" if with_c1
                 else "C2-C5 scoring ONLY (C1 gate skipped)")
        print(f"VERDICT: NO LOOK-AHEAD DETECTED IN — {scope}.")
        print("Re-running the pipeline in a world that ends at "
              f"{a.asof} reproduces every pre-{a.asof} posterior, archetype "
              "probability, VaR, ES, CRPS and PIT EXACTLY.")
        print()
        print("SCOPE — what this does NOT cover. The audit rebuilds the")
        print("scoring chain from stored states and vintages. It does not")
        print("rebuild the feature store (s01), the HMM backbone (s02), the")
        print("threshold calibration (s03), or the canonical price panel.")
        print("Causality of those is argued from their construction and from")
        print("the point-in-time identity map, not demonstrated here. Nor can")
        print("this gate see a PUBLICATION lag: it truncates on TR_DATE, so a")
        print("row using flow data that had not yet been reported at the time")
        print("would pass every gate above.")
    else:
        print("VERDICT: LEAKAGE DETECTED — see the failing gate above.")
    print("=" * 78)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
