# [migrated from Colab: paths now come from fii.paths; see
#  scripts/migrate_colab_modules.py — research logic unchanged]
from fii.paths import VALIDATION_DATA, ISIN_MAPPING  # noqa: E402
# ============================================================================
# MODULE 12D · SIGNAL vs IMPLEMENTATION — read-only diagnosis of 12B/12C
#
# 12C verdict (net of 15bps) is FROZEN: S1H inconclusive, S2H/S3H subtract.
# This module asks WHY, from the stored daily series only (no backtests
# re-run, nothing re-tuned): is there incremental GROSS signal that costs
# destroy, or is there no incremental signal at all?
#
# Inputs: bt12_baselines.parquet + bt12_hmm.parquet (pnl_gross, turnover).
#
# PRE-REGISTERED INTERPRETATION (TEST era decides, stated before results):
#   gross dSharpe 95% CI > 0  -> "SIGNAL REAL, COSTS BIND"
#       (HMM adds information; implementation/turnover is the constraint)
#   gross dSharpe 95% CI < 0  -> "NO INCREMENTAL SIGNAL"
#       (overlay adds nothing beyond the mechanical baseline)
#   else                      -> "INCONCLUSIVE"
# Also reported (context, not verdicts): each book's own gross Sharpe CI,
# breakeven one-way cost (bps) = mean gross pnl / mean turnover,
# and net Sharpe at 0/5/10/15/30 bps.
# ============================================================================
import numpy as np
import polars as pl
from pathlib import Path

DRIVE = VALIDATION_DATA
base = pl.read_parquet(DRIVE / "bt12_baselines.parquet")
hmm = pl.read_parquet(DRIVE / "bt12_hmm.parquet")
PAIRS = [("S1_REV20", "S1H_gated_reversal"),
         ("S2_FLOW10", "S2H_filtered_flow"),
         ("S3_PROXY", "S3H_regime_book")]
COSTS = [0, 5, 10, 15, 30]


def series(df, strat, era):
    s = (df.filter((pl.col("strat") == strat) & (pl.col("era") == era))
           .sort("date"))
    return s["pnl_gross"].to_numpy(), s["turnover"].to_numpy()


def sharpe(x):
    sd = x.std(ddof=1)
    return x.mean() / sd * np.sqrt(252) if sd > 0 else np.nan


def boot(fn, args, block=20, B=2000, seed=11):
    """95% CI of fn over jointly block-resampled daily arrays."""
    rng = np.random.default_rng(seed)
    n = len(args[0])
    k = int(np.ceil(n / block))
    vals = np.empty(B)
    for b in range(B):
        idx = np.concatenate([np.arange(s, s + block) % n for s in
                              rng.integers(0, n, k)])[:n]
        vals[b] = fn(*[a[idx] for a in args])
    return np.percentile(vals, [2.5, 97.5])

print("=" * 70)
print("MODULE 12D · gross-signal diagnosis (read-only)")
print("=" * 70)
for bn, hn in PAIRS:
    print(f"\n{hn}  vs  {bn}")
    for era in ("TRAIN", "TEST"):
        bg, bt = series(base, bn, era)
        hg, ht = series(hmm, hn, era)
        n = min(len(bg), len(hg))
        bg, bt, hg, ht = bg[:n], bt[:n], hg[:n], ht[:n]
        sb, sh = sharpe(bg), sharpe(hg)
        lo, hi = boot(lambda a, b: sharpe(a) - sharpe(b), (hg, bg))
        slo, shi = boot(lambda a: sharpe(a), (hg,))
        beb = 1e4 * bg.mean() / max(bt.mean(), 1e-9)
        beh = 1e4 * hg.mean() / max(ht.mean(), 1e-9)
        print(f"  {era:5s} GROSS Sharpe: base {sb:+.2f}  HMM {sh:+.2f} "
              f"(own CI [{slo:+.2f},{shi:+.2f}])")
        print(f"        gross dSharpe 95% CI [{lo:+.2f}, {hi:+.2f}]"
              + ("   << DECISIVE" if era == "TEST" else ""))
        print(f"        breakeven cost: base {beb:+.1f}bps  "
              f"HMM {beh:+.1f}bps  (one-way; >15 = survives our toll)")
        row = "        net Sharpe by cost:"
        for c in COSTS:
            hnet = hg - c * 1e-4 * ht
            row += f"  {c}bps {sharpe(hnet):+.2f}"
        print(row + "   (HMM book)")
        if era == "TEST":
            verdict = ("SIGNAL REAL, COSTS BIND" if lo > 0 else
                       "NO INCREMENTAL SIGNAL" if hi < 0 else
                       "INCONCLUSIVE")
            print(f"        TEST VERDICT: {verdict}")
print("\n" + "=" * 70)
print("Reminder: 12C net-of-15bps verdicts remain the frozen headline.")
print("This module only separates 'no signal' from 'costs bind'.")
print("=" * 70)
