"""
MODULE C3 · SOFT-THRESHOLD ARCHETYPE OVERLAY  —  Phase III

Turns the three state probabilities from C2 into a 7-column archetype
probability matrix, using the walk-forward thresholds and their bootstrap
standard errors from C1.

NAMING (AUDIT ITEM 7).  These are WEIGHTS, not calibrated probabilities, and
the output columns say so: `w_<ARCH>`, `max_archetype_weight`,
`archetype_weight_entropy`. The `pa_*` / `pa_max` / `arch_entropy` names are
retained as aliases for backward compatibility ONLY and are deprecated.

Two measurements force the distinction, both made on the pre-audit build:

  * IDENTIFIABILITY. The seven archetype outcome densities differ by a median
    pairwise KS distance of 0.012 (max 0.021 at h=1). Mixture weights on
    near-identical components are not recoverable from outcomes, so NO
    outcome-based test can validate or falsify them. This is also the
    mechanism behind C6's finding that conditioning adds nothing measurable.
  * MAGNITUDE. Median |pa_max - p_max| is 0.0000, and so is the 90th
    percentile: on ~80% of rows the soft weight equals the hard call exactly.
    The threshold-uncertainty channel moves mass on about 20% of rows.

Calling these probabilities would assert a frequency interpretation the design
cannot test. They are the weights the mixture actually uses -- no more.

WHAT THE WEIGHT MEANS.  The threshold theta is ESTIMATED, not known.
C1 produced theta_hat with a bootstrap standard error sd. The feature F is
OBSERVED without error. So for a rule of the form "tagged if F < theta",

    P(tagged) = P(theta > F) = Phi((theta_hat - F) / sd)

is the probability that this day's observed F lies on the tagged side of the
TRUE threshold, given our uncertainty about where that threshold is. It is
estimation uncertainty in the boundary, not measurement noise in F. That
distinction matters if a referee asks, and it is why F never appears inside a
variance term below.

TAXONOMY (7 exhaustive, mutually exclusive archetypes)

    SELL    HOSTAGE        F_entity_s     < th_hostage       (dispersed sell)
            SELL_MID       between
            SHARK_DIST     F_entity_s     > th_shark_dist    (concentrated)
    NEUTRAL ROBOT          (no overlay)
    BUY     DISPERSED_ACC  F_entity_buy_s < th_dispersed_acc (dispersed buy)
            BUY_MID        between
            SHARK_ACC      F_entity_buy_s > th_shark_acc     (concentrated)

    w(archetype) = P(state) * P(band | state)      -> sums to 1 by construction

    The product is a coherent weight on the simplex. It is not a calibrated
    probability of the archetype being "true": there is no observable
    archetype to score against, and it inherits any miscalibration in the
    state posterior, which is itself unscoreable (the regime is latent).

TERMINAL BY DESIGN.  This overlay is NOT fed back into the C2 recursion.
Threshold uncertainty is a property of the labelling boundary, not of the
state process; injecting it into the forward filter would corrupt a posterior
that is currently exact given its parameters. C3 reads C2 and stops.

CAUSALITY.  Day t uses the thresholds of the same vintage that scored it in
C2 (carried in `vintage_id`), so causality is inherited. Re-asserted as H0.

Gates
    H0  causality      — every row's vintage asof is strictly before its date
    H1  simplex        — the 7 probabilities sum to 1
    H2  degenerate     — as sd -> 0 the soft matrix must collapse EXACTLY to
                         the hard rule applied with the same vintage thresholds
    H3  band overlap   — raw within-state tail probabilities must not exceed 1

Input : outputs/phase3/daily_posteriors.parquet
        outputs/phase3/vintages.parquet
Output: outputs/phase3/archetype_probs.parquet
"""
from __future__ import annotations

import time

import numpy as np
import polars as pl
from scipy.stats import norm

from fii.paths import OUTPUTS
from fii.lineage import write_lineage  # audit item 12
# ---- truncation hook (used only by module_c9_lookahead_audit) --------------
# P3_TRUNC=<ISO date> makes the module behave as if today were that date:
# every dated input is cut there and every vintage fitted after it is dropped.
# P3_SUFFIX renames the outputs so a truncated run never overwrites the real
# one. With both unset this is a no-op and the module is unchanged.
import os as _os, datetime as _dt
_TRUNC = _os.environ.get("P3_TRUNC")
_TRUNC = _dt.date.fromisoformat(_TRUNC) if _TRUNC else None
_SUF = _os.environ.get("P3_SUFFIX", "")


def _cut(df, col):
    return df.filter(pl.col(col) <= _TRUNC) if _TRUNC is not None else df


ARCH = ["HOSTAGE", "SELL_MID", "SHARK_DIST", "ROBOT",
        "DISPERSED_ACC", "BUY_MID", "SHARK_ACC"]
OUT = OUTPUTS / "phase3" / f"archetype_probs{_SUF}.parquet"

t_start = time.time()

d = pl.read_parquet(OUTPUTS / "phase3" / f"daily_posteriors{_SUF}.parquet")
vt = _cut(pl.read_parquet(OUTPUTS / "phase3" / "vintages.parquet")
          .sort("asof"), "asof")
print(f"C2 posteriors {d.height:,} rows | vintages {vt.height}")

v = d["vintage_id"].to_numpy()

# ---- H0 · causality inherited from C2 --------------------------------------
asof = vt["asof"].to_numpy()
dates = d["TR_DATE"].to_numpy()
viol = int((asof[v] >= dates).sum())
print(f"\n[H0] causality — rows whose threshold vintage is not strictly before "
      f"the scored date: {viol}  {'PASS' if viol == 0 else 'FAIL'}")
assert viol == 0

# ---- per-row thresholds and their bootstrap SEs -----------------------------
TH = {c: vt[c].to_numpy()[v] for c in
      ("th_hostage", "th_shark_dist", "th_dispersed_acc", "th_shark_acc")}
SD = {c: vt[c + "_sd"].to_numpy()[v] for c in
      ("th_hostage", "th_shark_dist", "th_dispersed_acc", "th_shark_acc")}

Fs = d["F_entity_s"].to_numpy()          # sell-side concentration
Fb = d["F_entity_buy_s"].to_numpy()      # buy-side concentration
p_sell = d["p_sell"].to_numpy()
p_neut = d["p_neutral"].to_numpy()
p_buy = d["p_buy"].to_numpy()


def bands(F, th_lo, sd_lo, th_hi, sd_hi):
    """P(F < theta_lo), P(F > theta_hi), and the middle, all given
    uncertainty in the two thresholds. Returns (lo, mid, hi)."""
    lo = norm.cdf((th_lo - F) / sd_lo)     # P(theta_lo > F)
    hi = norm.cdf((F - th_hi) / sd_hi)     # P(theta_hi < F)
    return lo, 1.0 - lo - hi, hi


ho, sell_mid, sd_ = bands(Fs, TH["th_hostage"], SD["th_hostage"],
                          TH["th_shark_dist"], SD["th_shark_dist"])
da, buy_mid, sa = bands(Fb, TH["th_dispersed_acc"], SD["th_dispersed_acc"],
                        TH["th_shark_acc"], SD["th_shark_acc"])

# ---- H3 · band overlap ------------------------------------------------------
ov_s, ov_b = (ho + sd_) - 1.0, (da + sa) - 1.0
n_ov = int((ov_s > 1e-12).sum() + (ov_b > 1e-12).sum())
print(f"[H3] band overlap — rows where the two tail probabilities exceed 1: "
      f"{n_ov}  {'PASS' if n_ov == 0 else 'FAIL'}")
print(f"     worst sell-side excess {ov_s.max():+.3e} | "
      f"buy-side {ov_b.max():+.3e}")
print(f"     (threshold separation ~1.39 vs SE ~0.027, so no collision "
      f"is expected)")
assert n_ov == 0, "thresholds overlap: the middle band would be negative"

P = np.column_stack([p_sell * ho, p_sell * sell_mid, p_sell * sd_,
                     p_neut,
                     p_buy * da, p_buy * buy_mid, p_buy * sa])

# ---- H1 · simplex -----------------------------------------------------------
s1 = np.abs(P.sum(axis=1) - 1.0).max()
print(f"[H1] simplex max|sum-1| {s1:.2e}  {'PASS' if s1 < 1e-12 else 'FAIL'} "
      f"| min {P.min():.2e}")
assert s1 < 1e-12 and P.min() >= 0.0

# ---- H2 · degenerate-threshold limit ---------------------------------------
# With sd -> 0 the Gaussian CDF becomes a step function and the soft matrix
# must reproduce the hard rule exactly. This is the correctness gate: it
# checks the band algebra, the state multiplication and the taxonomy ordering
# all at once.
EPS = 1e-12
ho0, sm0, sd0 = bands(Fs, TH["th_hostage"], EPS, TH["th_shark_dist"], EPS)
da0, bm0, sa0 = bands(Fb, TH["th_dispersed_acc"], EPS, TH["th_shark_acc"], EPS)
hard_state = np.column_stack([p_sell, p_neut, p_buy]).argmax(axis=1)
P0 = np.column_stack([ho0, sm0, sd0, np.ones_like(ho0), da0, bm0, sa0])
P0[:, :3] *= (hard_state == 0)[:, None]
P0[:, 3] *= (hard_state == 1)
P0[:, 4:] *= (hard_state == 2)[:, None]
soft_deg = P0.argmax(axis=1)

hard = np.where(
    hard_state == 1, 3,
    np.where(hard_state == 0,
             np.where(Fs < TH["th_hostage"], 0,
                      np.where(Fs > TH["th_shark_dist"], 2, 1)),
             np.where(Fb < TH["th_dispersed_acc"], 4,
                      np.where(Fb > TH["th_shark_acc"], 6, 5))))
# Exact ties are real and expected: C1's thresholds are empirical quantiles of
# observed F, so a threshold IS one of the data points. At F == theta the
# degenerate limit gives Phi(0) = 0.5 on each side — a genuine coin flip —
# while the hard rule's strict `<` forces it into the middle band. The gate
# therefore demands exactness off the ties and reports the ties separately.
tie = ((Fs == TH["th_hostage"]) | (Fs == TH["th_shark_dist"])
       | (Fb == TH["th_dispersed_acc"]) | (Fb == TH["th_shark_acc"]))
match = float((soft_deg[~tie] == hard[~tie]).mean())
print(f"[H2] degenerate limit (sd -> 0) reproduces the hard rule on the "
      f"{int((~tie).sum()):,} non-tie rows: {match:.6%}  "
      f"{'PASS' if match == 1.0 else 'FAIL'}")
print(f"     exact ties (F == threshold to the bit): {int(tie.sum())} "
      f"({tie.mean():.5%}) — soft gives 0.5/0.5, hard rule's strict '<' "
      f"sends them to the middle band")
assert match == 1.0

# ---- assemble ---------------------------------------------------------------
out = (d.select(["cisin", "TR_DATE", "era", "vintage_id", "param_age_days",
                 "p_max", "obs_age", "stock_out_of_window",
                 "F_entity_s", "F_entity_buy_s"])
        # AUDIT ITEM 7 — primary names say "weight", not "probability".
        .with_columns([pl.Series(f"w_{a}", P[:, i])
                       for i, a in enumerate(ARCH)])
        .with_columns(
            pl.Series("arch_soft", [ARCH[i] for i in P.argmax(axis=1)]),
            pl.Series("arch_hard", [ARCH[i] for i in hard]),
            pl.Series("max_archetype_weight", P.max(axis=1)),
            pl.Series("archetype_weight_entropy",
                      -(P * np.log(P + 1e-300)).sum(axis=1)))
        # DEPRECATED aliases, written so C4/C5/C7/C8 and any stored analysis
        # keep working. Read the names above; these will be dropped once every
        # consumer has moved.
        .with_columns([pl.col(f"w_{a}").alias(f"pa_{a}") for a in ARCH])
        .with_columns(
            pl.col("max_archetype_weight").alias("pa_max"),
            pl.col("archetype_weight_entropy").alias("arch_entropy")))
OUT.parent.mkdir(parents=True, exist_ok=True)
out.write_parquet(OUT)
write_lineage(OUT, inputs=[OUTPUTS / "phase3" / f"daily_posteriors{_SUF}.parquet",
                          OUTPUTS / "phase3" / "vintages.parquet"],
              extras={"module": "C3", "suffix": _SUF, "rows": out.height})

# ---- report -----------------------------------------------------------------
print("\n" + "=" * 72)
print("C3 SUMMARY — expected counts (soft) vs argmax counts (hard)")
print(f"{'archetype':16s}{'E[n] soft':>12s}{'share':>9s}"
      f"{'n hard':>10s}{'share':>9s}")
n = out.height
for i, a in enumerate(ARCH):
    es = P[:, i].sum()
    nh = int((hard == i).sum())
    print(f"  {a:14s}{es:12,.0f}{es / n:9.2%}{nh:10,}{nh / n:9.2%}")
print(f"  {'TOTAL':14s}{P.sum():12,.0f}{1.0:9.2%}{n:10,}{1.0:9.2%}")

agree = float((out["arch_soft"] == out["arch_hard"]).mean())
print(f"\n  soft argmax vs hard rule agreement: {agree:.2%}")
print(f"  mean max archetype WEIGHT:          "
      f"{float(out['max_archetype_weight'].mean()):.3f}   "
      f"(a weight, not a calibrated probability — see item 7 in the header)")
print(f"  mean archetype weight entropy:      "
      f"{float(out['archetype_weight_entropy'].mean()):.3f} nats "
      f"(max {np.log(7):.3f})")

print("\n  where does archetype uncertainty come from?")
pm = out["p_max"].to_numpy()
pam = out["pa_max"].to_numpy()
print(f"    state-certain rows (p_max>0.90), archetype-uncertain (pa_max<0.70):"
      f" {int(((pm > .90) & (pam < .70)).sum()):,}")
print(f"    state-uncertain rows (p_max<0.70): {int((pm < .70).sum()):,}, of "
      f"which archetype-uncertain: {int(((pm < .70) & (pam < .70)).sum()):,}")
print(f"    ANY archetype uncertainty (pa_max<0.70): "
      f"{int((pam < .70).sum()):,} ({(pam < .70).mean():.2%})")

print("\n  the two archetypes Phase I never had:")
for a in ("SELL_MID", "BUY_MID", "DISPERSED_ACC"):
    i = ARCH.index(a)
    print(f"    {a:14s} E[n] {P[:, i].sum():10,.0f}  "
          f"({P[:, i].sum() / n:6.2%})")

print(f"\n  wrote {OUT}")
print(f"  total {time.time() - t_start:.1f}s")
print("=" * 72)
