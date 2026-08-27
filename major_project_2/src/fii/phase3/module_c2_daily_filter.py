"""
MODULE C2 · DAILY VECTORISED FILTER  —  Phase III

Produces a causal filtered posterior P(S_t | x_{1:t}) for every scorable
stock-day, using the walk-forward parameter vintages from C1.

Design decisions (each one is asserted or measured below):

 1. VINTAGE ASSIGNMENT.  Day t is scored with the latest vintage whose
    `asof` is strictly BEFORE t.  A vintage fitted on data through its
    asof date may never score that date.  Enforced by searchsorted(left)
    and re-checked by assertion G0.

 2. GAP-AWARE PROPAGATION.  16A applied the transition matrix once per
    observation ROW regardless of elapsed time.  Measured gap structure
    in states_v3: 4.8% of observations follow a >7d gap, 1.3% follow a
    >30d gap, largest gap 3543 days.  Under A^1 a decade-long gap keeps
    full one-step memory.  Here we propagate with A^k where k is the
    number of TRADING days elapsed, so belief decays to the stationary
    distribution on its own and no arbitrary reset threshold is needed.
    This also handles the 2021-05/06 and 2023-06/09/11 data holes by
    construction rather than by special-casing.

 3. RE-ENTRY PRIOR.  A stock's first scorable observation gets the
    stationary distribution of the in-effect vintage, not `startprob`.
    startprob is an artifact of how sequences were chunked during
    fitting; a stock entering mid-sample is not at "the start of the
    market".  Sensitivity to this choice is measured (it washes out
    within ~9 trading days) and reported.

 4. CROSS-VINTAGE CARRY.  The posterior is carried across a vintage
    boundary unchanged.  This is only legitimate because C1 enforced
    state alignment (0 flips across 106 vintages, median alignment
    margin 32x), so state k means the same thing in every vintage.
    Re-asserted here as G0b.

Gates:
    G0  causality      — no vintage scores a date at or before its asof
    G0b state order    — means[:,0] strictly increasing in every vintage
    G1  cross-check    — independent log-space implementation agrees
    G2  future-invariance — corrupting data after a cut date leaves every
                         posterior at or before the cut bit-identical
    G3  sanity         — simplex, finite, expected coverage

Input : data/VALIDATION_DATA/states_v3.parquet
        outputs/phase3/vintages.parquet
Output: outputs/phase3/daily_posteriors.parquet
"""
from __future__ import annotations

import json
import time

import numpy as np
import polars as pl

from fii.paths import OUTPUTS, VALIDATION_DATA
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
# P3_UNIT_STEP=1 forces one transition per OBSERVED ROW instead of A^k over the
# elapsed trading-day gap. DIAGNOSTIC ONLY, never production.
#
# AUDIT ITEM 3 — this switch no longer describes a live defect. C1 now fits a
# GAP-AWARE likelihood (fii.models.gap_aware_hmm): a transition across k
# trading days is A^k, so the estimated matrix is genuinely PER TRADING DAY and
# this module's A^k propagation reads it in the unit it was estimated in. Both
# sides also share one calendar (fii.phase3.trading_calendar).
#
# Before that fix, A was estimated per observed FII row and reinterpreted here
# as per-trading-day. Measured cost: 77% of mixing half-life (11.5d vs 20.4d on
# k=1 pairs), concentrated in thin stocks and 1.7x over-represented in C6's
# PRIMARY stratum. The switch is kept so that gap can still be reproduced on
# demand; it is not a supported way to run the engine.
_UNIT = _os.environ.get("P3_UNIT_STEP") == "1"


def _cut(df, col):
    return df.filter(pl.col(col) <= _TRUNC) if _TRUNC is not None else df


FEATS = ["F_persist", "F_block", "F_entity_s", "F_entity_buy_s"]
K = 3
KCAP = 250          # A^k beyond this is stationary to <1e-8
SNAME = {0: "SELL_REGIME", 1: "NEUTRAL", 2: "BUY_REGIME"}
OUT = OUTPUTS / "phase3" / f"daily_posteriors{_SUF}.parquet"

t_start = time.time()

# ---------------------------------------------------------------- load ------
st = (pl.read_parquet(VALIDATION_DATA / "states_v3.parquet")
        .sort(["cisin", "TR_DATE"]))
vt = _cut(pl.read_parquet(OUTPUTS / "phase3" / "vintages.parquet")
          .sort("asof"), "asof")

# ---- trading calendar, defined by PRICES not by the FII feed ---------------
# The custodian feed carries rows stamped on non-trading days: 8,256 on
# Saturdays and Sundays plus 910 on market holidays, 97% of them in 2016-17.
# A trade reported on a Saturday has no day-t price, so it can carry no
# forward return, and admitting such a date to the calendar poisons every
# multi-day outcome window that spans it. They are dropped here, at the head
# of Phase III, so C3 and C4 inherit a calendar on which every day prices.
_rp = _cut(pl.read_parquet(VALIDATION_DATA / "returns_panel_v3.parquet"),
           "date")
_td = (_rp.group_by("date").agg(pl.col("ret_adj").is_finite().sum().alias("n"))
          .filter(pl.col("n") >= 50)["date"])
trading_days = set(_td.to_list())
st = _cut(st, "TR_DATE")
_n0 = st.height
st = st.filter(pl.col("TR_DATE").is_in(list(trading_days)))
print(f"dropped {_n0 - st.height:,} rows on non-trading dates "
      f"({(_n0 - st.height) / _n0:.4%}) — weekends and market holidays "
      f"carried by the FII feed")

print(f"states  {st.height:,} rows | {st['cisin'].n_unique()} stocks | "
      f"{st['TR_DATE'].min()} -> {st['TR_DATE'].max()}")
print(f"vintages {vt.height} | {vt['asof'].min()} -> {vt['asof'].max()}")

NV = vt.height
MU = np.zeros((NV, K, len(FEATS)))
SIG2 = np.zeros((NV, K, len(FEATS)))
A = np.zeros((NV, K, K))

for i in range(NV):
    MU[i] = np.asarray(json.loads(vt["means"][i]), dtype=float)
    cv = np.asarray(json.loads(vt["covars"][i]), dtype=float)
    SIG2[i] = np.array([np.diag(c) for c in cv]) if cv.ndim == 3 else cv
    A[i] = np.asarray(json.loads(vt["transmat"][i]), dtype=float)

# ---- G0b · state order is identical in every vintage -----------------------
order_ok = np.all(np.diff(MU[:, :, 0], axis=1) > 0)
gaps = np.diff(np.sort(MU[:, :, 0], axis=1), axis=1)
print(f"\n[G0b] state order (F_persist increasing) in all {NV} vintages: "
      f"{'PASS' if order_ok else 'FAIL'} | min adjacent gap {gaps.min():.3f}")
assert order_ok, "state indices are not consistently ordered across vintages"
assert np.allclose(A.sum(axis=2), 1.0), "transition rows do not sum to 1"

# stationary distribution per vintage (left eigenvector for eigenvalue 1)
STAT = np.zeros((NV, K))
LAM2 = np.zeros(NV)
for i in range(NV):
    w, v = np.linalg.eig(A[i].T)
    j = int(np.argmin(np.abs(w - 1.0)))
    p = np.real(v[:, j])
    STAT[i] = p / p.sum()
    LAM2[i] = np.sort(np.abs(w))[-2]
print(f"[info] lambda2 across vintages: min {LAM2.min():.4f} "
      f"med {np.median(LAM2):.4f} max {LAM2.max():.4f} "
      f"| half-life {np.log(.5)/np.log(np.median(LAM2)):.1f} steps")
print(f"[info] memory after {KCAP} steps <= {LAM2.max()**KCAP:.2e} (stationary)")

# precomputed matrix powers, A^1 .. A^KCAP, per vintage
APOW = np.zeros((NV, KCAP + 1, K, K))
for i in range(NV):
    M = np.eye(K)
    APOW[i, 0] = M
    for k in range(1, KCAP + 1):
        M = M @ A[i]
        APOW[i, k] = M

# ------------------------------------------------- vintage assignment -------
asof = vt["asof"].to_numpy()
dates = st["TR_DATE"].to_numpy()
vidx = np.searchsorted(asof, dates, side="left") - 1
scorable = vidx >= 0

# ---- G0 · causality --------------------------------------------------------
used = vidx[scorable]
viol = int((asof[used] >= dates[scorable]).sum())
print(f"\n[G0] causality — vintages scoring a date at/before their own asof: "
      f"{viol}  {'PASS' if viol == 0 else 'FAIL'}")
assert viol == 0

param_age = (dates[scorable] - asof[used]).astype("timedelta64[D]").astype(int)
print(f"     scorable {scorable.sum():,}/{len(dates):,} ({scorable.mean():.1%}) "
      f"| burn-in {(~scorable).sum():,}")
print(f"     parameter age (days): med {np.median(param_age):.0f} "
      f"p95 {np.percentile(param_age, 95):.0f} max {param_age.max():.0f}")

# ------------------------------------------------- trading calendar ---------
cal = np.sort(_td.to_numpy())
cal_idx = {d: i for i, d in enumerate(cal)}
assert set(dates.tolist()) <= set(cal.tolist()), "a scored date does not price"
print(f"[info] trading calendar {len(cal):,} days "
      f"({cal.min()} -> {cal.max()}) — priced days only")

# ------------------------------------------------- restrict to scorable -----
sc = st.filter(pl.Series(scorable))
v_row = vidx[scorable]
d_row = dates[scorable]
X = sc.select(FEATS).to_numpy()
cis = sc["cisin"].to_numpy()
tpos = np.array([cal_idx[d] for d in d_row], dtype=np.int64)
N = len(cis)

# ------------------------------------------------- emissions ----------------
# normalised per row (max over states = 1) so the linear recursion cannot
# underflow; the normalisation cancels when the posterior is rescaled.
LL = np.empty((N, K))
for i in range(NV):
    m = v_row == i
    if not m.any():
        continue
    xi = X[m]
    for k in range(K):
        d = xi - MU[i, k]
        LL[m, k] = (-0.5 * ((d * d) / SIG2[i, k]).sum(axis=1)
                    - 0.5 * np.log(2 * np.pi * SIG2[i, k]).sum())
B = np.exp(LL - LL.max(axis=1, keepdims=True))


def forward(emis, vrow, tp, seg_starts, seg_ends):
    """Linear-space gap-aware forward filter. Returns (N, K) posteriors."""
    P = np.empty((len(emis), K))
    steps = np.zeros(len(emis), dtype=np.int32)
    for s, e in zip(seg_starts, seg_ends):
        v = vrow[s]
        a = STAT[v] * emis[s]
        P[s] = a / a.sum()
        for t in range(s + 1, e):
            v = vrow[t]
            k = 1 if _UNIT else min(int(tp[t] - tp[t - 1]), KCAP)
            steps[t] = k
            a = (P[t - 1] @ APOW[v, k]) * emis[t]
            P[t] = a / a.sum()
    return P, steps


starts = np.flatnonzero(np.r_[True, cis[1:] != cis[:-1]])
ends = np.r_[starts[1:], N]
print(f"[info] {len(starts):,} stock segments over {N:,} scorable rows")

t0 = time.time()
POST, STEPS = forward(B, v_row, tpos, starts, ends)
print(f"[info] forward filter {time.time() - t0:.1f}s")

# ============================ GATES =========================================
print("\n" + "=" * 72)

# ---- G1 · independent log-space reimplementation ---------------------------
LOGA = np.log(APOW + 1e-300)


def forward_log(ll, vrow, tp, seg_starts, seg_ends):
    P = np.empty((len(ll), K))
    for s, e in zip(seg_starts, seg_ends):
        a = np.log(STAT[vrow[s]] + 1e-300) + ll[s]
        a -= a.max()
        P[s] = np.exp(a) / np.exp(a).sum()
        for t in range(s + 1, e):
            v = vrow[t]
            k = 1 if _UNIT else min(int(tp[t] - tp[t - 1]), KCAP)
            prev = np.log(P[t - 1] + 1e-300)
            m = prev[:, None] + LOGA[v, k]
            mm = m.max(axis=0)
            a = mm + np.log(np.exp(m - mm).sum(axis=0)) + ll[t]
            a -= a.max()
            P[t] = np.exp(a) / np.exp(a).sum()
    return P


rng = np.random.default_rng(42)
pick = rng.choice(len(starts), size=min(400, len(starts)), replace=False)
sub_s, sub_e = starts[pick], ends[pick]
Plog = forward_log(LL, v_row, tpos, sub_s, sub_e)
mask = np.zeros(N, bool)
for s, e in zip(sub_s, sub_e):
    mask[s:e] = True
d1 = np.abs(POST[mask] - Plog[mask]).max()
print(f"[G1] linear vs independent log-space, {mask.sum():,} rows: "
      f"max|diff| {d1:.3e}  {'PASS' if d1 < 1e-10 else 'FAIL'}")
assert d1 < 1e-10

# ---- G2 · future-perturbation invariance -----------------------------------
# Corrupt every observation strictly AFTER a cut date and re-run. Any leak of
# future information into the past shows up as a non-zero difference here.
cut = np.datetime64("2020-06-30")
future = d_row > cut
Bp = B.copy()
Bp[future] = rng.random((int(future.sum()), K)) * 10.0
POSTp, _ = forward(Bp, v_row, tpos, starts, ends)
past = ~future
d2 = np.abs(POST[past] - POSTp[past]).max()
print(f"[G2] future-perturbation invariance ({int(future.sum()):,} rows after "
      f"{cut} randomised)")
print(f"     max|diff| on the {int(past.sum()):,} rows at/before cut: "
      f"{d2:.3e}  {'PASS' if d2 == 0.0 else 'FAIL'}")
assert d2 == 0.0, "look-ahead detected: past posteriors moved"

# ---- G3 · sanity -----------------------------------------------------------
simplex = np.abs(POST.sum(axis=1) - 1).max()
finite = bool(np.isfinite(POST).all())
print(f"[G3] simplex max|sum-1| {simplex:.2e} | all finite {finite} | "
      f"min {POST.min():.2e} max {POST.max():.6f}  "
      f"{'PASS' if simplex < 1e-12 and finite else 'FAIL'}")
assert simplex < 1e-12 and finite

# ---- sensitivity: stationary vs startprob re-entry prior --------------------
SP = np.stack([np.asarray(json.loads(vt["startprob"][i]), dtype=float)
               for i in range(NV)])
_stat_backup = STAT.copy()
STAT = SP
POSTs, _ = forward(B, v_row, tpos, starts, ends)
STAT = _stat_backup
age = np.zeros(N, dtype=np.int32)
for s, e in zip(starts, ends):
    age[s:e] = np.arange(e - s)
print("\n[sens] re-entry prior stationary vs startprob, max|diff| by age:")
for lo, hi in [(0, 0), (1, 2), (3, 5), (6, 9), (10, 19), (20, 10**9)]:
    m = (age >= lo) & (age <= hi)
    if m.any():
        lab = f"{lo}" if lo == hi else (f"{lo}-{hi}" if hi < 10**9 else f"{lo}+")
        print(f"       obs {lab:>6}: {np.abs(POST[m] - POSTs[m]).max():.2e} "
              f"({m.sum():,} rows)")

# ------------------------------------------------- out-of-window flag -------
ws_d = vt["window_start"].to_list()      # python date objects for polars
asof_d = vt["asof"].to_list()
parts = []
for i in range(NV):
    inw = st.filter((pl.col("TR_DATE") >= ws_d[i])
                    & (pl.col("TR_DATE") <= asof_d[i]))
    parts.append(inw.select("cisin").unique()
                 .with_columns(pl.lit(i, dtype=pl.Int64).alias("vintage_id")))
inwin = pl.concat(parts).with_columns(pl.lit(True).alias("_in"))

# ------------------------------------------------- assemble ------------------
out = (sc.select(["cisin", "TR_DATE", "era"] + FEATS)
         .with_columns(
             pl.Series("p_sell", POST[:, 0]),
             pl.Series("p_neutral", POST[:, 1]),
             pl.Series("p_buy", POST[:, 2]),
             pl.Series("fstate", [SNAME[i] for i in POST.argmax(axis=1)]),
             pl.Series("p_max", POST.max(axis=1)),
             pl.Series("vintage_id", v_row.astype(np.int64)),
             pl.Series("param_age_days", param_age.astype(np.int32)),
             pl.Series("gap_steps", STEPS),
             pl.Series("obs_age", age))
         .join(inwin, on=["cisin", "vintage_id"], how="left")
         .with_columns(pl.col("_in").is_null().alias("stock_out_of_window"))
         .drop("_in"))

OUT.parent.mkdir(parents=True, exist_ok=True)
out.write_parquet(OUT)
write_lineage(OUT, inputs=[VALIDATION_DATA / "states_v3.parquet",
                          OUTPUTS / "phase3" / "vintages.parquet"],
              extras={"module": "C2", "suffix": _SUF, "trunc": str(_TRUNC),
                      "unit_step_diagnostic": _UNIT, "rows": out.height})

# ------------------------------------------------- report --------------------
print("\n" + "=" * 72)
print("C2 SUMMARY")
print(f"  rows            {out.height:,}")
print(f"  stocks          {out['cisin'].n_unique():,}")
print(f"  dates           {out['TR_DATE'].n_unique():,} "
      f"({out['TR_DATE'].min()} -> {out['TR_DATE'].max()})")
print(f"  out-of-window   {out['stock_out_of_window'].sum():,} "
      f"({out['stock_out_of_window'].mean():.3%}) — scorable (emissions are "
      f"global), flagged for downstream")
print(f"  gap>1 step      {(STEPS > 1).sum():,} ({(STEPS > 1).mean():.2%}) | "
      f"gap>20 {(STEPS > 20).sum():,} | at cap {(STEPS >= KCAP).sum():,}")
print("\n  filtered state mix vs Phase-I Viterbi labels:")
mix = (out.group_by("fstate").agg(pl.len().alias("n"))
          .with_columns((pl.col("n") / out.height).alias("share")).sort("fstate"))
vit = (sc.group_by("state").agg(pl.len().alias("n"))
         .with_columns((pl.col("n") / sc.height).alias("share")).sort("state"))
vd = dict(zip(vit["state"], vit["share"]))
for r in mix.iter_rows(named=True):
    print(f"    {r['fstate']:12s} filtered {r['share']:6.2%}   "
          f"viterbi {vd.get(r['fstate'], float('nan')):6.2%}")
agree = float((out["fstate"] == sc["state"]).mean())
print(f"\n  filtered vs Viterbi agreement: {agree:.2%}")
print("  mean posterior confidence p_max: "
      f"{float(out['p_max'].mean()):.3f}")
print(f"\n  wrote {OUT}")
print(f"  total {time.time() - t_start:.1f}s")
print("=" * 72)
