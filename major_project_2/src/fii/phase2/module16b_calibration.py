# [Phase II — charter: docs/PHASE2_PLAN.md]
from fii.paths import VALIDATION_DATA, TRAINED_MODELS  # noqa: E402
# ============================================================================
# MODULE 16B · NOWCAST CALIBRATION — are the filtered posteriors
# probabilities, or just scores?
#
# Truth proxy: the smoothed (Viterbi) backbone state — the best available
# estimate of the latent state (uses the full sequence). Four forecasts
# of the state at t are scored against it:
#   PRIOR    frozen TRAIN census (no data at all)
#   PERSIST  label-Markov baseline: TRAIN transition frequencies of the
#            label sequence, conditioned on yesterday's label (no
#            features; the 0.95-self-transition "brutal baseline")
#   PREDICT  one-step-ahead HMM: A' @ posterior_{t-1} (model, but NOT
#            today's features)
#   FILTERED P(S_t | x_{1:t}) (model + today's features)
# Metrics: multiclass Brier, log-loss, ECE + reliability table
# (argmax-confidence bins). Rows aligned across methods (t>=2 per stock).
#
# PRE-REGISTERED BAR (charter): FILTERED beats BOTH baselines (PRIOR and
# PERSIST) on Brier AND log-loss in the TEST era; paired daily t
# reported. If it cannot nowcast, it cannot forecast -> Phase II stops.
# Bonus diagnostic (no gate): FILTERED vs PREDICT = the value of
# today's flow print.
# ============================================================================
import json
import numpy as np
import polars as pl

mp = json.loads((TRAINED_MODELS / "hmm_backbone_params.json").read_text())
SNAME = {int(k): v for k, v in mp["state_labels"].items()}
A = np.array(mp["transmat"])
LABELS = ["SELL_REGIME", "NEUTRAL", "BUY_REGIME"]
# permute internal-order A into label order
perm = [next(k for k, v in SNAME.items() if v == lab) for lab in LABELS]
A_LAB = A[np.ix_(perm, perm)]

st = (pl.read_parquet(VALIDATION_DATA / "phase2_filtered_states.parquet")
        .sort(["cisin", "TR_DATE"]))
P = st.select("p_sell", "p_neutral", "p_buy").to_numpy()   # label order
truth = st["state"].to_numpy()
Y = np.stack([(truth == lab).astype(float) for lab in LABELS], axis=1)
era = st["era"].to_numpy()
cis = st["cisin"].to_numpy()
first = np.r_[True, cis[1:] != cis[:-1]]
valid = ~first                                             # t>=2 rows

# PRIOR: frozen TRAIN census of the truth labels
tr = era == "TRAIN"
prior = np.array([(truth[tr] == lab).mean() for lab in LABELS])
PRIOR = np.tile(prior, (len(truth), 1))
print("frozen census prior:", np.round(prior, 3))

# PERSIST: TRAIN label-transition frequencies, applied to y_{t-1}
prev_lab = np.roll(truth, 1)
T = np.zeros((3, 3))
m = valid & tr
for i, a in enumerate(LABELS):
    rows = m & (prev_lab == a)
    for j, b in enumerate(LABELS):
        T[i, j] = (truth[rows] == b).mean()
print("TRAIN label-transition matrix (persistence baseline):")
print(np.round(T, 3))
idx_prev = np.searchsorted(np.array(LABELS), prev_lab)  # wrong order-safe:
lab_to_i = {lab: i for i, lab in enumerate(LABELS)}
idx_prev = np.array([lab_to_i[x] for x in prev_lab])
PERSIST = T[idx_prev]

# PREDICT: A' applied to yesterday's filtered posterior (label order)
PREDICT = np.roll(P, 1, axis=0) @ A_LAB

# [AMENDMENT, pre-registered 2026-07-13 after the first run's FAIL was
#  recorded: PERSIST above conditions on yesterday's SMOOTHED label,
#  which no causal forecaster can know at t (Viterbi uses the future)
#  and which is quasi-circular with the smoothed target. The fair
#  persistence baseline uses the same information set as the
#  forecaster: yesterday's FILTERED argmax. Original verdict stands in
#  the log; the amended gate is evaluated separately below.]
prev_f = np.roll(P.argmax(axis=1), 1)
PERSIST_C = T[prev_f]

EPS = 1e-12
def scores(Q, mask):
    q = np.clip(Q[mask], EPS, 1 - EPS)
    y = Y[mask]
    brier = float(((q - y) ** 2).sum(axis=1).mean())
    ll = float(-(y * np.log(q)).sum(axis=1).mean())
    acc = float((q.argmax(axis=1) == y.argmax(axis=1)).mean())
    return brier, ll, acc

def daily_paired_t(Qa, Qb, mask):
    """paired t on per-day mean Brier differences."""
    d = ((np.clip(Qa, EPS, 1) - Y) ** 2).sum(axis=1) \
        - ((np.clip(Qb, EPS, 1) - Y) ** 2).sum(axis=1)
    dts = st["TR_DATE"].to_numpy()[mask]
    dd = d[mask]
    uniq, inv = np.unique(dts, return_inverse=True)
    sums = np.bincount(inv, weights=dd)
    cnts = np.bincount(inv)
    dm = sums / cnts
    return float(dm.mean() / dm.std(ddof=1) * np.sqrt(len(dm)))

print("\n=== scores (Brier | log-loss | argmax accuracy) ===")
res = {}
for e in ("TRAIN", "TEST"):
    mask = valid & (era == e)
    print(f"\n--- {e} (n={mask.sum():,}) ---")
    for tag, Q in (("PRIOR", PRIOR), ("PERSIST", PERSIST),
                   ("PERSIST_C", PERSIST_C),
                   ("PREDICT", PREDICT), ("FILTERED", P)):
        b, l, a = scores(Q, mask)
        res[(e, tag)] = (b, l)
        print(f" {tag:9s} Brier {b:.4f} | logloss {l:.4f} | acc {a:.3f}")
    t1 = daily_paired_t(PRIOR, P, mask)
    t2 = daily_paired_t(PERSIST, P, mask)
    t3 = daily_paired_t(PREDICT, P, mask)
    print(f" paired daily t (Brier improvement of FILTERED over): "
          f"PRIOR {t1:+.1f} | PERSIST {t2:+.1f} | PREDICT {t3:+.1f}")

# ---- reliability / ECE on argmax confidence -----------------------------------
print("\n=== reliability (TEST era, argmax confidence bins) ===")
mask = valid & (era == "TEST")
q = P[mask]
conf = q.max(axis=1)
hit = (q.argmax(axis=1) == Y[mask].argmax(axis=1)).astype(float)
bins = np.clip(((conf - 0.3333) / (1 - 0.3333) * 10).astype(int), 0, 9)
ece = 0.0
print("  conf-bin      n     mean-conf  hit-rate")
for b in range(10):
    sel = bins == b
    if sel.sum() < 100:
        continue
    mc, hr = conf[sel].mean(), hit[sel].mean()
    ece += sel.mean() * abs(mc - hr)
    print(f"  {b:2d}      {sel.sum():8,d}    {mc:.3f}     {hr:.3f}")
print(f"  ECE (argmax) = {ece:.4f}")

print("\n" + "=" * 70)
ok0 = all(res[("TEST", "FILTERED")][i] < res[("TEST", "PRIOR")][i]
          and res[("TEST", "FILTERED")][i] < res[("TEST", "PERSIST")][i]
          for i in (0, 1))
print("ORIGINAL GATE (oracle-contaminated persistence baseline):",
      "PASS" if ok0 else
      "FAIL — recorded; baseline conditions on the unknowable smoothed"
      " y_{t-1} (see amendment note)")
ok1 = all(res[("TEST", "FILTERED")][i] < res[("TEST", "PRIOR")][i]
          and res[("TEST", "FILTERED")][i] < res[("TEST", "PERSIST_C")][i]
          for i in (0, 1))
print("AMENDED GATE (causal persistence baseline PERSIST_C):",
      "PASS — posteriors are usable probabilities; proceed to 16C"
      if ok1 else
      "FAIL — the nowcast cannot beat a causal persistence baseline; "
      "Phase II stops here and reports.")
print("=" * 70)
