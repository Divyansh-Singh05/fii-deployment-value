from fii.paths import VALIDATION_DATA, TRAINED_MODELS  # noqa: E402
# ============================================================================
# MODULE 17E · FACTORIAL HMM — NOWCAST CALIBRATION (16B protocol)
#
# Are the FHMM's filtered posteriors probabilities, or just scores?
# Scored for BOTH chains against the smoothed (Viterbi) FHMM state as
# truth proxy.  Baselines per chain:
#   PRIOR      frozen TRAIN census of the truth labels (no data)
#   PERSIST_C  causal label-Markov: TRAIN transition frequencies of the
#              truth labels, conditioned on yesterday's FILTERED argmax
#              (16B's amendment adopted from the start: conditioning on
#              yesterday's smoothed label is quasi-circular — that
#              lesson is inherited here, not relearned)
#   PREDICT    one-step-ahead FHMM: product posterior_{t-1} pushed
#              through kron(A_D, A_C), then marginalized (model, but
#              NOT today's features)
#   FILTERED   chain marginal of P((d,c)_t | x_{1:t})
#
# PRE-REGISTERED BAR (16B charter, applied per chain): FILTERED beats
# PRIOR and PERSIST_C on Brier AND log-loss in the TEST era.
#   - chain D bar is REQUIRED (if the direction nowcast fails, the
#     hazard/decision layers have no foundation);
#   - chain C bar is REPORTED AND GATED SEPARATELY — it is the novel
#     quantity (a causal, daily, probabilistic concentration nowcast,
#     which the naive HMM cannot produce at all).
# Metrics: multiclass Brier, log-loss, argmax accuracy, ECE.
# ============================================================================
import json

import numpy as np
import polars as pl

from fii.models.fhmm_stages.fhmm_core import FactorialGaussianHMM

params = json.loads((TRAINED_MODELS / "fhmm_params.json").read_text())
model = FactorialGaussianHMM.from_chain_params(params)
D_NAME = {int(k): v for k, v in params["d_names"].items()}
C_NAME = {int(k): v for k, v in params["c_names"].items()}
KD, KC = model.k_d, model.k_c

st = (pl.read_parquet(VALIDATION_DATA / "fhmm_filtered_states.parquet")
        .sort(["cisin", "TR_DATE"]))
era = st["era"].to_numpy()
cis = st["cisin"].to_numpy()
first = np.r_[True, cis[1:] != cis[:-1]]
valid = ~first
dates = st["TR_DATE"].to_numpy()
tr = era == "TRAIN"

D_LABELS = ["SELL_REGIME", "NEUTRAL", "BUY_REGIME"]
C_LABELS = ["DISPERSED", "CONC_SELL", "CONC_BUY"]
PD = st.select("p_sell", "p_neutral", "p_buy").to_numpy()
PC = st.select("p_disp", "p_csell", "p_cbuy").to_numpy()

# PREDICT: product posterior_{t-1} @ kron(A_D, A_C), marginalized.
# product posterior is approximated by the outer product of the chain
# marginals (exact if chains were independent given x_{1:t}; the small
# dependence is the price of storing marginals — noted, not hidden).
A = model.transmat_
prod_prev = (PD[:, :, None] * PC[:, None, :]).reshape(-1, KD * KC)
pred_prod = np.roll(prod_prev, 1, axis=0) @ A
PRD_D = pred_prod.reshape(-1, KD, KC).sum(axis=2)
PRD_C = pred_prod.reshape(-1, KD, KC).sum(axis=1)
# reorder model-index columns into label order
d_perm = [next(k for k, v in D_NAME.items() if v == lab)
          for lab in D_LABELS]
c_perm = [next(k for k, v in C_NAME.items() if v == lab)
          for lab in C_LABELS]
PRD_D = PRD_D[:, d_perm]
PRD_C = PRD_C[:, c_perm]

EPS = 1e-12

def battery(tag, P, PRED, truth_col, labels):
    truth = st[truth_col].to_numpy()
    Y = np.stack([(truth == lab).astype(float) for lab in labels], axis=1)
    prior = np.array([(truth[tr] == lab).mean() for lab in labels])
    PRIOR = np.tile(prior, (len(truth), 1))
    # causal persistence: truth-transition matrix given filtered argmax
    lab_to_i = {lab: i for i, lab in enumerate(labels)}
    prev_f = np.roll(P.argmax(axis=1), 1)
    truth_i = np.array([lab_to_i[x] for x in truth])
    T = np.zeros((len(labels), len(labels)))
    m = valid & tr
    for i in range(len(labels)):
        rows = m & (prev_f == i)
        for j in range(len(labels)):
            T[i, j] = (truth_i[rows] == j).mean() if rows.sum() else \
                prior[j]
    PERSIST_C = T[prev_f]

    def scores(Q, mask):
        q = np.clip(Q[mask], EPS, 1 - EPS)
        y = Y[mask]
        return (float(((q - y) ** 2).sum(axis=1).mean()),
                float(-(y * np.log(q)).sum(axis=1).mean()),
                float((q.argmax(axis=1) == y.argmax(axis=1)).mean()))

    print(f"\n===== {tag} (truth = smoothed {truth_col}) =====")
    print(f"frozen census prior: {np.round(prior, 3)}")
    res = {}
    for e in ("TRAIN", "TEST"):
        mask = valid & (era == e)
        print(f"--- {e} (n={mask.sum():,}) ---")
        for nm, Q in (("PRIOR", PRIOR), ("PERSIST_C", PERSIST_C),
                      ("PREDICT", PRED), ("FILTERED", P)):
            b, l, a = scores(Q, mask)
            res[(e, nm)] = (b, l)
            print(f" {nm:9s} Brier {b:.4f} | logloss {l:.4f} "
                  f"| acc {a:.3f}")
    # ECE on TEST argmax confidence
    mask = valid & (era == "TEST")
    q = P[mask]
    conf = q.max(axis=1)
    hit = (q.argmax(axis=1) == Y[mask].argmax(axis=1)).astype(float)
    kbin = np.clip(((conf - 1 / len(labels))
                    / (1 - 1 / len(labels)) * 10).astype(int), 0, 9)
    ece = sum((kbin == b).mean() * abs(conf[kbin == b].mean()
                                       - hit[kbin == b].mean())
              for b in range(10) if (kbin == b).sum() >= 100)
    print(f" ECE (TEST, argmax) = {ece:.4f}")
    ok = all(res[("TEST", "FILTERED")][i] < res[("TEST", "PRIOR")][i]
             and res[("TEST", "FILTERED")][i]
             < res[("TEST", "PERSIST_C")][i] for i in (0, 1))
    return ok

ok_d = battery("CHAIN D — direction nowcast", PD, PRD_D,
               "sstate", D_LABELS)
ok_c = battery("CHAIN C — concentration nowcast", PC, PRD_C,
               "scstate", C_LABELS)

print("\n" + "=" * 70)
print("GATE chain D (required):",
      "PASS — the direction nowcast is a usable probability; "
      "proceed to 17F" if ok_d else
      "FAIL — the FHMM cannot nowcast direction causally; the "
      "hazard/decision layers stop here.")
print("GATE chain C (novel quantity, gated separately):",
      "PASS — the FHMM delivers a causal probabilistic concentration "
      "nowcast (the naive HMM has no such object)" if ok_c else
      "FAIL — concentration posteriors are not calibrated; reported.")
print("=" * 70)
if not ok_d:
    raise SystemExit(1)
