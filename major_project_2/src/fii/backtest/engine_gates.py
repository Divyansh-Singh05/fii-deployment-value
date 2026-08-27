"""Engine correctness gates (module12a) as a pipeline stage."""
import numpy as np

from fii.backtest.engine import backtest, metrics

# ============================ GATES =========================================
print("=" * 70)
print("MODULE 12A · engine gates")
print("=" * 70)
rng = np.random.default_rng(7)
T, N = 800, 100
R = rng.normal(0.0, 0.02, (T, N))
A = rng.normal(0.0, 1.0, (T, N))
A[rng.random((T, N)) < 0.1] = np.nan          # holes in the universe

# ---- G1: vectorized == naive loop ------------------------------------------
bt = backtest(A, R, lag=1, tcost_bps=15.0)
W = np.nan_to_num(bt["W"], nan=0.0)
pnl2 = np.zeros(T)
to2 = np.zeros(T)
hprev = np.zeros(N)
for t in range(T):
    h = W[t - 2] if t >= 2 else np.zeros(N)
    pnl2[t] = float(np.dot(h, np.nan_to_num(R[t], nan=0.0)))
    to2[t] = float(np.abs(h - hprev).sum())
    hprev = h
g1 = (np.allclose(pnl2, bt["pnl_gross"], atol=1e-12)
      and np.allclose(to2, bt["turnover"], atol=1e-12))
print("\nG1 exactness (vectorized vs naive loop):",
      "PASS" if g1 else "FAIL")

# ---- G2: alignment / lag ----------------------------------------------------
cheat = np.full((T, N), np.nan)
cheat[:T - 1] = R[1:]                          # alpha(t) = return of day t+1
m0 = metrics(**{"pnl": backtest(cheat, R, lag=0, tcost_bps=0)["pnl_gross"],
                "to": backtest(cheat, R, lag=0, tcost_bps=0)["turnover"]})
m1 = metrics(**{"pnl": backtest(cheat, R, lag=1, tcost_bps=0)["pnl_gross"],
                "to": backtest(cheat, R, lag=1, tcost_bps=0)["turnover"]})
print(f"\nG2 alignment: cheat-alpha Sharpe lag=0 {m0['sharpe']:+.1f} "
      f"(expect >10), lag=1 {m1['sharpe']:+.1f} (expect |.|<3)")
g2 = (m0["sharpe"] > 10) and (abs(m1["sharpe"]) < 3)
print("G2:", "PASS" if g2 else "FAIL")

# ---- G3: turnover & cost exactness ------------------------------------------
T3 = 12
A3 = np.zeros((T3, 2))
A3[:, 0] = [1, -1] * (T3 // 2)                 # daily flip
A3[:, 1] = -A3[:, 0]
R3 = np.zeros((T3, 2))
b3 = backtest(A3, R3, lag=0, tcost_bps=10.0)
flips = b3["turnover"][2:]                     # steady-state flip days
g3a = np.allclose(flips, 2.0, atol=1e-12)
g3b = np.allclose(b3["pnl_net"],
                  b3["pnl_gross"] - 10e-4 * b3["turnover"], atol=1e-15)
print(f"\nG3 costs: flip-day turnover uniques -> "
      f"{sorted(set(np.round(flips, 10)))} (expect [2.0]); "
      f"net==gross-cost {g3b}")
g3 = g3a and g3b
print("G3:", "PASS" if g3 else "FAIL")

print("\n" + "=" * 70)
if g1 and g2 and g3:
    print("VERDICT: ALL GATES PASS — engine certified.")
    print("Functions weights_from_alpha / backtest / metrics / show /")
    print("daily_ic are live in this session. Proceed to 12B.")
else:
    print("VERDICT: GATE FAILURE — do NOT run 12B. Paste this output.")
print("=" * 70)
