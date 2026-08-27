# Backtesting

Phase: `backtest`. Engine: `src/fii/backtest/engine.py`; gates:
`engine_gates.py`; strategies: `module12b/c/d/e`. Results log: §3o.

## Engine design (WorldQuant-BRAIN-style daily L/S)

Daily cross-sectional alpha → portfolio weights:

1. demean within day over the valid universe;
2. scale to unit gross exposure $\sum_i |w_{it}| = 1$;
3. truncate $|w_{it}| \le w_{max} = 0.05$ and renormalize (iterated).

**Execution timing** (the part that decides whether a backtest is honest):
a signal formed at close $t$ is traded at close $t + L$ and earns the
close-to-close return of day $t + L + 1$. Default lag $L = 1$. Positions:

$$H_t = W_{t-1-L}, \qquad pnl^{gross}_t = \sum_i H_{it}\, r_{it},$$
$$TO_t = \sum_i |H_{it} - H_{i,t-1}|, \qquad
pnl^{net}_t = pnl^{gross}_t - c \cdot TO_t,$$

with one-way cost $c$ = 15 bps default (grid 0/5/10/15/30).

**Real-time discipline for regime events**: an episode END is knowable only
at the close of the first day *after* the run:
$\text{impulse}_t = \mathbb{1}[S_{t-1} = X \wedge S_t \ne X]$.

## Engine gates (pre-registered; the engine may not change without them)

| Gate | Test | Detects |
|---|---|---|
| G1 exactness | vectorized == naive per-day loop | silent vectorization bugs |
| G2 alignment | cheat alpha (= tomorrow's return): Sharpe > 10 at lag 0, collapses at lag 1 | look-ahead; fake lag |
| G3 costs | constructed flip book: turnover exactly 2.0; net == gross − c·TO | cost/turnover accounting |

Observed: G2 cheat Sharpe +211.5 → −0.5. All PASS.

## Metrics (definitions)

With daily pnl $x_t$ (fraction of book), $A = 252$:

- Annualized return $\bar x A$; volatility $\sigma_x\sqrt{A}$;
  **Sharpe** $= \bar x/\sigma_x \cdot \sqrt{A}$.
- **Sortino** uses downside deviation
  $\sqrt{\mathbb{E}[\min(x_t,0)^2]}$ in place of $\sigma_x$.
- **Max drawdown** on the cumulative sum; **Calmar** = ann. return / maxDD.
- **Turnover** = mean $TO_t$ (fraction of book traded per day).
- **Margin** = $\bar x / \overline{TO}$ in bps per dollar traded — the
  number to compare against the cost assumption.
- **Fitness** (WQ) $= \text{Sharpe}\cdot\sqrt{|\text{ann ret}| / \max(\overline{TO}, 0.125)}$.
- **Breakeven cost** = gross margin: the one-way bps at which net = 0.

## Strategy pairs (with vs without the regime model)

| Pair | Without HMM | With HMM |
|---|---|---|
| S1 | 20-day loser reversal | reversal gated to 20d after SHARK_DIST/ACC ends |
| S2 | 10-day FII flow-following | flow signal zeroed on concentrated-regime days |
| S3 | mechanical climax/quiet-seller proxies | regime book (post-END reversal + HOSTAGE short) |
| S4 | proxy style-switch | trend *within* concentrated episodes, reversion after END, short HOSTAGE, flat otherwise |

Baselines are run and **frozen before** the HMM twins (no tuning against the
comparison). Verdict rule, pre-registered: TEST-era net-of-cost paired
ΔSharpe, moving-block bootstrap (block 20d, B = 2000) 95% CI.

## Results and the honest reading

- **Baselines**: the mechanical concentration proxy (S3) is the best no-model
  book (gross Sharpe +1.46/+1.44 both eras) — the *concept* carries signal
  even without the model. Nothing survives 15 bps.
- **Formal verdicts (net, TEST)**: S1H inconclusive; S2H/S3H "HMM subtracts";
  S4H "HMM adds value" — but the S4 verdict is driven mostly by the twin's
  cost bleed, and S3H had a construction flaw (unequal leg magnitudes made it
  a HOSTAGE-short churn book), both owned in the log.
- **Gross diagnosis (the meaningful layer)**: S2H improves gross Sharpe in
  *both* eras (ΔSharpe CI [+0.03,+0.22] / [+0.07,+0.49]) — removing
  concentrated-regime days helps flow-following, a portfolio-level
  confirmation of the headline result. S4H is the only book gross-positive
  in both eras (+1.00/+0.54) against a gross-negative twin.
- **Breakeven costs 2–8 bps one-way** — below institutional execution costs.

**Interpretation (paper Appendix E)**: the regime model's information is real
and appears exactly where the research says it should, but the reversal
(~50 bp/20d on ~7% of stock-days) cannot be harvested net of costs at daily
rebalance — a limits-to-arbitrage account of why the regularity persists.
These are economic-significance results, not trading claims. Short legs
additionally face India's SLB constraints.
