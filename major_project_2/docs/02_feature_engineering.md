# Feature Engineering

Phase: `features`. Stage: `feature_store`
(`src/fii/features/module1_feature_store_v2.py`).
Column-level detail: `FII_stockday_data_dictionary.md`.

## Purpose

Transform raw FII trade records into a per-(stock, day) feature store of
**flow-only** variables. All models consume *only* this store — no model
rebuilds features internally.

## Inputs

`ISIN_MAPPING/2011.parquet … 2025.parquet` — masked NSDL FII trades:
date, masked entity IDs (FII / SUB_ACC / BRKER), ISIN, buy/sell flag
(TR_TYPE 01/04), quantity, rate. Filter: RATE > 0.

## Output

`ISIN_MAPPING/stockday_features_v2.parquet` — ~788k complete-case stock-days,
10 features + raw components (buy/sell value & counts, entity HHIs,
coverage flags).

## The base panel

Per (stock, day): $BUY_{it}$, $SELL_{it}$ (values), counts, quantities. Then

$$NET_{it} = BUY_{it} - SELL_{it}, \quad GROSS_{it} = BUY_{it} + SELL_{it}.$$

## The three axes and their features

**Axis 1 — persistence** (is the flow a campaign or a blip?):
trailing ~20-day mean of $\operatorname{sign}(NET)$ scaled by intensity.
Low → transient two-sided flow; high → sustained directional pressure.

**Axis 2 — trade size / blockiness**: mean trade size vs the stock's own
history. Large prints suggest conviction; fragmentation suggests
algorithmic/indexed flow.

**Axis 3 — entity concentration** (the axis the headline result lives on):
participation-weighted Herfindahl of each entity's share of the day's
sell-side (resp. buy-side) book:

$$HHI^{sell}_{it} = \sum_{e} \left(\frac{SELL_{e,it}}{\sum_{e'} SELL_{e',it}}\right)^2 .$$

High = one/few institutions dominate (concentrated); low = many small
participants (dispersed). **Constraint discovered by the entity audit**:
masked IDs are re-minted ~monthly (0.0% twelve-month persistence over
84–98-month spans), so concentration is *within-day/within-month only* — no
cross-month fire-sale tracking is possible. This is a designed-around
limitation, stated in the paper.

Supporting features: participation breadth (distinct entities), directional
imbalance $NET/GROSS$, streaks, flow-beta, activity, size dispersion.

## Normalization: within-day cross-sectional probit ranks

Every feature is rank-transformed **within its day** and mapped through the
standard normal quantile:

$$F_{it} = \Phi^{-1}\!\left(\frac{\operatorname{rank}_i(x_{it})}{N_t + 1}\right).$$

Why: FII participation breadth grew ~254% over the sample with a structural
break in 2021. Cross-sectional ranking removes the market-wide level each
day, so features answer "is this stock's flow unusual *relative to today's
cross-section*" — stationary by construction across eras.

## Leakage discipline (non-negotiable)

1. **Strictly backward windows** — every rolling statistic uses data through
   day $t$ only.
2. **May–June 2021 masked** before any window is computed (embargo around the
   train/test boundary).
3. **No price, VIX, or macro variables in features** — flow only. Price data
   enters only at validation, so economic validation is a genuine
   out-of-domain test, not a fit.
4. **NET_INNOV is not a feature.** The flow-surprise object is deliberately
   reserved as a *validation yardstick* (built with hindsight, Module 9);
   using it as a feature would forfeit it as an independent control.

## Validation

- Complete-case accounting per year; the "Axis-3 coverage = 1.0" claim was
  itself caught as a NaN-bug artifact and corrected (real usable share falls
  to ~0.53 by 2025 — 2024–25 results are coverage-confounded, stated).
- Known redundancy documented: blockiness × activity r = 0.88; imbalance
  tails compress (ties at ±1).
- Feature signatures replicate train → test with drift ≤ 0.15 (Module 3).
