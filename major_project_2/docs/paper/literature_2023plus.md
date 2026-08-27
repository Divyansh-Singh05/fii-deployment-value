# Curated Literature (2023+) — 60 Papers
### For the FII flow-regimes paper: 30 via Consensus (academic search engine) + 30 via direct scholarly-web scraping (Crossref / OpenAlex / Semantic Scholar APIs)

All entries are 2023 or later, journal-published (no preprints in the final list). Each has a
one-line note on where it plugs into the paper. Themes: (A) FII/foreign flows & India,
(B) HMM & regime-detection methodology, (C) regime-based decisions/portfolios,
(D) institutional price impact & order flow, (E) fire sales & flow-driven pressure,
(F) informed trading / PIN, (G) herding, (H) crypto regimes.

---

## PART 1 — Consensus (30)

### A. FII / institutional flows in India and emerging markets
1. Thapa, C., et al. (2025). *Policy information uncertainty and foreign institutional investors trading behavior: Evidence from India.* **Review of Quantitative Finance and Accounting.** — Uses NSDL-style transaction-level FII data; directly comparable data setting to ours.
2. Dey, M., et al. (2024). *A study on how institutional investors respond to risk, return and volatility: Evidence from the Indian stock market.* **Journal of the Knowledge Economy.** — FII vs DII causality with market risk/return; positioning for the India flow literature.
3. Naik, P. K., et al. (2026). *Asymmetric and temporal effects of investor sentiment and institutional trading behavior on returns and volatility: Evidence from the Indian stock market.* **Review of Behavioral Finance.** — Institutional trading imbalances and volatility; our composition axis extends their imbalance measure.
4. Batra, S., et al. (2023). *Stabilizing or destabilizing: The effect of institutional investors on stock return volatility in an emerging market.* **Multinational Business Review.** — Institutional ownership and Indian volatility; frames the stabilize/destabilize debate our decomposition resolves.

### B. HMM & regime-detection methodology
5. Tampouris, A., et al. (2025). *Adaptive hierarchical hidden Markov models for structural market change.* **Journal of Risk and Financial Management.** — Meta-regime HMM; motivates why plain HMMs miss higher-order structure (our overlay problem).
6. Qin, S., et al. (2024). *On robust estimation of hidden semi-Markov regime-switching models.* **Annals of Operations Research.** — Robust HSMM estimation; cite for dwell-time/sojourn modelling beyond first-order HMM.
7. Oelschläger, L., & Adam, T. (2024). *fHMM: Hidden Markov models for financial time series in R.* **Journal of Statistical Software.** — Canonical software/methods reference for financial HMMs.
8. Čeryová, B., et al. (2025). *Decoding the stock market dynamics in the banking sector: Short versus long-term insights.* **North American Journal of Economics and Finance.** — Hierarchical HMM separating short/long-term trends; parallels our backbone-vs-overlay split.
9. Wang, L., et al. (2025). *Early warning of regime switching in a financial time series: A heteroskedastic network model.* **PLOS ONE.** — HMM + network hybrid for regime early-warning; cite in the forecasting (hazard) section.
10. Koukorinis, A., et al. (2025). *Generative-discriminative machine learning models for high-frequency financial regime classification.* **Methodology and Computing in Applied Probability.** — HMM embeddings + SVM; supports "HMM as feature-generator" reading of our design.
11. Saidane, M. (2026). *Forecasting regime-dependent tail risk in digital asset portfolios with a mixture of hidden Markov factor analyzers.* **Journal of Forecasting.** — Factor-analytic HMM mixtures; closest published cousin of the factorial-HMM extension we propose.

### C. Regime-based decisions and portfolios
12. Shu, Y., et al. (2024). *Downside risk reduction using regime-switching signals: A statistical jump model approach.* **Journal of Asset Management.** — Jump models vs HMMs for regime persistence; the modern benchmark our HMM should be compared against.
13. Pun, C. S., et al. (2023). *Data-driven distributionally robust CVaR portfolio optimization under a regime-switching ambiguity set.* **M&SOM.** — Regime-switching ambiguity sets; methodological support for regime-conditional decisions.
14. Ma, G., et al. (2023). *Dynamic trading with Markov liquidity switching.* **Automatica.** — Optimal trading when price impact itself switches regimes — the theory mirror of our SHARK/HOSTAGE asymmetry.
15. Blanchard, E., et al. (2025). *Data-driven investment strategies using Bayesian inference in regime-switching models.* **Applied Stochastic Models in Business and Industry.** — Smoothing to stabilize HMM regimes for trading; cite for false-signal/transaction-cost discussion.
16. Kitvanitphasu, A., et al. (2025). *Bitcoin wild moves: Evidence from order flow toxicity and price jumps.* **Research in International Business and Finance.** — VPIN predicts jumps in crypto; links informed-trading measures to regime shifts.

### D. Institutional price impact and order flow
17. Xu, L., et al. (2025). *When order execution meets informed trading.* **Quantitative Finance.** — Conjectures transient impact is caused by the informed/uninformed mix — the theoretical frame for our transitory–permanent split.
18. Fan, Y., et al. (2025). *Institutional granular impact is benign on asset sales and price efficiency.* **Journal of Financial Markets.** — Common vs granular (large-player) trading shocks; the closest published analogue of our dispersed-vs-concentrated axis.
19. Glossner, S., et al. (2025). *Do institutional investors stabilize equity markets in crisis periods? Evidence from COVID-19.* **Management Science.** — Downscaling trades reverse, repositioning trades persist — exactly our transitory/permanent decomposition in a crisis setting.
20. Naviglio, M., et al. (2025/2026). *Why is the estimation of metaorder impact with public market data so challenging?* **Quantitative Finance.** — Metaorder impact and reversion from public data; grounds our worked-order (multi-day episode) interpretation.
21. Xu, L., et al. (2024). *Optimal trading and competition with information in the price impact model.* **Quantitative Finance.** — Competition among informed traders and price efficiency; theory for the dispersed-information channel.
22. Jain, P., et al. (2023). *Determinants of commodity market liquidity.* **Financial Review.** — Permanent (Amihud) vs transitory (noise) liquidity components; vocabulary matches our decomposition.
23. Tsaknaki, I.-Y., et al. (2023). *Online learning of order flow and market impact with Bayesian change-point detection methods.* **Quantitative Finance.** — Order-flow regimes via online change-point detection; the causal-filtering benchmark for our Phase-II layer.
24. Kolm, P., et al. (2023). *Deep order flow imbalance: Extracting alpha at multiple horizons from the limit order book.* **Mathematical Finance.** — State-of-the-art ML on order flow; the "model class is not the bottleneck" comparison point.

### E. Fire sales and flow-driven price pressure
25. Giannetti, M., et al. (2024). *Bond price fragility and the structure of the mutual fund industry.* **Review of Financial Studies.** — Ownership concentration limits fire-sale exposure; concentration as a state variable, as in our archetypes.
26. Kundu, S. (2023). *Financial covenants and fire sales in closed-end funds.* **Management Science.** — Constraint-driven selling creates price pressure; mechanism support for liquidity-demanding concentrated selling.
27. Sim, M. (2026). *Short-term return reversals and fund exits.* **Emerging Markets Finance and Trade.** — Active-institution exits drive reversals in Korea; the EM twin of our post-episode reversal.

### F. Informed trading / PIN
28. Ghachem, M., & Ersan, O. (2025). *Estimation of the probability of informed trading models via an expectation-conditional maximization algorithm.* **Financial Innovation.** — Modern PIN estimation; methodological backing for our MLE implementation.
29. Kropiński, P., Bosek, B., et al. (2024). *State ownership, probability of informed trading, and profitability potential: Evidence from the Warsaw Stock Exchange.* **International Review of Financial Analysis.** — PIN applied to an investor class; supports our FII-slice PIN design.
30. Quang, L. T., et al. (2026). *Herding behaviour of institutional investors in stock price manipulation.* **Spanish Journal of Finance and Accounting.** — Institutional herding by investor type; adjacent to our participation-concentration reading.

*(Consensus URLs for all 30 are in the reference list at the bottom of this file.)*

---

## PART 2 — Scholarly-web scrape (30) — Crossref / OpenAlex / Semantic Scholar APIs

### A. FII / foreign flows, India and EM
31. Sharma, V. K., Bhatia, S., et al. (2023). *Investment behavior of foreign institutional investors and implied volatility dynamics in India.* **Journal of Risk and Financial Management.** https://doi.org/10.3390/jrfm16110470
32. Shruti, R., & Thenmozhi, M. (2024). *Foreign institutional ownership stability and stock price crash risk.* **Journal of International Financial Markets, Institutions and Money.** https://doi.org/10.1016/j.intfin.2024.101937
33. Haider, Z. A., & Onali, E. (2026). *Foreign institutional investors and corporate governance: A transaction cost perspective.* **Finance Research Letters.** https://doi.org/10.1016/j.frl.2026.109784
34. Bian, J., Chan, K., & Han, B. (2023). *Cross-border equity flows and information transmission: Evidence from Chinese stock markets.* **Journal of International Financial Markets, Institutions and Money.** https://doi.org/10.1016/j.intfin.2023.101755
35. Choi, S., & Havel, J. (2024). *Geopolitical risk and U.S. foreign portfolio investment: A tale of advanced and emerging markets.* **Journal of International Money and Finance.** https://doi.org/10.1016/j.jimonfin.2024.103253
36. Kacperczyk, M., Nosal, J., et al. (2025). *Global volatility and firm-level capital flows.* **Journal of Financial Economics.** https://doi.org/10.1016/j.jfineco.2025.104078
37. Converse, N., Levy-Yeyati, E., & Williams, T. (2023). *How ETFs amplify the global financial cycle in emerging markets.* **Review of Financial Studies.** https://doi.org/10.1093/rfs/hhad014
38. Sharma, R. (2025). *Extreme capital flow episodes in emerging markets: Incidence and drivers.* **Review of International Economics.** https://doi.org/10.1111/roie.70026
39. Pandey, A., & Sharma, A. K. (2024). *The cumulative prospect theory and fund flows in emerging markets.* **Investment Analysts Journal.** https://doi.org/10.1080/10293523.2024.2312707

### D. Institutional trading, price impact, reversals
40. Akepanidtaworn, K., Di Mascio, R., et al. (2023). *Selling fast and buying slow: Heuristics and trading performance of institutional investors.* **The Journal of Finance.** https://doi.org/10.1111/jofi.13271
41. Barardehi, Y. H., Bernhardt, D., et al. (2025). *Institutional liquidity costs, internalized retail trade imbalances, and the cross-section of returns.* **Journal of Financial and Quantitative Analysis.** https://doi.org/10.1017/s0022109025000043
42. Goyal, A., Reed, A. V., et al. (2025). *Stealthy shorts: Informed liquidity supply.* **Journal of Financial Economics.** https://doi.org/10.1016/j.jfineco.2025.104155
43. Deuskar, P., Khatri, A., & Sunder, J. (2025). *Insider trading restrictions and informed trading in peer stocks.* **Management Science.** https://doi.org/10.1287/mnsc.2022.02907
44. Ha, J. (2025). *Institutional trading and satellite data.* **Finance Research Letters.** https://doi.org/10.1016/j.frl.2024.106341
45. Hong, X., Yao, J., & Zhuang, Z. (2025). *Institutional trading and short-term stock returns — Evidence from Dragon and Tiger lists.* **Emerging Markets Finance and Trade.** https://doi.org/10.1080/1540496X.2025.2520377
46. Zhu, Z., & Sun, L. (2023). *Economic policy uncertainty and short-term reversals.* **Journal of Financial Research.** https://doi.org/10.1111/jfir.12371
47. Chen, C., Stivers, C. T., & Sun, L. (2024). *Short-term momentum and reversals, turnover, and a stock's price-to-52-week-high ratio.* **Journal of Empirical Finance.** https://doi.org/10.1016/j.jempfin.2024.101556

### F. Informed trading / PIN
48. Su, S., & Sha, Y. (2023). *Good (bad) news and the probability of informed trading: Evidence from illegal insider trading.* **Emerging Markets Finance and Trade.** https://doi.org/10.1080/1540496x.2023.2266111
49. Li, Y., Shi, Y., & Sha, Y. (2026). *Short selling and the probability of informed trading: Insights from interlocking markets.* **Pacific-Basin Finance Journal.** https://doi.org/10.1016/j.pacfin.2026.103091
50. Choi, H.-E. (2025). *Transition to proof-of-stake and informed trading.* **Finance Research Letters.** https://doi.org/10.1016/j.frl.2024.106570

### B. Regime / HMM methodology
51. Ding, Y., Kambouroudis, D., & McMillan, D. G. (2025). *Forecasting realised volatility using regime-switching models.* **International Review of Economics & Finance.** https://doi.org/10.1016/j.iref.2025.104171
52. Otranto, E., & Scaffidi Domianello, L. (2025). *On using fuzzy clustering for detecting the number of states in Markov switching models.* **Annals of Operations Research.** https://doi.org/10.1007/s10479-025-06585-w
53. Xu, X., Peng, H., & Chen, Y. (2026). *Deep switching state space model for nonlinear time series forecasting with regime changes.* **International Journal of Forecasting.** https://doi.org/10.1016/j.ijforecast.2025.05.001
54. Cortese, F. P., Di Ruzza, S., et al. (2025). *A statistical sparse jump model for automatic identification of dynamical transitions.* **Nonlinear Dynamics.** https://doi.org/10.1007/s11071-025-11171-7
55. Kabašinskas, A., Kopa, M., et al. (2024). *Stress testing for second-pillar life-cycle pension funds using hidden Markov models.* **Annals of Operations Research.** https://doi.org/10.1007/s10479-024-06041-1
56. Kijkarncharoensin, A. (2025). *Identifying risk regimes in a sectoral stock index through a multivariate hidden Markov framework.* **Risks.** https://doi.org/10.3390/risks13070135

### H. Cryptocurrency regimes and liquidity
57. Pakštaitė, V., Filatovas, E., et al. (2025). *Bitcoin price regime shifts: A Bayesian MCMC and hidden Markov model analysis.* **Mathematics.** https://doi.org/10.3390/math13101577
58. Shakourloo, A., & Azimli, A. (2026). *Regime-switching in bitcoin volatility under global uncertainty: Markov-switching evidence.* **Research in International Business and Finance.** https://doi.org/10.1016/j.ribaf.2026.103295
59. Farag, H., Luo, D., & Yarovaya, L. (2025). *Returns from liquidity provision in cryptocurrency markets.* **Journal of Banking & Finance.** https://doi.org/10.1016/j.jbankfin.2025.107411

### G. Herding
60. Yang, W.-R., & Chuang, M.-C. (2023). *Do investors herd in a volatile market? Evidence of dynamic herding in Taiwan.* **Finance Research Letters.** https://doi.org/10.1016/j.frl.2022.103364

---

## How to deploy these in the IJF draft
- **Review of Literature §2:** items 1–4, 31–39 (India/EM flows), 17–24, 40–47 (impact/reversal), 25–27 (fire sales), 28–30, 48–50 (PIN), 60 (herding).
- **Methodology §4 (why HMM):** 5–11, 51–56 — establishes HMM as the standard latent-regime tool and motivates the hybrid design.
- **Future research:** 11 (HMM factor analyzers), 12/54 (jump models), 53 (deep switching SSM), 23 (causal change-point filtering).
- **Crypto generalization paragraph (optional):** 16, 57–59.

## Verification status
Part 2 DOIs come directly from Crossref/OpenAlex records (machine-read, reliable).
Part 1 entries come from Consensus metadata; before citing in the submitted manuscript,
resolve each to its DOI (the consensus.app links below carry the source links).
Consensus source links (Part 1, in order 1–30):
1. https://consensus.app/papers/details/92485b620a975fc5adc7de7e476eb5d1/ (Thapa)
2. https://consensus.app/papers/details/47f2540a2b3d5e19ab07f8f536c1076a/ (Dey)
3. https://consensus.app/papers/details/f382fc61312c500fafc9939f8f5b79c4/ (Naik)
4. https://consensus.app/papers/details/0f9b8ef0feed5d23b9ae80888be0e512/ (Batra)
5. https://consensus.app/papers/details/d24569a4e5ba5252a4a5ae4c78240cc0/ (Tampouris)
6. https://consensus.app/papers/details/26ce381f362553ec928e671a60974795/ (Qin)
7. https://consensus.app/papers/details/b5bb72240e9b5980b0d66cc106365f03/ (Oelschläger)
8. https://consensus.app/papers/details/d787ac48cb50522ab492ae9eb42b9dbe/ (Čeryová)
9. https://consensus.app/papers/details/b61dd533b63d568cbc535f12f93b4377/ (Wang)
10. https://consensus.app/papers/details/f1d560a60774526e9836966c3fb9fbb7/ (Koukorinis)
11. https://consensus.app/papers/details/fc7575b1d5115c1481cc1254ee14b1e3/ (Saidane)
12. https://consensus.app/papers/details/af35e1366f905ee684be45e4e031242f/ (Shu)
13. https://consensus.app/papers/details/05518e72407e555380b0afd7eb56f5f9/ (Pun)
14. https://consensus.app/papers/details/2897ce63ab405c50845bc2d5f1663e92/ (Ma)
15. https://consensus.app/papers/details/5de21320a42859deb18a9ec7c176718a/ (Blanchard)
16. https://consensus.app/papers/details/f6b9e19fbb2f55a9baf74be46f092f34/ (Kitvanitphasu)
17. https://consensus.app/papers/details/2823da67e7a55d9da9e5e5926a224b74/ (Xu 2025)
18. https://consensus.app/papers/details/02268d53863f5b9d8a46ab6ec761d61a/ (Fan)
19. https://consensus.app/papers/details/ba20435a5128554ab2c37a7344398290/ (Glossner)
20. https://consensus.app/papers/details/a9dd51b2d4a65b8194419ce34f0495e7/ (Naviglio)
21. https://consensus.app/papers/details/a0032afb6b645322ae223233e15da599/ (Xu 2024)
22. https://consensus.app/papers/details/47437a43b4c35cd4b33ecb2be9b00de0/ (Jain)
23. https://consensus.app/papers/details/16fc1fb31e8f5f9c9f9f5004497dc1f3/ (Tsaknaki)
24. https://consensus.app/papers/details/ef38ad83b7f55ee48aa76ff6725fb41c/ (Kolm)
25. https://consensus.app/papers/details/5ac61b9a659f5629a5c7f7bac9b74521/ (Giannetti)
26. https://consensus.app/papers/details/b2aae028c51f54dea16202fcb9172e82/ (Kundu)
27. https://consensus.app/papers/details/0354a0cb5aa95824a3039ba5af309d7d/ (Sim)
28. https://consensus.app/papers/details/8d4128a91fc75dd7818cb1cbf20021b5/ (Ghachem & Ersan)
29. https://consensus.app/papers/details/d94e293439d55c7d8d0ec64992abe0e1/ (Kropiński/Bosek)
30. https://consensus.app/papers/details/ed30432d21e55fa6b303b139274ea5af/ (Quang)
