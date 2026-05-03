# CCP Concentration Risk Model — ODD Protocol
*CMS-MD v1.1.0 | CCP Concentration Risk Model | Simudyne Java SDK | April 2026*

---

## Domain
Financial Markets / Financial Market Infrastructure

## Question
How does the concentration of clearing member exposures and heterogeneous capital structures interact with CCP margin and default-fund mechanisms to propagate or absorb member defaults across FTSE 100 and FTSE 250 markets?

---

## Patterns

- **ConcentrationDefaultCascade**: A banking clearing member's default exhausts Level 1 and Level 2 waterfall resources, drawing on pooled default fund (Level 3) at least once per 510-tick trading day when stressed 2008-2009 data is used
- **MarginCallClustering**: Margin calls cluster at 60-tick intervals during high-volatility periods; >=3 simultaneous margin calls in a single hour indicates systemic stress
- **CapitalRatioSeparation**: Non-banking CMs freeze order submission when capital_ratio <= 0.08 (stop-out), while banking CMs deleverage actively -- producing asymmetric liquidity withdrawal under stress
- **DefaultFundAdequacy**: The cover-2 methodology keeps totalDF >= sum of 2 largest CM loss scenarios; exchange skin-in-the-game (10%) is consumed at Level 2 before socialised mutualism at Level 3
- **LiquidityFragmentation**: Under stressed conditions, ZI traders account for a disproportionate share of volume as CM order flow declines, widening bid-ask spreads

---

## Scales

| Dimension | Value |
|---|---|
| Temporal | 1 step = 1 minute (deltaT = 1/3600 of a trading day); 1 trading day = 510 steps; margin call every 60 steps (hourly); default fund recalculated every 510 steps |
| Spatial | No spatial dimension; star topology: all clearing members connected to single MatchingEngine (CCP) via bidirectional CMToEX / EXToCM links |
| Population | 1 MatchingEngine, 10 BankingClearingMembers, 10 NonBankingClearingMembers, 10 ZeroIntelligenceTraders (31 total); population static (no entry/exit except default) |

---

## Agents

| Agent | Count | Key State | Behavior |
|---|---|---|---|
| MatchingEngine | 1 | bidsFTSE100/250, asksFTSE100/250, txnPrice, midPrice, totalDF, ownDF, cmBalanceSheets, defaultList | Maintains two LOBs (FTSE 100, FTSE 250); price-time priority matching; mid-price update; broadcasts NewPrice; calculates cover-2 default fund; runs 5-level default waterfall on DefaultFlag receipt |
| BankingClearingMember | 10 | cash ~U[5B,10B], balanceSheet, zScore~N(0,1), limitPriceFTSE100/250, queuedBids/Asks, pnl, callIndicator | Computes limit prices via z_score × sigma_fundamental offset from fundamental signal; if capital_ratio > 0.08: normal order flow; if capital_ratio <= 0.08: forced deleveraging (fire sale) to restore 8% threshold; margin call every 60 ticks; defaults when cash <= 0 |
| NonBankingClearingMember | 10 | cash ~U[5M,10M], balanceSheet, zScore~N(0,1), limitPriceFTSE100/250, queuedBids/Asks, pnl, callIndicator | Same limit-price logic as BCM; crucially stops all order submission when capital_ratio <= 0.08 (no deleveraging action); margin call every 60 ticks; defaults when cash <= 0 |
| ZeroIntelligenceTrader | 10 | queuedBids, queuedAsks, cancelOrders | Generates random order flow: limit orders (prob alpha=0.15), market orders (prob mu=0.025), cancellations (prob delta=0.025); provides background liquidity independent of fundamental value |

---

## Design Concepts

| Concept | Description |
|---|---|
| Emergence | Systemic stress (default waterfall exhaustion, spread widening) emerges from individual CMs independently targeting capital adequacy; no agent is coded to "cause a crisis" |
| Sensing | CMs observe only: market price broadcast (NewPrice), their own balance sheet (cash, positions, PnL), and transaction confirmations (BuyerWin/SellerWin). CMs cannot observe other CMs' capital ratios, default status, or order books |
| Stochasticity | z_score drawn from N(0,1) at initialization (fixed, determines each CM's private valuation offset); ZI order submission governed by Bernoulli draws (alpha, mu, delta); order quantity drawn from Discrete[1,10] each step; jump-diffusion overlaid on GBM fundamental (Poisson(lambda=3), N(0,0.01)) though currently commented out in favour of data-driven fundamental signal |
| Adaptation | No learning; capital ratio check creates path-dependent state (hasDefaulted flag); credit-quality-like differentiation is structural (BCM vs NBCM) not adaptive |
| Prediction | No explicit expectations; limit price = fundamentalSignal + z_score × sigma -- agents act on private valuation vs current market price without forecasting future price |
| Objectives | CMs maximise trading activity subject to capital adequacy constraint (8% minimum); BCMs deleverage to restore 8%; NBCMs halt trading below 8%; MatchingEngine has no objective (institutional clearing function) |

---

## Mechanisms

| # | Mechanism | Category | Causal Hypothesis |
|---|---|---|---|
| 1 | HeterogeneousPrivateValuation | behavioral | Each CM has a fixed z_score from N(0,1) offsetting their limit price from the fundamental signal; this creates heterogeneous reservation prices that generate persistent order flow even with a common fundamental anchor |
| 2 | CapitalAdequacyConstraint | institutional | 8% minimum capital ratio (cash / \|tradePosition\|) mirrors Basel III; BCMs fire-sale to restore compliance, NBCMs stop out -- asymmetric behavior under stress creates differential liquidity effects |
| 3 | VariationMarginCall | institutional | PnL vs maintenance margin threshold (imPercent × (1-mmPercent) × price × volume) is checked every 60 ticks; variation margin settlement directly transfers cash, creating potential for sequential cash depletion under adverse price moves |
| 4 | DefaultFundCoverN | institutional | Cover-2 methodology sets totalDF = sum of 2 largest CM loss scenarios (dfPercent × price × volume) recalculated every 510 ticks; ensures pre-funded mutualized loss-sharing capacity |
| 5 | FiveLevelDefaultWaterfall | institutional | Sequential absorption: (1) defaulted CM IM + DF, (2) exchange SITG (10%), (3) pooled CM DF, (4) surviving CM cash pro-rata, (5) exchange absorbs remainder; models real CCP waterfall (LCH, CME, Eurex structures) |
| 6 | PositionAuction | institutional | Defaulted CM's open positions are auctioned to the surviving CM with the largest opposing position at recovery rate (0.6); transfers position risk without open-market liquidation |
| 7 | LimitOrderBookMatching | structural | Price-time priority matching across two separate LOBs; mid-price updates from best bid/ask after each match; orders expire after 10 ticks, preventing stale-book buildup |
| 8 | ZeroIntelligenceLiquidity | structural | Background noise traders provide exogenous order flow independent of fundamental or margin pressures; calibrated to Cont-Stoikov-style ZI parameters (alpha=0.15, mu=0.025, delta=0.025) |
| 9 | DataDrivenFundamentalSignal | environmental | Fundamental signal loaded from historical FTSE 100/250 Bloomberg CSV (2008-2009 stressed or 2018-2019 normal); 510-value intraday series per trading day drives limit-price anchor -- empirically grounded rather than synthetic GBM |

---

## Interactions

| Producer | Consumer | Message | Payload |
|---|---|---|---|
| ZeroIntelligenceTrader | MatchingEngine | CancelOrder | List of orderIDs to remove from LOB |
| BankingClearingMember | MatchingEngine | BuyOrder / SellOrder | List of Bid / Ask orders (symbol, price, quantity, orderID, expiry) |
| NonBankingClearingMember | MatchingEngine | BuyOrder / SellOrder | Same as BCM |
| MatchingEngine | BankingClearingMember / NBCM | BuyerWin / SellerWin | Transaction confirmation (price, filled qty, remaining qty, IM, MM, orderID, symbol) |
| BankingClearingMember / NBCM | MatchingEngine | CMBalanceSheet | Full balance sheet snapshot (cash, positions, margins, default status) |
| MatchingEngine | All CMs | NewPrice | Current FTSE100 and FTSE250 transaction prices |
| MatchingEngine | Individual CM | CMBalanceSheet | Updated balance sheet after DF recalculation (dfAmount, dfContribution) |
| BankingClearingMember / NBCM | MatchingEngine | DefaultFlag | Signal that CM cash <= 0; triggers link removal and waterfall |
| ZeroIntelligenceTrader | MatchingEngine | ZI order types | Stochastic limit/market/cancel orders per ZI probabilities |

---

## Step Sequence

| Order | Agent | Action |
|---|---|---|
| 1 | ZeroIntelligenceTrader | Draw Bernoulli(delta); if cancel: send CancelOrder with stale orderIDs |
| 2 | MatchingEngine | handleCancellations: remove cancelled orderIDs from both FTSE 100 and FTSE 250 LOBs |
| 3 | BCM, NBCM, ZI | sendOrder (Split parallel): submit BuyOrder / SellOrder messages based on capital ratio check and limit price vs market price comparison |
| 4 | MatchingEngine | match: populate LOBs from messages; sort by price-time priority; match FTSE 100 then FTSE 250; send BuyerWin / SellerWin confirmations; expire stale orders; update mid-prices and globals |
| 5 | All CMs | receiveTransactionReports (Trader.receiveTransactionReports): update queuedBids/Asks, positions, margin accounts from transaction confirmations |
| 6 | BCM, NBCM | sendBalanceSheet: transmit current CMBalanceSheet to MatchingEngine |
| 7 | MatchingEngine | updateCMBalanceSheets: aggregate balance sheets; recalculate dfContribution weights (\|tradePosition\| / totalTP) |
| 8 | MatchingEngine | calculateDefaultFund (every 510 ticks): compute cover-2 totalDF; set CM dfAmount; deduct exchange SITG from cash; distribute updated balance sheets to CMs |
| 9 | All CMs | receiveDefaultFundReports: update local balance sheet with new dfAmount |
| 10 | MatchingEngine | sendLatestPrice: broadcast NewPrice (FTSE100 and FTSE250 txn prices) to all CMs |
| 11 | BCM, NBCM | marginCall (every 60 ticks): compute PnL vs maintenance margin; settle variation margin; update tradePosition; send updated balance sheet |
| 12 | MatchingEngine | updateCMBalanceSheets: absorb post-margin balance sheets |
| 13 | BCM, NBCM | checkDefault: if cash <= 0 send DefaultFlag, remove CMToEX links, set hasDefaulted flag |
| 14 | MatchingEngine | defaultWaterfall: run 5-level waterfall for each defaulted CM; auction positions to surviving CMs |

---

## Initialization

| Component | Setup |
|---|---|
| MatchingEngine | cash = 10,000,000; defaultFundContribution = 10% of cash; exchangeID = "LSEG"; empty LOBs and balance sheets |
| BankingClearingMember | cash ~ U[5B, 10B] (uniform × 1,000,000,000); BalanceSheet initialized with zero positions and margins; zScore ~ N(0,1); agentID = Simudyne internal ID |
| NonBankingClearingMember | cash ~ U[5M, 10M] (uniform × 1,000,000); same BalanceSheet setup; zScore ~ N(0,1) |
| ZeroIntelligenceTrader | No explicit state initialization beyond agent creation |
| Fundamental Signal | Read from HDFS CSV (FTSE100/250 Bloomberg data); 510-value intraday array pre-populated; poissonDist = Poisson(jump_lambda=3); jumpDist = N(0, jump_size=0.01) |
| Network | BCM and NBCM groups fully connected to MatchingEngine group (CMToEXLink, EXToCMLink); ZI traders fully connected to MatchingEngine (ZIToEXLink, EXToZILink); no inter-CM links |
| Output Files | Transaction CSV, BalanceSheet CSV, ModelParameter CSV written to HDFS on setup; seed captured in filename |

---

## Observables

| Metric | Level | Frequency | Aggregation | Empirical Target |
|---|---|---|---|---|
| marketPriceFTSE100 | system | per_tick | mid-price update | Track FTSE 100 intraday price path; compare to Bloomberg CSV fundamental signal |
| marketPriceFTSE250 | system | per_tick | mid-price update | Track FTSE 250 intraday price path |
| totalFTSE100BuyVolume / SellVolume | system | per_tick | sum | Order book depth; stressed periods should show thinning buy side |
| CM defaults (accumulator) | system | per_tick | count | 0 defaults in normal 2018-2019; >=1 default episode in stressed 2008-2009 runs |
| marginCall (accumulator) | system | per_tick | count | Clustered margin calls during high-volatility hours (every 60 ticks) |
| totalDF | system | every 510 ticks | cover-2 loss sum | Default fund adequacy: totalDF >= loss of 2 largest members |
| capital_ratio | agent | every 60 ticks | per_CM value | BCMs maintain >8% (fire sale); NBCMs stop at 8%; stressed periods see more sub-8% episodes |
| balanceSheet (cash, tradePosition, IM, MM, dfAmount) | agent | per_tick / every 60 ticks | full state | Written to HDFS BalanceSheet CSV for post-hoc analysis |
| defaultWaterfall level | event | on default | categorical | Distribution of Level 1/2/3/4/5 resolutions; cover-2 design targets Level 3 sufficiency |

---

## Ablation Predictions

| # | Remove Mechanism | Expected Effect | Falsifies If |
|---|---|---|---|
| 1 | Remove CapitalAdequacyConstraint | NBCMs continue trading below 8% CAR → excess leverage → higher default frequency | Default rate unchanged |
| 2 | Remove BCM forced deleveraging (fire sale) | BCMs stop out like NBCMs → less liquidity withdrawal but more gradual cash depletion | Market price impact unchanged |
| 3 | Remove VariationMarginCall | Cash stays constant between trades → defaults only from trading losses, not margin settlements | Default frequency drops to near zero in stressed period |
| 4 | Remove ZeroIntelligenceLiquidity | No background order flow → LOBs deplete during CM stress periods → matching failure | Bid-ask spread unchanged |
| 5 | Set coverNumber=1 (cover-1 DF) | totalDF halved → more Level 3/4/5 defaults → surviving CMs absorb larger cash losses | Default waterfall level distribution unchanged |
| 6 | Remove PositionAuction | Defaulted positions not transferred → orphaned exposure → exchange absorbs all at Level 5 more frequently | Exchange solvency rate unchanged |

---

## Staging

| Stage | Complexity | Components Active |
|---|---|---|
| Stage 1 | Baseline LOB | MatchingEngine + ZI only; verify price discovery without clearing mechanics |
| Stage 2 | + Clearing Members | Add BCM + NBCM with capital ratio check but no margin calls or default fund |
| Stage 3 | + Margin | Enable 60-tick variation margin settlement; verify cash depletion dynamics |
| Stage 4 | + Default Fund | Enable 510-tick cover-2 default fund recalculation |
| Stage 5 | Full Waterfall | Enable 5-level default waterfall + position auction; validate cascade absorption |
| Stage 6 | Stressed Data | Switch from 2018-2019 to 2008-2009 Bloomberg data; validate stressed default scenarios |

---

## Data Sources

| Source | Purpose | Period | Format |
|---|---|---|---|
| FTSE100_0809_BB.csv | Stressed fundamental signal for FTSE 100 | 2008-2009 (GFC) | Bloomberg daily prices; loaded to 510-step intraday array |
| FTSE250_0809_BB.csv | Stressed fundamental signal for FTSE 250 | 2008-2009 (GFC) | Bloomberg daily prices |
| FTSE100_1819_BB.csv | Normal fundamental signal for FTSE 100 | 2018-2019 (pre-COVID) | Bloomberg daily prices |
| FTSE250_1819_BB.csv | Normal fundamental signal for FTSE 250 | 2018-2019 (pre-COVID) | Bloomberg daily prices |
| Storage | HDFS cluster at 10.115.10.175:8022/ccp/input/ | — | CSV; read via Hadoop FileSystem API |

---

## Calibration

| Parameter | Default | Source | Method |
|---|---|---|---|
| imPercent (Initial Margin) | 0.20 | LCH SwapClear / Eurex Clearing rulebooks | Regulatory floor; EMIR Art. 41 requires IM to cover 99% 2-day exposure |
| mmPercent (Maintenance Margin threshold) | 0.95 | CME margin methodology | Maintenance = 95% of IM; variation margin triggered when PnL > 5% of IM |
| recoveryRate | 0.60 | Historical CCP default studies (CPMI-IOSCO 2017) | 60% assumed recovery on auctioned positions; consistent with GFC haircuts |
| dfPercent (DF stress scenario) | 0.20 | Cover-2 EMIR/CPMI-IOSCO standard | 20% stressed loss scenario per member position |
| exDFRatio (Exchange SITG) | 0.10 | EMIR Art. 45 skin-in-the-game | Exchange contributes 10% of totalDF as first-loss before CM pooled fund |
| coverNumber | 2 | EMIR/Dodd-Frank cover-2 requirement | CCP must cover simultaneous default of 2 largest members |
| alpha / mu / delta (ZI) | 0.15 / 0.025 / 0.025 | Cont-Stoikov (2008) LOB calibration | Limit order arrival / market order / cancellation rates typical for equity markets |
| jump_lambda | 3.0 | Merton (1976) jump-diffusion | 3 jumps per trading day on average; consistent with FTSE event frequency |

---

## Validation & Verification

| Test | Type | Expected Result |
|---|---|---|
| LOBMatchingDeterminism | V&V | Same seed → identical transaction sequence across runs |
| MarginCallFrequency | Functional | Every 60 ticks exactly: (ticks+1) % 60 == 0 triggers variation margin |
| DefaultFundCover2 | Functional | totalDF = sum of losses of 2 largest CMs; exact cover-2 arithmetic |
| WaterfallOrdering | Functional | Level 2 only consumed after Level 1 exhausted; Level 3 only after Level 2; etc. |
| BCMFireSaleActivation | Functional | BCM with capital_ratio <= 0.08 submits deleveraging orders in same step; NBCM does not |
| StressedVsNormal | Behavioural | 2008-2009 data produces higher default rate and margin call clustering vs 2018-2019 |
| MidPriceUpdate | Functional | midPrice = 0.5×bestBid + 0.5×bestAsk; globals.marketPrice updated each match step |

---

## Validation Tests

- `default_count_stressed >= 1` per 510-tick trading day (2008-2009 data, seed 42)
- `default_count_normal == 0` in 95%+ of 510-tick runs (2018-2019 data)
- `margin_calls >= 3` per stressed trading day (clustered around high-volatility hours)
- `totalDF >= (sum of 2 largest CM loss scenarios)` at every 510-tick recalculation
- `exchange.cash > 0` throughout normal scenario (exchange solvency)
- `midPriceFTSE100` tracks within 5% of Bloomberg fundamental signal under normal conditions
