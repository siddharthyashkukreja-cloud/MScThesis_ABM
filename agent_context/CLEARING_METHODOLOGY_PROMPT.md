Prompt to feed another AI model for writing the clearing-layer methodology.
Copy everything in the code block below. (Updated for the current committed design — June 2026.)

---

```
You are helping write the METHODOLOGY section for the central-clearing layer of an MSc
thesis: an agent-based model (ABM) of a centrally-cleared single-asset market. The asset is
the E-mini S&P 500 (ES) front-month future (CME, $50 multiplier). The model has two coupled
tiers — a limit-order-book market layer (calibrated to ES stylised facts) and a CENTRAL
CLEARING layer — and the thesis question is whether clearing clients THROUGH members rather
than directly changes how stress propagates, plus the effect of procyclical vs
through-the-cycle margin.

SOURCES TO READ (in order; model.md and clearing_writeups.tex are the source of truth):
1. clearing_writeups.tex — the LaTeX clearing methodology already aligned to the committed
   model (§A participants/balances, §B margin & funding cycle, §C default management). Match
   its notation and content; your job is to expand it into full methodology prose.
2. model.md  — technical reference: §2.2 (clearing-tier architecture), §3 (margin cycle/IM/VM,
   default fund, waterfall/close-out/porting), §4 (parameter tables), §8.1 (element→source map).
3. CLEARING_LAYER_REVIEW.md — a HISTORICAL snapshot, use ONLY for real-world grounding numbers
   (CME initial margin ~$292B, prefunded default fund ~$9.4B, SITG ~$100M, DF/IM ~3-4%, client
   share of margin ~82%) and source URLs — NOT for the model's current mechanics.
4. extras/tex_drafts/model.tex for STYLE and introduction_refs.bib for the exact \citep{} keys.

WHAT TO WRITE: a LaTeX methodology subsection for the clearing layer, same style as the existing
chapters (\citep/\citet author-year; prose/paragraphs, NOT bullet lists; present tense; academic
register; light equations only where a formula clarifies). Structure:
 (a) Two-tier architecture: one CCP; 10 banking clearing members (BCM) that also trade a small
     own account (own book capped at 2x cash); 5 non-banking members (NBCM), pure intermediaries
     with no own book; 90 cleared end-clients (30 FT + 20 MT + 40 ZI). Only 5 of the
     10 BCMs plus all 5 NBCMs carry clients. Clients are assigned to clearers by a
     LEVERAGE-BALANCED (capacity-proportional) rule — each client routed to the clearer whose
     resulting client-book-to-capital ratio is lowest, so large asset-manager clients clear
     through high-capital bank-CMs and small accounts through non-bank CMs. Balance-sheet scales:
     BCM $5-10B, NBCM $0.5-3B, FT $0.5-3B, MT $0.2-1B, ZI $0.2-0.5B. A figure reference for
     clearing_topology.png is welcome.
 (b) Margin. Initial margin is charged on a participant's GROSS exposure (own book plus the
     gross notional of its client book, the absolute client positions summed with NO netting
     across clients). The rate is
        f = max( 6%,  alpha99 * sigma_daily * sqrt(MPOR) ),
     with alpha99 = 3.0 the EMPIRICAL 99% quantile of volatility-standardised ES returns
     (filtered historical simulation — the standard CCP method; the Gaussian 2.33 understates the
     fat-tailed ES move by ~1/3), MPOR = 2 days the close-out horizon (a daily move scaled by
     sqrt(MPOR) — volatility grows with the square root of time), and sigma_daily an
     exponentially-weighted estimate of recent returns (RiskMetrics ~11-day half-life). The 6%
     floor is the anti-procyclicality minimum (EMIR Art. 28): it binds in calm (margin holds at
     ~6%), and the VaR term takes over in stress (rate climbs to ~12-25%). The rate is the H2
     lever — reactive (tracks current vol) by default, with flat and static variants as
     comparators. A cleared CLIENT posts its OWN initial margin: the CCP's call passes down
     through its clearing member, the client funds it from its own cash, and the member transfers
     it to the CCP (pass-through; the member holds none of the client's market risk unless the
     client defaults). Variation margin is called every hourly cycle: each position is re-marked
     to the current price and its P&L settled in cash that same cycle, with a 95%-of-IM
     maintenance level the trigger to restore full margin as the price moves. Initial margin is
     physically ESCROWED as cash in the CCP's segregated account (posted when a position opens,
     returned as it closes, seized first on default). A participant that cannot fund a call from
     its own cash defaults — default is a LIQUIDITY event (Nasdaq 2018, LME 2022).
 (c) Default fund: cover-2 Stress-Loss-Over-Initial-Margin, SLOIM_i = max(0, (s - f) * E_i),
     with s an extreme-but-plausible 2-day move (empirical ~99.9% quantile, floored at 10% and
     capped at 35%); fund = top-two sum + 10% buffer, cash-prefunded; each member's contribution
     is prefunded cash in proportion to its margin and (for an NBCM) charged entirely against its
     gross client book; exchange skin-in-the-game ~3% of the fund.
 (d) Default management: the close-out (see CRITICAL POINTS), client porting (EMIR Art. 48,
     capacity-checked), and the 5-level waterfall (defaulter IM seized first, then: defaulter DF
     contribution, CCP skin-in-the-game, pooled surviving-member DF, surviving members' cash
     pro-rata, CCP own capital ~$1.5B).
 (e) A consolidated parameter table with sources (mirror model.md §4 + §8.1).

CRITICAL CORRECTNESS POINTS (the model was revised — do not describe superseded behaviour):
 - The close-out is SPLIT by who resolves it. A defaulted CLIENT is closed out by its MEMBER,
   which assumes the position and LIQUIDATES IT ON THE OPEN MARKET over an Almgren-Chriss schedule
   — this open-market client fire-sale IS part of the baseline and is the client-level price-impact
   channel. A defaulted MEMBER's residual book (after porting) is resolved by the CCP via a
   synthetic TRANSFER at a recovery rate of 0.80 (a 20% haircut to the waterfall; NO open-market
   disposal, no endogenous price impact for the member book), with the defaulter's posted IM
   seized first. Justify 0.80 as a conservative stressed recovery between Lehman/LCH 2008
   (a hedged book closed out WITHIN initial margin — cite bell_holden_2018 / BIS Dec 2018) and a
   concentrated, illiquid book (Nasdaq/Aas 2018, which exhausted the defaulter's collateral and
   DREW the mutualised fund — same source).
 - Two DEFERRED stress arms exist and must be described as off-baseline sensitivities, not the
   headline: (i) CCP OPEN-MARKET disposal of the member book (CLOSEOUT_MODE="firesale") — endogenous
   liquidity-dependent loss + price impact; (ii) a DISORDERLY-close-out recovery of 0.60
   (Nasdaq/Aas-class). The headline analysis is the calm and stressed (COVID) regimes at ACTUAL
   severity, plus a GBM scenario ensemble.
 - Solvency / leverage. The binding member constraint is a Basel-style LEVERAGE RATIO,
   kappa = cash (net of posted IM) / exposure >= 8% (NOT a cash/IM margin-coverage ratio). On a
   breach a BCM DELEVERAGES its own book (Almgren-Chriss) toward a 10% buffer then freezes; an NBCM,
   holding no own book, STOPS OUT. The BCM own book is bounded by a STATIC gross-leverage cap
   (|own| <= 2 x cash) — a flat dealer-book limit with no volatility input (there is no sigma-VaR
   own-book limit; that earlier device was removed). The client position cap is the broker HOUSE
   MARGIN, a fixed 15% of notional (~6.67x leverage; the post-2020 security-futures statutory
   minimum, 17 CFR 242.400-406), distinct from the procyclical CCP IM above.
 - This model EXTENDS the Simudyne CCP Risk Model ODD (cite simudyne_odd / deloitte_ccp). Flag
   each deviation with a reason: (i) flat 20% IM -> procyclical empirical-FHS VaR floored at 6%;
   (ii) the client tier is the thesis's addition (the ODD has no clients); (iii) the close-out is
   split (member transfer at 0.80 — above the ODD's 0.60 — and an open-market client fire-sale);
   (iv) FTSE -> ES asset. All regulatory constants are taken from regulation/data, never fitted.

HEADLINE FRAMING (state the resilience as the finding; do NOT claim an endogenous cascade):
 - The tiered CCP is RESILIENT: across a GBM stress-scenario ensemble the mutualised default fund
   is drawn in 0 of 6 paths (even at a -56% drawdown); under DIRECT clearing the same shocks draw
   it in 4 of 6 — so the intermediation tier LOCALISES losses at the member (H1).
 - Procyclical margin raises the stress-time collateral demand ~1.4-1.6x and drives the
   leverage-cycle deleverage (H2). Net (omnibus) client margining ~halves posted IM vs gross (NET).
 - The client share of posted IM is ~75% (toward the empirical ~82%); calm is clean.

LIMITATIONS to disclose honestly: the SLOIM-IM coupling (a higher reactive IM mechanically shrinks
the cover-2 fund, so report the H2 result with a fixed-fund robustness arm — DF_DECOUPLE_IM); the
calm default-fund-to-IM ratio runs a little high (a small-N artifact — cover-2 over only 15 members
is a larger IM share than a real 50-100-member CCP); and deep mutualisation (waterfall L3+) is
rarely reached at COVID severity — that is the resilience finding, not a gap.

CITATIONS: use \citep{key} with the keys in model.md §8.1 (e.g. emir_rts, cftc_reg117, basel_frtb,
euronext_a9, almgren_chriss_2000, vuillemey_2023, glasserman_wu_2018, duffie_zhu_2011,
galbiati_soramaki_2013, bell_holden_2018, cont_2017, brunnermeier_pedersen_2009, simudyne_odd,
deloitte_ccp, cftc_sec_secfutures_margin). Where §8.1 marks a source "ADD", insert a \citep{}
placeholder and list it for the author.

RULES: do not invent numbers — every figure must trace to clearing_writeups.tex, model.md,
CLEARING_LAYER_REVIEW.md, or a cited source. Prefer prose over lists. Flag each ODD deviation.
Keep formulas light (one per concept; detail in words). Keep the register precise and defensible
(a thesis examiner will read it). Output LaTeX only for the section, plus a short trailing list of
any \citep keys that still need adding to the .bib.
```
