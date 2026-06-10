# writing.md — thesis-writing context for an AI assistant

**Read this together with `README.md`. Those two files are all you need.**
`README.md` is the *technical reference* (model mechanics, agents, clearing tier, formulas,
parameters, how to run). **This** file is the *thesis context*: research question,
contributions, results with exact numbers and honest caveats, how each reference is used, the
chapter structure, and writing conventions. When a claim needs a mechanism/formula detail, the
relevant `README.md` section is named inline. The developer change-log (`AGENT.md`) holds the
full per-step decision history ("D-series") and is only for deep code archaeology — not needed
for writing.

---

## 1. Thesis at a glance
- **Topic:** an agent-based model (ABM) of central clearing — client clearing, contagion, and
  CCP systemic risk — on ES E-mini S&P 500 futures. MSc thesis, VU Amsterdam.
  Repo: github.com/siddharthyashkukreja-cloud/MScThesis_ABM.
- **One line:** a two-tier ABM — a *calibrated* limit-order-book market-microstructure layer +
  a *central-clearing* layer (CCP, clearing members, cleared clients, with real
  margin / default-fund / waterfall mechanics) — used to study how clearing-member tiering
  propagates market stress.
- **Regimes:** *calm* (2019 ES) and *stressed* (the Feb–Apr 2020 COVID crash window).

## 2. Motivation, research question & contributions

**Motivation (literature-grounded — see §7).** CCPs mutualise counterparty risk but *concentrate*
it in their clearing members (banking CMs / FCMs / general clearing members), which guarantee client
trades, post collateral, and absorb the first loss. Most quantitative work is CCP-centric; the
clearing-member perspective is comparatively under-studied (mostly qualitative / regulatory). Three
channels make it systemically important and motivate the thesis:
- **Single-agent dependency** — the modal client has *one* clearing agent and no backup (OFR 2026);
  when that agent is distressed, clients lose CCP access and de-risk (Credit Suisse / Archegos).
- **Crowding** — clearing-member positions are positively correlated ("crowded positions",
  Menkveld 2015); correlated portfolios → simultaneous defaults that standard margin overlooks, and
  crowding spikes with volatility (Huang-Menkveld-Yu 2021: crowding ≈ 17% of the worst CCP exposure
  spikes).
- **Procyclical margin in stress** — Mar-2020: IM at the four largest EU/UK CCPs rose ≈ one third,
  daily VM on euro-area fund derivative exposures quintupled, 6% of funds couldn't cover it
  (ESRB 2020). "The workings of the client-clearing market is an open question" (Menkveld & Vuillemey).

**Research question — candidates (pick one as the thesis RQ; all are addressable with the built ABM):**
- **RQ-A — buffer vs single point of failure:** does the clearing-member layer *stabilise or amplify*
  systemic risk — a GCM as an extra loss-absorbing buffer between a client default and the CCP, versus
  single-agent dependency as a single point of failure — and does **client type** (volatility/HFT-like
  vs directional) change the answer?
- **RQ-B — cover-2 sufficiency / contagion drivers:** under realistic *calibrated* stress, is the
  **cover-2** default-fund framework sufficient to contain client→CM→CCP contagion, and what drives a
  breach — the *magnitude* of the move, its *concentration / speed* (overnight gaps vs a single shock),
  or *crowding*?
- **RQ-C — crowding & margin methodology:** do crowded client positions / cross-product margin offsets
  create a structural **liquidity exposure** for clearing members in stress (when CCP gross margin
  surges while correlation haircuts become unreliable), and would a crowding-aware margin (CoMargin;
  Menkveld's Margin(A)) reduce it?

> The built model most directly answers **RQ-B** (it already produces the cover-2 containment +
> gaps-vs-shock results) and a version of **RQ-A** (clearing-tier / client-type effects). **RQ-C**
> would need a crowding mechanism + a CoMargin comparison added — scope it before committing.

**Contributions:**
1. A *calibrated* two-tier ABM coupling a realistic, ES-calibrated microstructure to a
   *regulation-grounded* clearing tier — most CCP ABMs assume the price process; here it is calibrated
   to the empirical ES stylised facts, so the environment the clearing layer acts in is earned, not
   assumed. (Addresses the noted gap: little quantitative work from the clearing-member perspective.)
2. Real margin methodology — procyclical VaR/SPAN initial margin, a cash/IM capital ratio, a cover-2
   Stress-Loss-Over-IM default fund, a deficit-based five-level waterfall, and Almgren–Chriss
   fire-sales — rather than placeholder haircuts.
3. Findings on the *drivers* of contagion (concentration/speed vs magnitude; the overnight-gap
   structure as a mitigant; cover-2 containment at COVID severity with mutualisation onset ≈ 2.5×)
   plus a methodological result (the market calibration is not clearing-invariant under stress).

## 3. The model in one paragraph
(README "The model in brief" / "Market and agents" / "Clearing tier".) A 1-minute call-auction
LOB with **price-time priority**; trades print at the resting limit price, never the mid. Three
calibrated agent types: **fundamental** traders (value / mean-reversion around an exogenous
Kalman-smoothed real-ES fundamental), **momentum** traders (EWMA trend signal, limit-only), and
**zero-intelligence** traders (Cont-Stoikov-Talreja noise: limit / market / cancel). On top: a
**CCP**, **banking CMs** (also trade own books under a VaR house limit), **non-banking CMs**, and
~90 **cleared clients**, with hourly variation margin, procyclical initial margin, an 8% cash/IM
capital-ratio gate, a cover-2 default fund, a five-level loss waterfall, and Almgren–Chriss
deleveraging. Order lifetime is governed by agent cancellation — no blanket time-to-live (D58).

## 4. Key results & numbers — cite exactly; flag interim values

### 4.1 Market layer — stylised facts (Ch. 5)
Reproduces the Cont (2001) stylised-fact set: **fat tails** (calm Hill ≈ 2.7–3.0 vs empirical
≈ 2.96; stressed comparable), **volatility clustering** (positive |return| ACF), and **near-zero
return autocorrelation**. Headline baseline loss: **D_grid = 43.8 (calm) / 7.9 (stressed)**
(3-seed scale). *Kurtosis is a diagnostic only* — it runs above empirical (outlier-dominated);
Hill is the tail measure the loss targets, and it matches.
> **LOCKED (D60).** The thesis-final baseline θ is the **grid optimum** (the defensible headline;
> calm 7³ = 343 / stressed 5⁴ = 625 nodes, 3 seeds — `output/baseline_grid/`, full loss surface in
> `output/calibration_grid_{regime}.csv`), **cross-validated by the high-res surrogate**
> (160 LHS / 6 seeds, `output/baseline_hires/`) which lands on the same optimum — the
> "two methods, one answer" check passed. **calm:** ft_sigma_c=0.80, zi_alpha=0.34, zi_delta=0.05;
> **stressed:** ft_sigma_c=0.50, zi_alpha=0.26, zi_delta=0.05, p_zi=0.15; zi_mu pinned 0.025.
> Wired into `globals.CALIBRATED`. Report two caveats: (1) **D is comparable only at a fixed seed
> count** — the KS component's `s_KS` is sim-sized, so it rescales with n_runs (calm KS 16.8 at
> 2 seeds ≙ ≈28.6 at 6 seeds for the *same* fit; verified ×1.70 observed vs ×1.73 predicted);
> (2) several stressed parameters sit at *justified* bounds (ft_sigma_c at its 0.5 floor, zi_delta
> at the 0.05 book-stability floor, p_zi at 0.15), and zi_delta is a near-flat loss direction
> (calm D 43.8 at 0.05 vs 44.0 at 0.48) — benign, but disclose it.

### 4.2 Calibration methodology (Ch. 4)
**Surrogate-assisted simulated method of moments** (Gao et al. 2022 HFABM / XGB-Chiarella) —
*not* Bayesian optimisation. Surrogate = a single **XGBoost** regressor θ→D (fixed
hyperparameters, not tuned); stage-1 optimum by **Sobol pool argmin**; explore/exploit
**active-learning** rounds (2:1); **stage-2** grid refinement on the true simulator; held-out R²
as the surrogate-accuracy check. **Loss D** = five **Franke-standardised** components — KS
(distribution), V (return std), ACF1 (return ACF), ACF2 (|return| ACF, clustering), Hill (banded
tail index). Each moment distance is divided by its empirical **block-bootstrap** (Künsch 1989)
sampling SD, making components commensurable; they are summed (L1). This is inverse-SD weighting
(Franke-Westerhoff 2012) — *not* equal weights, *not* inverse-variance, *not* full Σ⁻¹.

### 4.3 Calibration robustness campaign (Ch. 5 — partly a NEGATIVE result, present it as a strength)
Eight configurations × 2 regimes were screened, then the promising ones re-run at higher
resolution and one re-confirmed again. Verdict:
- **The baseline is hard to beat.** Multi-parameter extensions did not robustly improve the fit.
- **E5** (FT/MT activation probability + per-order cancellation, 4 extra params) showed an
  apparent **−34% stressed win at screening (D 13.8 → 9.2)** that **did NOT replicate** at higher
  resolution (D → **16.9**, *worse* than baseline). The extra parameters are poorly identified →
  high run-to-run variance. Crucially, surrogate **R² ≈ 0.9+ did not guarantee a reproducible
  optimum** — replication across seeds is the real test. **Rejected.**
- The **combo** (E2+E5) and the **two-cohort / extra momentum traders** (E4) gave no robust gain
  → **rejected**. The two-cohort MT did **not** revive long-horizon clustering, confirming the
  single-timescale structural limit.
- The **only consistent improvement** was **E2** (`mt_lambda` calibrated in the loop): a small
  (~1 D), reliable *stressed* gain — the data mildly prefers a slower trend than the pinned 0.05;
  calm is neutral-to-slightly-worse. One well-identified extra parameter. *Optional to adopt.*
- **Framing:** a parsimony result — the extensions don't earn their parameters; over-
  parameterisation adds estimation *variance*, not fit. Present as a rigorous robustness study,
  and as a methodological caution (high surrogate R² ≠ reproducible optimum).

### 4.4 Clearing-tier mechanics & amounts (Ch. 6) — README "Clearing tier" / "Formulas" / "Parameters"
- **IM** = max(6%, z·σ_daily·√MPOR), z = 2.326 (99%), MPOR = 2 d → **~6% calm / ~12% stressed** (procyclical).
- **Capital ratio** = cash / IM, floored at **0.08** (CFTC Reg 1.17). Breach → BCM Almgren–Chriss
  deleverage; NBCM freeze (and its clients freeze).
- **Cover-2 default fund** = top-2 members' SLOIM = max(0, (stress_move − IM))·notional, ×1.10
  buffer (Euronext A9). Per-notional SLOIM rate ≈ **9% calm / ≈ 3.4% stressed** (calm is *higher*
  because the 15% stress-move floor dominates while IM sits at its 6% floor).
- **Five-level deficit-based waterfall:** L1 defaulter's own DF → L2 CCP skin-in-the-game →
  L3 pooled survivor DF → **L4 survivor cash pro-rata (mutualisation)** → L5 CCP cash.
- **Amounts:** CCP $7.5B; BCM $5–10B; NBCM $50M–$1B; clients FT $100–500M / MT $30–150M /
  ZI $60–150M. Contract $50/point (CME ES). VM hourly; DF recompute ~daily.

### 4.5 Contagion findings (Ch. 6 — headline results)
- **Containment at true COVID severity.** The cover-2 framework absorbs client defaults at the
  defaulter's own margin/DF layers — **no cross-member mutualisation, CCP solvent**. Under
  reverse-stress amplification, mutualisation (L3–L4) onsets only at **≈2.5× COVID (≈−41%)**.
- **Gaps vs a single shock (E6) — the clean structural finding.** At **equal total drawdown**, a
  single concentrated intraday shock is far more destructive than the real overnight-gapped path:
  ~**3× the client defaults** (≈29 vs ≈9), and it reaches CM default + L4 mutualisation at
  **c = 1 (−19%)** vs **c ≈ 2.5–3** for the gapped path. **Interpretation:** contagion severity
  tracks the *concentration / speed* of the move, not its total magnitude; the overnight-gap
  structure is a **mitigant** because variation margin collects incrementally between jumps,
  de-risking the book before the next gap, whereas a single shock hits at full exposure before VM
  can collect.
- **Clearing-in-loop sensitivity (E1) — a methodology caveat.** Calibrating with the clearing
  tier *active* leaves calm ≈ unchanged (no freezes fire) but materially alters stressed
  (D 13.8 → 28): freezes / deleverage / defaults during the crash feed back into the price the
  moments match. The stressed market-layer calibration is therefore **not clearing-invariant** —
  report this, and note clearing-active calibration as future work.

### 4.6 Fundamental & data handling (Ch. 3–4)
- **Fundamental** = Kalman-smoothed real ES mid (data-derived; XGB-Chiarella §2.5.2), with
  **overnight gaps retained** (D56) so the COVID episode carries its true ≈−33% drawdown. **RTH
  session boundaries are data-driven** (~405 bars, variable — D57), not a fixed 390. A synthetic
  Stein–Stein/Heston SV-jump fundamental is kept as a **robustness alternative** (notebook
  `FV_MODE` toggle): it reproduces the volatility regime but is a random path, not the real episode.
- **Order lifetime:** no blanket TTL (D58 removed the ODD §Mech #7 10-step ceiling); ZI cancels
  each resting order w.p. `zi_delta` (Cont-Stoikov-Talreja / Farmer), FT/MT replace-on-new.
  Deviation from the Simudyne ODD, flagged and cited.

## 5. Honest limitations (Ch. 7 — write candidly; several are strengths when framed as rigour)
- **Long-horizon volatility clustering** is not fully captured — a single momentum timescale
  can't produce multi-scale memory, and the two-cohort revival didn't fix it (a structural limit
  of this LOB).
- **Calm fit floors at D ≈ 44**, dominated by KS + |return|-ACF residuals — the simulated calm
  distribution body / clustering doesn't fully match empirical.
- **High-dimensional extensions are poorly identified** → add variance, not fit; the E5
  non-replication is the clearest case (the methodological lesson: surrogate R² ≠ reproducible
  optimum; replicate across seeds).
- **Contagion results are single-seed traces.** A Monte-Carlo ensemble over seeds (distributions
  of breach multiplier, default count, waterfall depth, mutualised loss) is future work.
- **Call-auction print convention:** crossings clear at the resting *ask* price (not a uniform
  clearing price or the midpoint), giving buy-initiated crossings price improvement — a minor
  asymmetry to disclose (defensible in a one-shot auction where maker/taker is ambiguous).
- **The stressed calibration is not clearing-invariant** (E1) — the bare-market θ understates the
  clearing-active stressed dynamics.

## 6. References & how each is used (priority order — every design choice must cite one; flag deviations)
1. **Simudyne CCP ODD** — clearing-tier scaffold and the ODD protocol the model description
   follows; source for margin cadence, waterfall structure, star topology. (Deviation: TTL
   removed, D58.)
2. **Deloitte / Simudyne CCP paper** — clearing design conventions (FT belief width
   `theta_v = sigma_fundamental`).
3. **Majewski et al. (extended Chiarella)** — FT value + MT momentum design; externally-fixed
   trend horizon (motivates pinning `mt_lambda`).
4. **Gao et al. HFABM / XGB-Chiarella (arXiv 2208.14207)** — surrogate-assisted SMM, the
   data-derived Kalman fundamental, the KS loss component, figure conventions.
5. **Gao 2023** — grid-search calibration as the cross-check method.
6. **ABM Liquidity (Vytelingum)** — liquidity / market-impact framing.
7. **Cont-Stoikov-Talreja (2008)** — ZI limit/market/cancel rates; the per-order cancellation
   that now governs order lifetime (replacing the TTL).
8. **Farmer et al. (ZI)** — zero-intelligence baseline.
- **Clearing regulation:** Euronext Clearing A9 (cover-2 SLOIM, reverse-stress), CFTC Reg 1.17
  (cash/IM net capital), CME SPAN + EMIR Art. 41 (VaR IM), EMIR Art. 45 (exchange SITG),
  Basel FRTB (BCM VaR house limit), CPMI-IOSCO 2017 (recovery), CME ES contract spec.

## 7. Literature, key empirical facts & data sources (lit-review)
*Depth = references + facts (the prose lit review is yours to write). §6 above is the
model-construction references; this section is the motivation / clearing-member literature.*

**Clearing-member / CCP systemic risk & client clearing.**
- **Menkveld & Vuillemey — "The Economics of Central Clearing"** (review): netting / insurance /
  fire-sale rationales for CCPs; states the *client-clearing market is an open question* (how CMs
  compete; account portability/pooling) — the gap this thesis targets.
- **OFR 2026 — "Clearing Markets and Client Clearing Services"** (DTCC CDS): clients = **73%** of
  margin at the largest US/EU CCPs; **modal client = single clearing agent**; Credit Suisse/Archegos
  → single-agent clients cut cleared positions. Motivation for single-agent dependency + client type.
- **OFR 2025 — "CCP Liquidity & Capital Demands on Clearing Members under Stress":** largest CMs can
  meet demands even in extreme scenarios, but sufficiency varies over time and correlated cross-CCP
  shocks stack demands.
- **Huang, Menkveld & Yu 2021 — "CCP Exposure in Stressed Markets"** (EMCF equity HF data): extreme
  exposure coincides with crowding + volatility; crowding ≈ **17%** of the top-100 exposure increases;
  top-5 members' share rises **28% → 42%** in the top-1% subsample.
- **Galbiati & Soramäki 2013 (BoE):** tiering *reduces* the CCP's total exposure but *raises* its
  expected single exposure to the average GCM (a trade-off the model's tiering speaks to).
- **Borovkova 2013 (network):** a CCP is not safer than bilateral for *all* nodes — depends on
  network position.
- **Riksbank (Blanck 2025):** margins/CCP/liquidity overview. **ESRB 2020:** Mar-2020 margin-call
  liquidity risk (facts below).

**Crowding & margin methodology.**
- **Menkveld 2014/2015 — "Crowded positions":** CrowdIx index; **Margin(A)** (delta-normal VaR of
  aggregate exposure, decomposes across members by shadow cost); crowding spikes with volatility.
- **Menkveld 2017:** social cost of crowding; some crowding is socially optimal; proposes a
  **Pigovian default-fund surcharge** on crowded traders.
- **Cruz Lopez, Harris, Hurlin & Pérignon 2017 — "CoMargin"** (CDCC data): CoVaR/copula conditional
  margin — margin a member on its breach probability *conditional* on others breaching; backtestable,
  no normality assumption (vs Margin(A)). The benchmark for RQ-C.
- **Duffie & Zhu 2011** (multilateral vs bilateral netting); **Jones & Pérignon 2013** (CME margin
  breaches cluster → systematic); **Menkveld, Pagnotta & Zoican 2015** (clearing → −8.8% vol, −9.8% volume).

**Market-microstructure ABM (construction — cross-ref §6).** Majewski, Ciliberti & Bouchaud 2018
(extended Chiarella; FV as a Kalman-filtered hidden state); Almgren & Chriss 2000 (optimal execution /
fire-sale impact); Gao et al. 2023 (Chiarella-Heston / deep hedging; volatility trader); Bookstaber,
Paddrik & Tivnan 2014 (ABM financial vulnerability — margin calls, fire sales, crowding); Farmer et al.
(zero-intelligence); Cont, Stoikov & Talreja 2008 (order-book dynamics); Vytelingum (ABM liquidity
risk; ZI + Almgren-Chriss); Lamperti et al. (ML-surrogate ABM calibration); Simudyne/Deloitte CCP model
(the clearing-tier reference implementation).

**Key empirical facts to cite (Mar-2020 / client clearing).**
- IM at the four largest EU/UK CCPs rose ≈ **one third**, mainly from **client** (not house)
  portfolios (ESRB 2020).
- Daily VM on euro-area fund derivative exposures **quintupled**; **6%** of funds lacked pre-stress
  liquidity to cover cumulative VM; for many, one day's VM exceeded their entire pre-crisis cash.
- CMs can apply **counterparty-specific IM add-ons up to 50%** and change terms at short notice;
  intraday VM is held overnight (timing asymmetry).
- Clients = **73%** of margin; **modal client = single agent** (OFR 2026).
- Crowding ≈ **17%** of worst exposure spikes; top-5 share 28→42% in stress (Huang et al. 2021).
- Nasdaq Clearing 2018 (Einar Aas): **€107M of a €166M** default fund tapped (near-miss).
- ABN AMRO Clearing: ≈ **$200M** loss on a single trade, Mar-2020 (Riksbank).

**Data sources** (★ = used by the built model; others = for empirical grounding/extension).
- ★ **CFTC FCM Financial Data** (free, monthly) — FCM adjusted net capital / segregated funds;
  grounds the BCM/NBCM cash ranges.
- ★ **ES E-mini 1-min futures data** (LSEG/Refinitiv-style) — the calm/stressed series the model
  calibrates to.
- ★ **CCP rulebooks / margin methodology** (CME SPAN, EMIR Art. 41/45, Eurex, Euronext A9, CFTC
  Reg 1.17, Basel FRTB, CPMI-IOSCO) — the stylised replica margin / DF / waterfall.
- **CPMI-IOSCO Public Quantitative Disclosures (PQDs)** (free, quarterly, ~25 CCPs) — CCP-level
  IM/VM/DF/SITG/concentration; *no member-level or intraday breakdown* (limitation).
- **ABN AMRO Clearing disclosures** (Pillar 3; public Correlation-Haircut methodology) — for the
  crowding / correlation-haircut angle (RQ-C); consolidated-bank level only.
- **WRDS** (OptionMetrics / Compustat / CRSP), **LSEG Workspace** — cross-asset / options for
  extensions. Paid aggregators (Risk Quantum, ClarusFT CCPView) exist.

## 8. Proposed thesis structure
1. **Introduction** — CCP systemic risk; the client-clearing tiering question; contributions; roadmap.
2. **Literature review** — ABM market microstructure (Chiarella, Cont-Stoikov, Farmer ZI); CCP /
   clearing risk & contagion; ABM calibration (SMM, surrogate methods).
3. **Model description (ODD)** — two-tier architecture; agents; the call-auction LOB; the
   fundamental process; the clearing tier (margins, DF, waterfall, fire-sales). [README "Market
   and agents", "Clearing tier", "Fundamental value process", "Formulas".]
4. **Data & calibration** — ES data, calm/stressed regimes, the Kalman fundamental + overnight
   gaps; the SMM loss + surrogate method; calibrated parameters + fit quality.
5. **Market-layer results** — stylised-facts reproduction (figures: `model_design.ipynb`,
   `empirical_analysis.ipynb`); the robustness campaign (§4.3) and the parsimony verdict.
6. **Clearing layer & contagion experiments** — margin/DF/waterfall behaviour; COVID containment
   + reverse-stress onset; gaps-vs-shock; clearing-in-loop sensitivity.
7. **Discussion & limitations** — contagion drivers; methodological findings; limitations (§5).
8. **Conclusion** — answers to the RQ; risk-policy implications; future work (Monte-Carlo
   ensembles, clearing-active calibration, multi-timescale momentum).

## 9. Writing conventions
- **Audience:** MSc examiners (quantitative finance / complex systems). Academic, precise,
  measured. No hype.
- **Be honest about limitations and negative results** — the E5 non-replication and the parsimony
  finding are *strengths* (rigorous robustness); present them as such, not as failures.
- **Numbers:** cite exact values from the results files; mark screening/interim numbers as such
  and don't over-state precision (screening D's are noisy). Final θ comes from the high-res
  baseline run (pending) — don't hard-code provisional θ as final.
- **Cite a reference for every modelling choice** (§6); flag deviations explicitly.
- **Hill, not kurtosis, is the tail measure.** If kurtosis appears, label it a diagnostic.
- **Figures / artefacts:** `model_design.ipynb` (design, clearing, stylised facts, intraday
  zooms, gaps-vs-shock); `empirical_analysis.ipynb` (empirical ES); `covid_contagion.py`
  (contagion traces); `output/campaign*/`, `output/e5_confirm/`, `output/baseline_hires/`
  (calibration tables).

## 10. Where things live (so the writing agent never needs another file)
- **Model mechanics, formulas, parameters, how-to-run** → `README.md`.
- **Findings, numbers, narrative, structure, references-usage, limitations** → *this file*.
- **Figures** → `model_design.ipynb`, `empirical_analysis.ipynb`. **Calibration results** →
  `output/campaign/`, `output/campaign_v2/`, `output/e5_confirm/`, `output/baseline_hires/`.
  **Contagion** → `covid_contagion.py`.
- **Full code decision history** (rarely needed for writing) → `AGENT.md`.
