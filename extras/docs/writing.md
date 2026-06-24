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

**RESOLVED (D66): the thesis runs TWO research questions, slotted into the lit review
(Overleaf `chapters/02_related_work.tex`).** RQ1/H1 — client clearing: does the tier absorb or
amplify stress vs direct clearing? (two-sided: CM capital absorbs at the centre; stop-out
freezes + failed porting transmit downward — descends from RQ-A). RQ2/H2 — margins: does
risk-sensitive IM amplify stress vs a through-the-cycle scheme (Glasserman-Wu anchored), and
at what calm-period collateral cost? (three-way: reactive / flat-12% TTC / flat-low —
descends from RQ-B). Both are paired same-path same-seed experiments on existing levers
(`direct=True`, `IM_MODE`), metrics already logged. H3 (collateral demand) stays an optional
add-on. The original candidates below are retained for the record:

**Research question — candidates (superseded by the resolution above):**
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
> **Both headline experiments are now implemented levers (D62):** H1 tiered-vs-direct
> (`build_clearing_tier(direct=True)` — identical client mechanics, only the loss-absorption
> structure changes) and H2 margin regimes (`IM_MODE`: reactive rolling-VaR / static / flat
> through-the-cycle), to be run as PAIRED comparisons (same path, same seeds) on the real COVID
> window + reverse stress, with SV-MJD episode ensembles (D63 generator) as robustness.
>
> **H2 REPORTING METRICS (D66b — BCBS-CPMI-IOSCO 2025 Phase-2 final report, the official
> yardsticks; compute per margin regime on the logged IM series):** (i) the Proposal-6
> responsiveness measure ΔIM% vs Δvolatility% over the crash window; (ii) margin
> peak-to-trough ratio; (iii) largest n-session IM increase ("sudden material increases",
> CCP resilience guidance). Also citable: CCPs manually overrode scan range / vol floor /
> MPOR / confidence interval in Mar-2020 and 2022 (legitimises IM_MODE + IM_MPOR_DAYS as
> policy levers), and Proposal 9's pass-through-vs-deviating client margins (legitimises the
> broker house-margin layer). Key: bcbs_cpmi_iosco_2025 (already in the Overleaf bib).
>
> **H3 candidate (D66 — instrumentation-only, recommended): SYSTEM COLLATERAL DEMAND**, à la
> Duffie-Scheicher-Vuillemey 2015. Measure aggregate collateral demand and its decomposition —
> client IM vs house IM vs DF prefunding/replenishment vs gross VM churn — across
> {tiered, direct} × {reactive, static, flat} × {MPOR 1d, 2d}. The MPOR knob already exists
> (`globals.IM_MPOR_DAYS`, read at call time): EMIR Art. 26 sets 2 days for ETD but the US rule
> for futures (CFTC 39.13(g)(2)(ii)) is 1 day — varying it converts a reviewer objection into a
> result (IM scales by √2 mechanically; the DEFAULT effect is non-linear because thinner margin
> exhausts client cash earlier). DSV hooks: customers post 75.5% of system margin in their
> baseline; clearing LOWERS demand given dealer-to-dealer IM; their two novel components
> (precautionary VM buffer = κ·std(net VM), and 1-day "velocity drag" on received margin) are
> NOT in this model — the buffer is a one-parameter, DSV-citable optional extension (idle cash
> → earlier freezes), velocity drag is future work (needs payment timing; pairs with the
> Paddrik-Young positioning).
>
> **Netting levers (measured, D65-era):** CCP margining is GROSS per US rules (CFTC
> 39.13(g)(8)(i): customer accounts margined gross; house/client segregated — the model is
> regulation-faithful for CME ES, so tiering carries NO CCP-margin netting benefit and H1
> isolates pure loss absorption). Measured member netting ratios |net|/gross run 0.01–0.73 —
> the DF (sized on |net|) is where netting bites. An EU-style NET-OMNIBUS variant is a one-line
> change (`client_notional` → |signed sum|) cutting client-CM IM by 27–99% at those ratios — a
> clean "netting efficiency vs mutualisation" experiment if a third structural lever is wanted.
> Disclose: SLOIM uses (stress − im)·|net| (same-basis-net, internally coherent); the
> per-account Euronext convention would give (stress − im)·gross, a 1.4–100× larger fund.
> Also post-relock: relabel `mm_percent` 0.95 (CME's convention is IM = 110% of maintenance,
> ratio ≈ 0.909; flag-only constant).

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
> **LOCKED (D65) — THE thesis-final baseline θ** (grid headline of the recomposed loop;
> `output/relock/`): **calm** ft_sigma_c=0.95, zi_alpha=0.26, zi_delta=0.02, zi_mu=0.0125
> (p_zi L2-pinned 0.543), **D_grid 43.93**; **stressed** ft_sigma_c=0.25, zi_alpha=0.32,
> zi_delta=0.02, p_zi=0.18, zi_mu=0.0583, **D_grid 5.76** (−27% vs D60). Wired into
> `globals.CALIBRATED`. **Validation chain (cite all four):** (1) fresh-common-seed re-rank
> keeps both optima #1 (calm 46.49 vs 46.96 runner-up; stressed 5.76→5.93, no winner's-curse
> inflation); (2) surrogate cross-check agrees on every identified stressed lever (ft low
> 0.34, zi_mu 0.12, p_zi 0.13; R² 0.84/0.95), disagreeing only on near-flat zi_delta;
> (3) beyond-floor probes: D RISES below the stressed ft box floor (0.25:6.29 → 0.15:6.86)
> with the surrogate interior at 0.344 — node-sitting is discretisation of a 0.25–0.35
> basin, NOT clipping; (4) stressed Hill 3.03 vs target 3.18, ret_std within 4%, KS 0.024.
> Calm remains the honest plateau (D≈44 floor, weakly identified; ks 0.145 / acf_absr_1
> 0.227 vs 0.296 — the documented body/clustering residual). zi_mu's re-inclusion: the C6
> "freely droppable" verdict was stale post-D58. D comparable only at fixed n_runs.
>
> **SUPERSEDED (D60 record, archived in output/baseline_grid/):** the previous baseline θ was the **grid optimum** (the defensible headline;
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
>
> **FRESH-SEED VALIDATED (D64 completion run) — cite these to close the selection-bias
> objection:** re-ranking the top-5 grid nodes on two fresh COMMON seed sets keeps the locked θ
> at **#1 in both regimes** — calm D_fresh 46.66 (runner-up 48.11), stressed D_fresh **7.15**,
> *below* its in-sample 7.92 (zero winner's-curse inflation). Source:
> `output/baseline_grid/grid_freshseed_{calm,stressed}.json`.
>
> **SV-MJD (scenario-generator) θ — RE-LOCKED on the D65 loop (`output/relock_gbm/`,
> completed 2026-06-12; supersedes the D64 `baseline_gbm` record below):** the stressed
> **ZI rates are invariant to the choice of fundamental** — grid argmin zi_alpha=0.32,
> zi_delta=0.02, zi_mu=0.0583 match the Kalman lock exactly — but the two *identified* levers
> shift on the synthetic path: grid ft_sigma_c=0.45 (vs 0.25 Kalman), p_zi=0.08 (vs 0.18),
> D_grid 7.35 (fresh-seed #1, D_fresh 8.51; fit worse than Kalman 5.76, as ablation C1
> predicts). The shift is largely **weak identification, not a contradiction**: the stressed
> *surrogate* interior (ft_sigma_c=0.32, p_zi=0.15, R²=0.89) sits between the two grid optima,
> so the synthetic-path stressed optimum lives in the same ft 0.25–0.45 / p_zi 0.08–0.18
> neighbourhood, only flatter. **Frame the robustness headline honestly as "same ZI rates;
> the tail/depth levers land in the same basin", NOT "identical optimum"** — the D64
> "identical" coincidence was an artefact of the old D60 loop (then both stressed optima were
> 0.50/0.26/0.05/0.15) and does not survive the recomposed D65 loop. Calm gbm stays
> **near-flat**: grid argmin {1.25, 0.26, 0.02, zi_mu 0.0812} (D 36.61, = fresh-seed #1,
> D_fresh 37.60) vs surrogate {0.98, 0.34, 0.09, zi_mu 0.10} (weak R² 0.78) — use the
> fresh-seed winner for calm gbm ensembles and disclose the flatness. Do NOT compare D across
> fundamentals or seed counts (s_KS caveat). `globals.CALIBRATED` stays the **Kalman** lock;
> the SV-MJD θ is robustness-only and is not wired into the simulator.
>
> **SUPERSEDED (D64 `baseline_gbm` record):** stressed optimum was then **identical to the
> Kalman lock** (ft_sigma_c=0.50, zi_alpha=0.26, zi_delta=0.05, p_zi=0.15; D_fresh 11.44;
> surrogate R²=0.95); calm near-flat (grid {1.25, 0.26, 0.125} D 35.91, fresh winner
> {0.50, 0.50, 0.05} 39.22). Kept only as the pre-relock record.

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

> **D65-θ + D67-fix BASELINES (current; seed 42, reactive IM — the numbers to build the
> ensemble around; single-seed until the Monte-Carlo driver runs). D67: the conservation
> audit found assumption transfers (client→CM, book→CCP) bypassed fill settlement, retro-
> charging the assumer for a move the defaulter had already paid — a phantom double-charge
> (−$14.1B in a forced cascade). Fixed by recording assumptions as fills at the default-
> cycle mid; total system cash now conserves to float precision through L4 cascades in
> BOTH modes (the ledger-closure certificate for the whole clearing layer).**
> **TIERED:** COVID c=1 contained — 3 client defaults (all MT, shortfalls ≤$0.4M,
> CM-absorbed), L0, DF $0.63B. Reverse stress: client defaults 3→50 across c=1→4; FIRST
> member defaults only at **c=4** (3 NBCMs, absorbed at **L2/SITG**) — **the pooled DF is
> never touched through 4× COVID (−56%)**.
> **DIRECT (same paths/seeds):** client deficits hit the CCP from c=1 (L2); pooled-DF
> mutualisation at **c=2.5 (L3)**; survivor-cash assessment from **c=3 (L4)**.
> **H1 first-pass headline: the client tier buys ≥1.5× COVID of mutualisation headroom and
> ≥1× of assessment headroom** (direct: L3 at 2.5×, L4 at 3×; tiered: L2 first touched at
> 4×), at similar client-default counts (46 vs 50 at c=4) — H1's buffer clause confirmed at
> the centre; the client-level clause (freeze channel, porting) is the ensemble's job.
> Marginal defaulters are MT (CTA) clients. E6 re-run pending at D65 θ. **H2 regimes**
> ready; the reactive DF scales strongly with amplified vol (c=4 DF $5.0B tiered) — itself
> an H2 procyclicality observable.
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
- **Calm fit floors at D ≈ 44 — but ~41% of that is a WINDOW-SELECTION artifact, not agent
  misfit (D67 diagnostic; report this).** The calm sim always runs the FIRST 20 sessions of
  2019 (post-Dec-2018 selloff: fundamental vol 1.72× the year average) yet is scored against
  FULL-YEAR targets. Scored against its own window's targets (same sims, same conservative
  full-year weights): **D = 26.5 vs 45.2**, with ret_std within 5% (3.98e-4 vs 4.20e-4) and
  |r|-ACF lag-1 dead on (0.225 vs 0.223) — the headline V and clustering "misses" are mostly
  the Jan–Feb-vs-year gap. The genuine residuals are the KS body shape (0.149) and a partial
  Hill gap. Also explains the calm plateau: the dominant components (dKS ≥ 11, dACF2 ≥ 4.3
  across the ENTIRE box) are θ-insensitive floors, so many corners reach ≈44 via
  compensating trade-offs. Improvement (future work, would need a re-lock): stride calm
  `v_start` across seeds so pooled sims sample the year.
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
  (cash/IM net capital), CME SPAN + EMIR Art. 41 (VaR IM), EMIR Art. 45 (exchange SITG —
  NB the actual rule is ≥25% of the CCP's minimum regulatory capital, not a DF share; the
  model's 10%-of-DF is a stylised assumption, disclose), **EMIR Art. 48(5)–(6)** (client
  porting / liquidation on CM default — the D61 porting mechanism), Basel FRTB (BCM VaR house
  limit — strictly a capital framework; the 5% desk limit is prop-desk practice), CPMI-IOSCO
  2017 (recovery), CME ES contract spec, **EMIR Art. 28 RTS APC options + BCBS-CPMI-IOSCO 2022
  margin review** (the H2 margin-regime comparators).
- **FV-process measurement (D63):** Barndorff-Nielsen & Shephard 2004/2006 (bipower variation),
  Mancini 2009 (threshold jump estimator), Huang & Tauchen 2005 (~5–7% S&P jump share
  benchmark), Lou, Polk & Skouras 2019 (overnight vs intraday decomposition; the gap pool).
- **H1 framing:** Duffie & Zhu 2011, Galbiati & Soramäki 2013 (tiering trade-off), OFR 2026
  (single-agent dependency — motivates capacity-checked porting).

## 7. Literature, key empirical facts & data sources (lit-review)
*Depth = references + facts (the prose lit review is yours to write). §6 above is the
model-construction references; this section is the motivation / clearing-member literature.*

**Clearing-member / CCP systemic risk & client clearing.**
- **Paddrik, Rajan & Young 2020 (Mgmt Sci, "Contagion in Derivatives Markets"; + OFR WP 16-12,
  17-06; Paddrik & Young 2021 OFR WP 21-02 "Assessing the Safety of CCPs")** — THE closest
  modelling cousin: variation-margin payment-shortfall cascades on the full DTCC CDS network
  (~900 firms), with "soft default" (delayed/partial payment) responses; finds the CCP
  contributes LESS to contagion than large peripheral net sellers. Differentiate the thesis:
  their network is fixed with exogenous shocks — here price formation is endogenous (LOB), the
  fire-sale feedback loops through the price, and the CLIENT tier exists. Cite in lit review +
  position the contribution against it.
- **Glasserman & Wu 2018 (Mgmt Sci, "Persistence and Procyclicality in Margin Requirements")**
  — the theoretical anchor for H2: with persistent/bursty volatility, the anti-procyclicality
  buffer needed corresponds to the UNCONDITIONAL quantile of price changes — i.e. the H2
  "flat-at-stressed-level" (through-the-cycle) comparator is exactly their stable-margin
  benchmark; the reactive EWMA-VaR regime is their risk-sensitive one. Frame H2's
  liquidity-vs-procyclicality trade-off with this + BCBS-CPMI-IOSCO 2022 margin review.
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
- **Duffie, Scheicher & Vuillemey 2015 (JFE, "Central Clearing and Collateral Demand")** — the
  H3 anchor: system collateral-demand decomposition (customer IM 75.5% of system margin in the
  baseline; d2d IM raises demand +69.7%; GIVEN d2d IM, mandatory clearing LOWERS demand absent
  CCP proliferation; client clearing's distributional effects depend on the cleared share; the
  precautionary-buffer and velocity-drag components were novel). Their IM is the same VaR-quantile
  family as the model's (99%/5-day for CDS = OTC convention vs 99%/2-day ETD here — correct per
  product class, RTS 153/2013 Art. 24/26).

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
  and don't over-state precision (screening D's are noisy). Final θ is the **D65 re-lock grid
  headline** (`output/relock/`), wired into `globals.CALIBRATED` — cite that as final; the D60
  record (`output/baseline_grid/`) is archived/superseded, not provisional-to-be-replaced.
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
