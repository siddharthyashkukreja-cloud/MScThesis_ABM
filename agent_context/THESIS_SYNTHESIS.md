# Thesis synthesis — state, presentation, decisions, and hypothesis framing

*A strategic read of where the model and thesis stand, how to present the results academically,
what is still undecided, and whether the hypothesis framing is defensible.*

> **D77 update (2026-06-21).** Margin config changed: **MPOR 2→1** (CME 1-day futures horizon) and the
> **IM cap dropped** (`IM_CAP=None`; it was non-binding at the 1-day MPOR). The H2 procyclicality figure
> below (**3.0×, 4%→12%**) is the MPOR=2/cap run and **restates to ~2.8×, 4%→~11.4% (uncapped)**; all IM
> dollar figures (e.g. the $32.6B tiered) rescale **≈0.71×** in stress once the suite is re-run. H1/NET
> directions unchanged. See AGENT.md D77.

> **D75 update (2026-06-19).** Several specifics below are superseded by this session's changes:
> floors (**client-clearing solvency = CFTC Reg 1.17 cash/IM≥8% for both tiers**, plus a bank-only
> Basel/eSLR leverage-ratio deleverage trigger; NBCMs resized to $0.5–3B), C1 close-out fix, FHS +
> close-to-close margin, `VOLUME_LOT` 18/32. The framing is *strengthened*: **calm and actual-COVID
> are benign** (matching reality), H1 becomes a **mutualisation-threshold** result (tiered draws the
> mutualised fund in 0/40 seeds; direct reaches L≥3 in 80%) — report as waterfall-depth vs drawdown,
> not a fixed-severity count, disclosing the IM confound ($32.6B tiered vs $10.3B direct). H2
> procyclicality is **3.0×** (IM fraction 4%→12%, cap binds ~38% of stressed sessions; close-to-close,
> in line with CME's Mar-2020 ES hike); NET cuts posted client IM to **0.58×**. Member defaults are now
> **graded across both tiers** (the small-NBCM-only-fails bias is fixed). See
> `FLOOR_BALANCE_RESULTS.md`, `OPTION_A_RESULTS.md`, `FLOOR_AND_MARGIN_ANALYSIS.md`,
> `VOL_ESTIMATOR_AUDIT.md`, `VM_AND_VOLUME_NOTES.md`.

---

## 1. State of the model and thesis

**The model is essentially final and results-ready.** Two layers:

- **Market layer** — FT/MT/ZI agents on a limit-order book, fundamental `V_t` = the contemporaneous
  empirical 1-min ES BBO mid (the Kalman local-level MLE is kept only as the diagnostic that justifies
  using the raw mid; SV-MJD is the deferred robustness arm), calibrated *bare-market* (clearing off) to
  ES stylised facts over matched calm (2019) and stressed (COVID 2020) windows. The behavioural loop is
  now 7 parameters per regime, same set both regimes (`ft_sigma_c, zi_alpha, zi_delta, zi_mu, p_zi,
  mt_lambda, mt_gamma`); calibration is validated (moment-coverage ratio + Franke J-test). **The
  calibrated point values are placeholders pending a widened-bounds 7-parameter surrogate re-lock;
  because calibration is bare-market, none of this session's clearing changes touch the structure — the
  market layer remains a validated ES proxy.**
- **Clearing layer** — CCP, 10 BCM (5 with client books) + 5 NBCM, 90 clients; physical IM escrow,
  procyclical reactive IM, hourly VM, daily DF (cover-2), 5-level waterfall with IM seized first,
  porting, and close-out. This is where this session's work landed: a realistic balance-sheet
  rebalance, a leverage-balanced client→clearer assignment, and a full audit of *why the system is
  resilient*.

**What changed this session (and is committed):** client cash ↓ to realistic asset-manager scale
(FT $0.5–3B / MT $0.2–1B / ZI $0.2–0.5B), BCM proprietary cap 6×→2×, the client→clearer rule
replaced with a capacity-proportional (leverage-balanced) assignment, an EOD-liquidation bug fixed
in the direct-clearing arm, and the experiment window extended to sessions 10–30 (captures the
COVID trough). Net effect: **client share of posted IM ~70% (toward the CME ~82% reality), calm
clean, stress fires the leverage cycle.**

**Gaps to close (none are model-logic problems):**

1. **Experiment results are stale.** `output/thesis_final/experiments/` predates the rebalance,
   the balanced assignment, the bugfix, and the window extension. **Every headline number must be
   regenerated** under the committed config before it goes in the thesis.
2. **Docs have drifted.** `model.md` §2.2/§4 still say `POSITION_LIMIT_X = 6` and "round-robin, 9
   per member"; `clearing_writeups.tex` still carries the 6× cap and the old client sizes
   ($100–500M). `Bell & Holden 2018` (BIS) is missing from the `.bib`.
3. **Thesis prose is partial.** Intro / model / calibration drafts exist in `extras/tex_drafts/`;
   there is no master document and **no results / discussion / conclusion chapters**.

---

## 2. Remaining decisions — and my recommendation

| # | Decision | Options | Recommendation |
|---|----------|---------|----------------|
| D1 | **Headline framing** | (a) "CCP is resilient; tiering is the buffer" vs (b) "CCP cascades under stress" | **(a).** The evidence is overwhelming that the baseline does *not* cascade — claiming (b) would be incorrect. Frame resilience as the finding and fragility as *conditional*. |
| D2 | **Baseline close-out** | transfer @ 0.80 (synthetic) vs open-market CCP disposal | **Keep transfer @ 0.80** as the committed baseline (Lehman-anchored, liquid). |
| D3 | **Waterfall demonstration** | how to show L2–L5 ever activate | **Add three stress arms**, each empirically anchored: (i) **direct clearing** (removes the member buffer → SITG/survivor), (ii) **open-market CCP disposal, short horizon** (endogenous loss + price impact), (iii) **recovery 0.60 "disorderly close-out"** (→ pooled DF). Report draw *frequency* across the GBM ensemble. |
| D4 | **Client share 70% vs 82%** | push to 82% or keep ~70% | **Keep ~70%.** Full-notional leverage is what makes the client book a contagion channel; pushing to 82% needs a Basel client-clearing offset that would kill that channel. State the gap honestly (full-notional treatment + 15 clearers vs ~60 real FCMs). |
| D5 | **Recovery citation** | Lehman vs Aas | **Lehman → 0.80 baseline** (liquid, within IM); **Aas/Bell-Holden 2018 → 0.60 disorderly arm** (DF drawn). Do **not** use the Lehman 8.625% CDS recovery — that is a bond/credit-event recovery, a different concept. |
| D6 | **IM/VM/DF cadence** | already VM hourly / IM hourly / DF daily | **Leave as-is.** Tested: daily IM ≈ hourly IM (the 2-day-half-life EWMA already makes IM daily-ish), so cadence is not a lever. |
| D7 | **GBM ensemble role** | calibrated run vs scenario stress test | **Frame as a scenario stress test** (note it uses the primary mid-calibrated θ on SV-MJD paths; the `relock_gbm` θ exists if you want a fully self-consistent run). |
| D8 | **Disposal horizon (if D3-ii used)** | AC_HORIZON=30 vs short | **Shorten for disposal.** At 30 slices the CCP rides the path and can *profit* on a rebound — an artifact. A short disposal realises the close-out loss deterministically. |

**Critical-path actions:** regenerate experiments under the committed config (D1–D3 settled) →
sync `model.md` + `clearing_writeups.tex` + `.bib` (the stale items) → write results/discussion/
conclusion from the regenerated numbers.

---

## 3. How to present the results academically

**Narrative arc (recommended):**

1. **Validate the instrument first.** Lead the results with calibration: the market layer
   reproduces ES stylised facts (fat tails, vol clustering, ACF of |r|), with the moment-coverage
   ratio and Franke J-test p-value. This earns the right to use the model for counterfactuals — a
   reader will not trust the clearing results unless the price engine is credible.
2. **Then the three clearing experiments as the contribution**, each as: *hypothesis → design
   (arms, seeds, windows, observables) → results (table + figure) → interpretation.*
3. **Close with the stress/robustness section** (GBM ensemble + the fragility arms), which is where
   the waterfall is shown to activate, and tie it to the empirical record (Lehman vs Aas).

**Tables.** One per hypothesis, means ± dispersion across the 40 seeds, both regimes. Report:
drawdown, client defaults, member defaults, deepest waterfall level, fraction of seeds drawing
L≥2 / L≥3, mean & peak IM, client loss absorbed. *Report distributions, not point estimates* — the
waterfall is a tail event, so the **frequency** of L≥3 draws across seeds/paths is the right
statistic, not a single run.

**Figures.** (i) Stylised-fact panels (calibration); (ii) IM and aggregate leverage time series,
calm vs stress, showing procyclical IM rising and members deleveraging (H2); (iii) tiered-vs-direct
waterfall-depth bars (H1); (iv) a waterfall-depth *distribution* across the GBM ensemble + the
fragility arms; (v) the margin-share-by-agent-type and volume figures already produced.

**Presentation discipline (the thing most likely to be marked down if ignored):** present the
**resilience as a positive result**, quantified, not as a disappointment. "Under empirical-class
shocks the tiered CCP localises every loss at the defaulting member; the mutualised fund is reached
only when [conditions]" is a stronger, more honest claim than a manufactured blow-up.

---

## 4. Hypothesis framing — and is it correct?

The original one-liners ("H1 tiering contains contagion; H2 procyclicality trade-off; NET netting
helps twice") are directionally right but **need sharpening to match what the model actually shows.**
Refined, defensible statements:

**H1 — Intermediation tier as a loss-localising buffer.**
*Claim:* the client-clearing tier absorbs client defaults at the member's capital, so the mutualised
default fund is not reached under market stress; removing the tier (direct clearing) routes the same
defaults straight to SITG / pooled DF / survivors.
*Evidence:* tiered baseline draws **L0 across the entire GBM ensemble to −56%** (member capital
absorbs even 20 client defaults from a −20% gap); direct clearing on the same shocks reaches
**L2–L4**. ✔ **Correct and well-supported.** This is the cleanest, most quantifiable result and
should be the thesis's headline contribution.

**H2 — Margin procyclicality: collateral-demand spike + leverage cycle.**
*Claim:* the daily close-to-close IM rises sharply in stress (from the 4% floor to the 12% cap, a
3.0× rise; the cap binds ~38% of stressed sessions), draining member funding and forcing deleveraging
(the leverage-ratio breach), trading lower calm collateral cost against stress-time coverage; flat
margin is cheaper in calm but lets more client defaults through.
*Evidence:* procyclicality ratio 3.0× measured (IM fraction 4%→12%; in line with CME's ~1.9–2.6×
ES-margin hike in March 2020); members breach the leverage floor and deleverage in stress; flat-5% IM
raises client defaults to ~4.1/seed (vs reactive ~1.3, flat-12% ~0.65). ✔ **Correct as a
"collateral-demand / leverage-cycle" claim.**
⚠ **Caveat before claiming "amplification":** to say procyclical margin *amplifies the crash* you
must show a price/volatility feedback (deleverage selling → lower price → more breaches). The
deleverage and client fire-sales *do* hit the LOB, so the channel exists — but **measure it**
(clearing-on vs clearing-off, or reactive vs flat volatility) before using the word "amplification."
If the feedback is weak, frame H2 as collateral demand + leverage cycle, not amplification.

**NET — Gross vs net client margining.**
*Claim:* net (omnibus) margining cuts posted client IM to ~0.58× of gross (≈42% saving), improving
collateral efficiency but concentrating residual risk at the member.
*Evidence:* net IM ≈ 0.58× gross stressed ($19.0B vs $32.6B), 0.57× calm ($8.3B vs $14.8B); defaults
barely change. ✔ **Correct.** Keep it as a focused efficiency-vs-risk result; it is a supporting
finding, not a headline.

**Overall correctness verdict.** The framing is **correct and academically defensible *provided*
the thesis:** (1) presents the baseline as resilient and does **not** claim an endogenous default
cascade under COVID-class stress — that would misrepresent the model; (2) frames the waterfall/
mutualisation as **conditional** on removable structure (tiering), concentration/illiquidity (Aas),
or disorderly close-out, each anchored to the empirical record (Lehman = resilient/high recovery;
Aas = fragile/DF drawn, Bell & Holden 2018); (3) treats the GBM ensemble as a scenario stress test;
(4) substantiates any "amplification" language with a measured volatility feedback. Under those
disciplines the contribution is genuine and novel: **a calibrated, regulation-faithful ABM that
quantifies the buffer value of the intermediation tier and the conditions under which a CCP moves
from absorbing a shock to mutualising it.**

The one framing that would be **incorrect**: presenting H1/H2 as a demonstrated systemic blow-up of
the cleared market under normal or COVID stress. The model says the opposite, and the opposite is
the more interesting and more publishable result.
