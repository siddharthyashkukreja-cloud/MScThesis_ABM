# Second-reader review — CCP-cleared ES ABM

*An adversarial verification of the model, code, design decisions, and headline results, in the
role of a critical thesis second reader. Every code claim is `file:line`; every number is measured
this session (sandbox, committed config from `model/globals.py`, real COVID window). Where a doc
and the code disagree, the code is treated as ground truth.*

Date: 2026-06-18. Reviewed at git working-tree state (not HEAD — see H2).

---

## 0. Verdict

This is a strong, unusually self-critical piece of work. The market layer is genuinely calibrated
and the calibration chapter (per `CALIBRATION_VALIDATION.md`) already pre-empts most of the
objections an examiner would raise. The clearing tier is regulation-literate and the design log is
honest about its own simplifications. **The framing discipline in `THESIS_SYNTHESIS.md` is correct:
present the baseline as resilient, mutualisation as conditional.**

I found **one issue that materially affects a headline result** (a cash/inventory non-conservation
in the committed close-out mode, which contaminates the H1 *direct* arm), **a serious process risk**
(the entire clearing model is uncommitted to git), and **a cluster of citation/labelling
imprecisions** on the clearing-tier constants that a clearing-literate examiner will catch. None of
these sink the thesis; the central H1 contribution survives once the close-out is made
conservation-safe (I verified this). But the committed experiment numbers must not go into the
thesis as they stand.

The single most important sentence in this review: **the qualitative H1 finding (direct clearing
reaches mutualisation, tiered does not) is robust, but the committed `transfer` close-out breaks
value conservation, so the direct-arm magnitudes — and the "0/6 vs 4/6" frequencies — are not
trustworthy until the close-out is fixed and the suite re-run.**

---

## 1. What I verified as sound (credit where due)

These I checked against code and/or re-ran, and they hold:

- **Calm is clean.** Tiered and direct, 0 client/member defaults, no deleverage, waterfall 0,
  cash and inventory conserved to float precision (measured).
- **Client IM share ≈ 75%.** Measured 75.4% calm / 71.9% stressed — matches the "~75% toward the
  CME ~82%" claim. CME's gross customer margining and house/customer segregation (which the model
  mirrors via `CLIENT_MARGIN_NETTING="gross"`) are confirmed against CME disclosure.
- **Stress fires the leverage cycle.** BCMs breach the 8% floor (κ down to 0.04) and deleverage but
  survive (0 member defaults at actual COVID) — the H1 mechanism.
- **Procyclicality ratio ≈ 1.43×** (stress/calm posted IM) — inside the claimed 1.4–1.6×.
- **The LOB** (`lob.py`) is a clean price-time-priority call auction; no look-ahead; overnight
  reprice preserves book shape. The print-at-resting-ask convention is disclosed (`model.md` §7).
- **Tiered baseline conserves** cash and inventory through client-default cascades (firesale client
  close-out transfers the book to the CM, then sells it back into the LOB — Σpos stays 0).
- **The calibration engine** is paper-faithful (verified independently in `CALIBRATION_VALIDATION.md`:
  KS+V+ACF1+ACF2+Hill, inverse-SD block-bootstrap weights, dual grid+surrogate, fresh-seed re-rank).
  The honest identifiability disclosure (only `zi_alpha` sharply identified) is exactly right.
- **The IM/DF math** (`globals.im_fraction`, `df_stress_move`): calm IM 6.0% (floor binds), stressed
  8.0%, DF stress move 10.4% — all behave as documented.
- **`_seize_im` and the DF-pool ledger** conserve cash internally (IM excess returns to the estate;
  DF pool = Σ contributions).

---

## 2. Findings, severity-ranked

| # | Severity | Finding | Affects |
|---|----------|---------|---------|
| **C1** | **Critical** | `transfer` close-out flattens a defaulted book without reassigning it → cash + inventory non-conservation (measured: direct arm **−$14.8B cash, −7,427 lot inventory drift**). `model.md:355` "conserved … in both tiered and direct modes" is false for the committed config. | H1 direct arm; any member default; the conservation claim |
| **H1** | High | Entire D61–D74 clearing model is **uncommitted** (git HEAD = "D60"); `gbm_lever.py`/`run_thesis_experiments.py` untracked. No reproducible snapshot of the thesis model. | reproducibility |
| **H2** | High | The 8% `cash/exposure` floor is labelled "Basel III leverage ratio" (real min **3%**, 5% G-SIB) and cross-cited to CFTC Reg 1.17 (which is 8% of *IM*, not notional). It is effectively a tuned structural knob. | clearing realism, defensibility |
| **H3** | High | 15% house margin justified via the **security-futures** statutory minimum (17 CFR 242.400-406) — but ES is a **broad-based index future**, not a security future (SPAN ~5–7%). Mis-applied regulation. | calibration of the client cap |
| **H4** | High | Headline numbers are **stale and from a superseded driver**. `model.md` §6 cites `run_experiments.py` / `output/experiments/`; the committed driver is `run_thesis_experiments.py`. Must regenerate all numbers under the committed config. | every quoted result |
| **M1** | Medium | Thin/asymmetric ensembles: `gbm_lever` = 6 paths × **1** agent seed; `run_thesis` = 40 agent seeds × **1** path. "0/6, 4/6" is a 6-sample, single-agent-seed statistic. | H1/recovery frequencies |
| **M2** | Medium | H1 direct cover-2 fund is sized to the largest **clients** (small) vs the largest **members** (big) in tiered — "direct draws DF more" is partly mechanical (smaller fund), not only "no member buffer." | H1 interpretation |
| **M3** | Medium | H2 forced-deleverage/amplification channel is **structurally weak**: a member can shed only its small own book, not the client book causing the breach → freeze-dominated, little price feedback. | H2 "amplification" claim |
| **M4** | Medium | Waterfall L4 has **no assessment cap and no recovery tools** (VMGH/tear-up); it drains unlimited survivor cash, then CCP cash. Not regulation-faithful on assessment caps. | waterfall realism |
| **M5** | Medium | Stressed calibration is **not clearing-invariant** (acknowledged): the experimental runs are not the calibrated object — clearing order flow perturbs the microstructure that was fit bare-market. | validity of contagion runs |
| **L1–L6** | Low | Doc/code drift (AGENT.md D74 deleverage contradiction; stale `capital_ratio` docstrings; dead `CAP_RATIO_MODE`; "~70%" vs "~75%"; §3.3 "rarely triggered"; "37 defaults" is a 2-seed sum). | hygiene |

---

## 3. Critical finding — close-out conservation (C1) — ✅ FIXED 2026-06-18

> **Status:** fixed. `transfer` close-out now reassigns the defaulted book to the largest-opposing
> survivor (`_transfer_book` / `_ccp_warehouse` in `simulation.py`), so inventory is conserved (Σinv
> = 0) and the direct-arm COVID leak fell from −$14.8B to −$0.30B (the genuine haircut). See
> `OPTION_A_RESULTS.md` §3. The original diagnosis is retained below for the record.

### Evidence

Committed config (`CLOSEOUT_MODE="transfer"`, `CLIENT_CLOSEOUT="firesale"`, `IM_ESCROW=True`), real
COVID window sessions 10–30, seed 42, −25.6% drawdown:

```
TIERED  : 26 client def, 0 member def, waterfall 0 | inventory drift   0 | Δcash    0.0B
DIRECT  : 17 client def, waterfall L4             | inventory drift -7427 | Δcash  -14.8B
```

`Σ(all agent inventory)` must stay 0 (fills conserve it). In the direct arm it drifts to −7,427
lots, and total system cash (`Σ agent.cash + ccp.cash + ccp.df_cash + ccp.im_account`) swings
−$14.8B — far beyond any plausible close-out haircut.

### Mechanism

`simulation.py` Phase 0 (direct client, `:426-433`) and Phase 2 (member, `:596-605`) in `transfer`
mode do `client.inventory = 0` / `cm.inventory = 0` **without assigning the position to any
surviving participant**. `model.md:336-338` and the code comment admit this ("sets the defaulter's
book flat … no specific surviving member receives the position"). But once a non-zero position is
deleted from one side, `Σpos ≠ 0`, so the VM identity `Σ pos·Δmid = 0` breaks and every subsequent
margin cycle leaks `Δmid · Σpos` of cash among the survivors. `model.md:355`'s "conserved to float
precision through L4 cascades in both tiered and direct modes" is therefore **false for the
committed transfer mode** — it held for the firesale-era audit (D61) and for runs with no member
default, which is what was actually checked.

### It is isolated and the fix is clean (verified)

Re-running the **same** direct scenario with `CLOSEOUT_MODE="firesale"` (the position is assumed and
sold back into the LOB):

```
DIRECT + firesale : 19 client def, waterfall L4 | inventory drift 0 | Δcash ~0.0B
```

Conservation is restored **and the waterfall still reaches L4.** So:

1. The bug is exactly the flatten-without-transfer in `transfer` mode.
2. **The qualitative H1 result is robust** — direct clearing reaches mutualisation even when the
   model conserves value.
3. The committed *magnitudes* (ccp_cash_used, survivor cash drawn, exact waterfall frequency) in
   the direct arm — and any tiered stress arm that produces a member default (`rec06`, `flat03`,
   `direct`, etc.) — are contaminated.

### Recommendation

Make `transfer` actually transfer. Two options:

- **(a) Minimal, faithful.** In `transfer` mode, reassign the defaulter's net book to the largest
  opposing surviving participant's `inventory` / `client_positions` (the PositionAuction the comment
  already describes) instead of zeroing it. The (1−recovery) haircut stays the loss; conservation
  holds. This is the correct fix and makes the code match `model.md`/`clearing_writeups.tex`.
- **(b) Pragmatic for H1.** In the **direct** arm, close out direct clients via `firesale` (their
  books are small — open-market execution is realistic, exactly as for tiered clients). This already
  conserves (measured) and is more realistic than a synthetic transfer for a small book. Keep
  `transfer` only for whole-member books (rare; option (a) for those).

Then re-run H1/recovery and re-derive every number. Also fix the `model.md:355` claim to scope the
conservation guarantee to firesale/no-member-default, or (after the fix) re-audit and keep it.

---

## 4. High-severity findings

### H1 — the model is not in version control

`git status` shows `M model/globals.py agents.py simulation.py clearing.py lob.py run_simulation.py
calibrate.py` and `?? gbm_lever.py run_thesis_experiments.py`; HEAD is the "D60 baseline lock"
commit. The diff vs HEAD is +959/−679 lines in `model/` alone. **Every clearing mechanism the
thesis describes (escrow, waterfall, porting, the rebalance) exists only in the working tree.** One
`git checkout` or disk loss erases D61–D74, and there is no tag matching the committed `output/`
ensembles. Commit the working tree (and tag the exact state the thesis numbers are generated from)
before doing anything else. This is cheap and it is the highest-expected-loss item here.

### H2 — the 8% leverage floor is mis-cited and effectively tuned

`capital_ratio()` returns `cash/exposure` under escrow (`agents.py:403-404, 466-467`), floored at
0.08 (`CCP_CALIBRATION["cap_ratio_floor"]`). `clearing_writeups.tex` Eq. (1) and `model.md:290` call
this "the Basel III leverage ratio." The Basel III leverage ratio minimum is **3%** (5% for G-SIBs),
not 8%. The "8%" originates in CFTC Reg 1.17 — but that is 8% of *risk margin (IM)*, a much smaller
base, not 8% of notional exposure. So the floor is a **hybrid**: a Reg-1.17 number applied to a
Basel-style base, at ~2.7× the Basel minimum.

This matters because the floor is **load-bearing and effectively calibrated**: the whole
"calm-clean / stress-breach" behaviour depends on it jointly with the client/member cash sizing
(both tuned, per the `globals.py` comments: cash bands "tuned so the 8% floor freezes clients …
under stress but not in calm"). That is legitimate ABM practice, but it should be presented
honestly: a **chosen leverage limit**, not a hard regulatory constant, with a **sensitivity sweep**
over the floor (e.g. 3%/5%/8%/10%) showing the qualitative results survive. Re-cite as "a leverage
ratio in the spirit of Basel III / the FCM net-capital regime, set to X% as a buffer above the 3%
statutory minimum," and stop calling 8% "the Basel III leverage ratio."

### H3 — ES is not a security future; the 15% citation is mis-applied

`globals.py` and `clearing_writeups.tex` justify `im_percent=0.15` as "the post-2020 security-futures
statutory minimum (17 CFR 242.400-406; 20%→15% in 2020)." But the E-mini S&P 500 is a **broad-based
stock-index future**, regulated solely by the CFTC and margined by SPAN (~3–12%, typically ~5–7%);
the SEC/CFTC security-futures margin rules apply to **single-stock and narrow-based-index** futures,
not ES. The LaTeX already hedges ("sits well above the SPAN minimum … the add-on an FCM imposes"),
so the *role* (a house add-on above the exchange floor) is fine — but the **primary citation is the
wrong regulation for this product.** Re-justify 15% as a representative FCM house-margin add-on (cite
NFA/FCM practice, which you already do secondarily) and drop or heavily caveat the security-futures
statutory basis. A clearing-literate examiner will catch this immediately.

### H4 — regenerate the headline numbers; fix the driver citation

`model.md` §6 says "Numbers below are from `output/experiments/` (driver: `run_experiments.py`)."
`AGENT.md` itself flags `run_experiments.py` as the **superseded** pre-rebalance path and the
committed `output/` ensembles as stale. So the §6 numbers (including 6.5/6.6, which use the old
`covid_contagion.py` reverse-stress `c`-amplifier and seed-fraction language) predate escrow + the
rebalance + the 15% margin. Regenerate everything with `run_thesis_experiments.py` + the
(conservation-fixed) `gbm_lever.py`, and repoint §6's provenance line. This is already "Open
decision (a)" — but it is gating: no number currently in the docs should be quoted.

---

## 5. Medium-severity findings

**M1 — ensemble design.** `gbm_lever.py` fixes `ASEED=42` and varies only `GSEEDS` (6 fundamental
paths); `run_thesis_experiments.py` varies 40 agent seeds but a single realised COVID path. Neither
samples *both* path and agent randomness. The "0 of 6 / 4 of 6" headline is therefore a 6-sample,
one-agent-seed frequency — too thin to state as a probability. Recommend a grid of ≥5 agent seeds ×
≥20 GBM paths and report the draw frequency with a CI. (Also: the per-seed GBM files
`data/fv_gbm_stressed_s*.csv` are not present — regenerate via `data/v_gbm.py` before re-running.)

**M2 — H1 is partly mechanical.** In `recompute_default_fund` the cover-2 fund is sized on the two
largest *members* (tiered) vs the two largest *direct clients* (direct) — and clients are far
smaller. So the direct fund is structurally smaller and easier to exhaust. "Direct draws the DF more
often" thus conflates (i) no member buffer with (ii) a smaller fund. Arguably correct (in direct
clearing the fund *is* sized to the participants), but say so explicitly so the result is not
over-read as a pure buffer effect.

**M3 — "amplification" is weak by construction.** On a floor breach the member can only
Almgren-Chriss its *own* book (`simulation.py:634-639`), which is capped at 2×cash and is usually
small; the client book that actually caused the breach cannot be shed intraday, so the member sheds
a little and **freezes**. The forced-selling price-feedback (Thurner/Aymanns-Farmer) is therefore
structurally muted. This *supports* the `THESIS_SYNTHESIS.md` H2 caution: do not claim
"amplification" without measuring a volatility feedback — and expect it to be small. Frame H2 as
collateral-demand + leverage-cycle (funding channel), which the model does show.

**M4 — waterfall L4 is uncapped and skips recovery tools.** `default_waterfall` (`clearing.py:247-255`)
draws `min(loss, Σ survivor cash)` with no per-member assessment cap and no VMGH/tear-up before
L5 (CCP cash). Real waterfalls cap assessments (typically ~1× DF contribution) then invoke recovery
tools. This overstates mutualisation capacity (CCP rarely insolvent) while making member-to-member
contagion unbounded. Disclose as a simplification; optionally add an assessment cap as a lever.

**M5 — calibration is bare-market; the runs are not.** θ is fit with clearing off; the contagion
runs add deleverage/close-out order flow that was never calibrated (and `AGENT.md` E1 shows the
stressed loss is not clearing-invariant). The "calibrated ES proxy" claim is valid for the price
engine, but state plainly that the experimental object is the price engine *plus* uncalibrated
clearing flow. Already in `model.md` §7 — keep it prominent.

---

## 6. Low-severity / hygiene (doc & code drift)

- **L1 — AGENT.md self-contradiction.** The D74 change-log entry says "deleverage buffer retired
  under escrow … a BCM … freezes like an NBCM." The code (`simulation.py:634-639`), `model.md` §3.3,
  `clearing_writeups.tex`, and AGENT.md's own "Committed model state" header all say the BCM
  **deleverages** toward `CAP_DELEVERAGE_TARGET=0.10` *and* freezes. Code wins (it deleverages
  regardless of `IM_ESCROW`). Fix the D74 entry.
- **L2 — stale `capital_ratio` docstrings.** `agents.py:391-397, 454-460` describe "adjusted net
  capital / initial margin … cash/IM ≥ 0.08" but the function returns `cash/exposure` under escrow.
- **L3 — dead flag.** `CAP_RATIO_MODE="im"` is committed but unreachable under `IM_ESCROW=True`
  (the escrow branch returns first). Harmless but fragile — a future `IM_ESCROW=False` would
  silently revert to the legacy ratio. Assert the combination or remove the dead path.
- **L4 — number drift.** Client-IM-share quoted as "~70%" (`globals.py:213`, `THESIS_SYNTHESIS`)
  and "~75%" (prompt, `model.md` §6); measured ~75% calm. `_client_cap_qty` comment says
  "im_percent = 20%, i.e. 5×" (`agents.py:84-85`) and `FT_CLIENT_CASH` says "exposure ≤ 5× cash"
  (`globals.py:205`) — both are the old 20%; committed is 15% (6.67×). `THESIS_SYNTHESIS` §1 says
  `model.md` "still says POSITION_LIMIT_X = 6 / round-robin" — but `model.md` is actually already
  updated to 2× and the balanced assignment, so that "gap to close" is itself stale.
- **L5 — `model.md` §3.3 vs §6.** §3.3 says the BCM deleverage is "rarely triggered at actual-COVID
  severity"; §6 (and my run: κ→0.04, 6–14 breach-cycles/seed) says the leverage cycle fires at
  actual COVID. §3.3 reads like a leftover from the 20%-margin regime. Reconcile.
- **L6 — "~37 client defaults"** appears to be the **sum over two seeds** in `verify_rebalance`
  (16+21); per seed it is ~18 (12-session window) to 26 (20-session window, seed 42). Report as a
  per-seed distribution, not a point figure.

---

## 7. Recommended order of operations

1. **Commit + tag** the working tree (H1). Nothing else is safe until this is done.
2. **Fix C1** (transfer close-out → reassign book, or firesale for direct clients), re-run the
   conservation probe, and confirm `Σpos ≡ 0` and `Δcash ≈ haircuts` in every arm.
3. **Regenerate** H1/H2/NET + the GBM ensemble (≥5 agent seeds × ≥20 paths) under the committed,
   fixed config (H4, M1). Treat all current numbers as void until then.
4. **Re-cite** the 8% floor (H2) and the 15% house margin (H3); add a floor sensitivity sweep.
5. **Add** the planned MCR + Franke J-test to the calibration chapter (already prototyped) and the
   identifiability slices; resolve the stressed `ft_sigma_c` box-floor.
6. **Reconcile docs** (L1–L6) once the numbers are regenerated.

Items 1–3 are gating for any results chapter; 4–6 are defensibility polish. The contribution itself
— a calibrated, regulation-faithful ABM that quantifies the buffer value of the intermediation tier
— is genuine and, after C1, well-supported by the model's own output.

---

## 8. Sources

Code & docs: `model/{globals,agents,simulation,clearing,lob}.py`, `run_simulation.py`,
`run_thesis_experiments.py`, `gbm_lever.py`, `verify_rebalance.py`, `model.md`, `AGENT.md`,
`THESIS_SYNTHESIS.md`, `CALIBRATION_VALIDATION.md`, `clearing_writeups.tex`. Measurements:
`audit_conservation.py`, `audit_probe2.py`, `verify_rebalance.py` (this session, sandbox).

External (verified this session):
- [Basel III leverage ratio framework (BIS, 3% min / 5% G-SIB)](https://www.bis.org/publ/bcbs270.htm)
- [CME — Customer Margining / gross customer margining & segregation](https://www.cmegroup.com/education/articles-and-reports/customer-margining-at-cme-clearing.html)
- [CME — E-mini S&P 500 margins (SPAN)](https://www.cmegroup.com/markets/equities/sp/e-mini-sandp500.margins.html)
- [BIS Quarterly Review, Dec 2018 — "Two defaults at CCPs, 10 years apart" (Lehman/LCH & Aas/Nasdaq)](https://www.bis.org/publ/qtrpdf/r_qt1812x.htm)
- [Clarus — Default at Nasdaq Clearing (Aas, €166m fund drawn)](https://www.clarusft.com/default-at-nasdaq-clearing/)
