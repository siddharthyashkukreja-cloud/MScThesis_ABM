# Option A implemented + tested — results and a tractable thesis design

> **Superseded floor (2026-06-19):** the type-differentiated "Option A" floor described here (BCM
> Basel-LR `cash/exposure≥4.25%` as its solvency floor; NBCM Reg 1.17; NBCM $50M–1B) was a stepping
> stone. The **committed** scheme is now: client-clearing solvency = CFTC Reg 1.17 `cash/IM≥8%` for
> **both** tiers, a **bank-only** Basel-LR deleverage trigger for BCMs, and NBCMs resized to $0.5–3B
> (fixes the "only NBCMs default" bias). See `FLOOR_BALANCE_RESULTS.md` + `AGENT.md`. The C1, FHS,
> close-to-close, and VOLUME_LOT results below are unaffected.

*Implements the regulation-accurate, type-differentiated solvency floors (Option A from
`FLOOR_AND_MARGIN_ANALYSIS.md`) and tests them end to end. All numbers measured this session
(sandbox, seed 42). Reproducers: `optA_test.py`, `floor_sweep.py`, `audit_conservation.py`.*

---

## 1. What changed in the model

Behind a single toggle `DIFFERENTIATED_FLOORS = True` (set it `False` to restore the legacy uniform
8% floor):

| Member type | Rule (Option A) | Constant | Was |
|---|---|---|---|
| Bank CM (BCM) | Basel III / US eSLR leverage ratio: `cash/exposure ≥ 4.25%` | `LR_FLOOR_BCM = 0.0425` | uniform 8% on exposure |
| Non-bank CM (NBCM) | CFTC Reg 1.17: `cash/IM ≥ 8%` (capital over risk margin) | `REG117_FLOOR_NBCM = 0.08` | uniform 8% on exposure |
| Client | FCM freeze near distress: `cash/exposure ≤ 4%` | `CLIENT_FREEZE_FLOOR = 0.04` | uniform 8% on exposure |
| BCM deleverage target | own-book buffer above the floor | `DELEVERAGE_TARGET_BCM = 0.06` | `CAP_DELEVERAGE_TARGET = 0.10` |

Files touched: `globals.py` (constants), `agents.py` (`NonBankingClearingMember.capital_ratio` →
cash/IM under Option A), `simulation.py` (per-type member floor in Phase 2; client freeze on
`CLIENT_FREEZE_FLOOR`; per-type porting headroom; BCM deleverage target). The legacy path is fully
preserved under the toggle.

**On "adjust other constants too":** the only constant that genuinely couples to the floor is the
deleverage target (now 6%, a buffer above 4.25%). The margin/DF floors (`IM_FLOOR = 6%`,
`DF_STRESS_FLOOR = 10%`) are independent — they are EMIR/CME-anchored margin parameters, not leverage
limits — so leave them. Member capital ($5–10B BCM / $50M–1B NBCM) and the 15% house margin stay
(both realistic; see §4). So Option A is a *small* coupled change, not a recalibration.

---

## 2. Results — the model now matches reality, and contagion is endogenous

Seed 42; **committed transfer close-out, now conservation-safe after the C1 fix** (§3); "reverse-stress
c" amplifies the COVID path's returns by `c` (the model's own deeper-stress lever). `ccp_used` = CCP
cash drawn:

| scenario | drawdown | arm | client def | BCM def | NBCM def | waterfall | ccp_used | inventory |
|---|---:|---|---:|---:|---:|---:|---:|:--:|
| calm | −5% | tiered | 0 | 0 | 0 | 0 | $0 | conserved |
| **actual COVID** | −26% | tiered | **0** | 0 | 0 | **0** | $0 | conserved |
| actual COVID | −26% | direct | 3 | 0 | — | **3** | $0.3B | conserved |
| reverse-stress c=2 | −45% | tiered | 55 | 0 | 2 | **0** | $0 | conserved |
| reverse-stress c=2 | −45% | direct | 60 | 0 | — | **4** | $0.6B | conserved |
| reverse-stress c=2.5 | −52% | tiered | 65 | **0** | 4 | **3** | $0.06B | conserved |
| reverse-stress c=2.5 | −52% | direct | 76 | 0 | — | **4** | $0.7B | conserved |
| GBM s110 (extreme tail) | −56% | tiered | 45 | **10** | 5 | **5** | CCP fails | conserved |
| GBM s110 (extreme tail) | −56% | direct | 66 | 0 | — | **4** | — | conserved |

Three clean takeaways:

1. **Calm and actual COVID are benign in the tiered structure** (0 defaults) — *more* faithful than
   the 8%-floor model (no major CCP member defaulted in March 2020; it was a margin/funding event).
   The direct arm already taps the pooled fund at COVID because its cover-2 fund is sized to small
   clients, not banks (review item M2) — itself a clean illustration of the tier's value.
2. **Contagion is endogenous and graded by severity.** It appears only under deeper-than-COVID
   stress, and the tier **localises it through ~−52%**: tiered holds banks at zero defaults and the
   mutualised fund at L0 to ~−45% (only small NBCMs fail by −52%, waterfall L3, ~$0.06B CCP cash),
   while direct reaches L4 throughout and draws ~$0.6–0.7B. A clean, honest H1.
3. **Tiering has a limit — far deeper than COVID, and path-dependent.** Under the conserving transfer,
   banks survive to ~−52% (0 BCM defaults); only on the vicious −56% GBM path does the whole member
   tier collapse (all 15, waterfall L5 / CCP failure). The tipping point tracks the concentration and
   speed of the move, not just its size (cf. §6.5 E6). The earlier "all members default at −56%"
   reading was a firesale-close-out price-impact artifact; the conserving transfer is materially more
   resilient. *Caveat:* the −56% extreme tail uses the CCP-warehouse fallback (no surviving
   counterparty), which adds the warehoused book's full mark-down on top of the haircut and so
   overstates the apocalyptic-tail loss — report the extreme tail qualitatively (CCP backstop
   reached), not by its dollar figure.

**Conservation (after the C1 fix):** total system inventory is conserved (Σinv = 0) in *both* arms
and *both* close-out modes; under the committed transfer mode the direct-arm COVID run that used to
leak −$14.8B / −7,427 lots now settles at −$0.30B with Σinv = 0 (the genuine haircut on 3 defaults).
`ccp_used` is now a clean metric under transfer ($0–0.7B in the H1 regime), because the book is
transferred to survivors rather than warehoused/disposed through the CCP node.

---

## 3. C1 close-out conservation — FIXED (2026-06-18)

The committed `transfer` close-out used to flatten a defaulted book without reassigning it, breaking
VM zero-sum and leaking cash (direct arm at COVID: −$14.8B / −7,427 lots). It now **transfers** the
book: `_transfer_book` reassigns the defaulter's net position to the surviving participant with the
largest opposing book (the ODD PositionAuction / EMIR-48 target), booked as a fill at `mid` so the
receiver's VM is charged from the transfer price; `_ccp_warehouse` is the no-counterparty fallback.
Three sites fixed in `simulation.py` (Phase 0 direct client → surviving direct client; Phase 1 tiered
client → its member assumes the book; Phase 2 member → surviving member, banks preferred). **Verified:**
Σinv = 0 in every run, and the direct COVID leak collapsed from −$14.8B to −$0.30B (the genuine
3-default haircut). The committed transfer mode is now conservation-safe, and the §2 numbers are run
under it. The only residual is the extreme-tail warehouse fallback double-count noted in §2 takeaway 3
(report the −56% tail qualitatively). Toggle nothing — this is now the default behaviour.

---

## 4. The strategic question — how to present this without doing too much

Under Option A "number of defaults" is the wrong headline observable (it is ~0 at COVID by design).
Lead with observables that exist *without* defaults, and use a severity axis where defaults matter.

**Recommended: two hypotheses + one supporting result, each one main figure.**

- **H2 (primary — margin procyclicality / collateral demand). No defaults needed.** The reactive IM
  rises ~1.4–1.6× calm→stress, draining members' free cash — the empirically-documented March-2020
  "dash for cash" (ESRB/BIS). *Observables:* posted-IM time series and peak/calm ratio; aggregate
  collateral demand; free-cash drawn. *Design:* {calm, stressed} × {reactive, flat IM}, tiered only.
  This is the strongest, most defensible result and is pure validation-plus-mechanism — it needs no
  cascade. (You already have the figures from `analyze_volume_margin.py`.)

- **H1 (structural — the intermediation tier raises the mutualisation threshold).** *Observable:*
  deepest waterfall level / probability the mutualised DF is drawn, **as a function of drawdown
  severity**, tiered vs direct. *Design:* a severity sweep (reverse-stress c, or GBM drawdown bins)
  × {tiered, direct}, one figure with two curves. Shows tiering localises through moderate-deep
  stress and where it breaks — a richer, more honest result than a single count.

- **NET (supporting — one number).** Net vs gross client margining ≈ halves posted IM (Duffie-Zhu).
  {gross, net}, stressed, IM observable. A sentence and a number, not a chapter.

**This scopes the presentation to:** H2 = 4 conditions (IM observable); H1 = 1 severity sweep × 2
structures (waterfall observable); NET = 1 number. You do **not** present calm×stressed×GBM ×
direct×tiered for every observable — calm/stressed live in H2 (margin), GBM/severity lives in H1
(structure). That is the way out of "too much."

**Framing.** The contribution is: *a calibrated, regulation-faithful ES + CCP ABM that (i) reproduces
the empirical margin-procyclicality / collateral-demand spike, and (ii) quantifies how the
client-clearing tier raises the stress threshold at which losses move from member balance sheets to
the mutualised fund.* Resilience to COVID-class stress is the validated baseline; mutualisation is
conditional on deeper stress or removing the tier (direct). This is accurate, novel, and tractable —
and it matches the historical record (Lehman within-IM/high-recovery; Nasdaq-Aas DF drawn).

---

## 5. Caveats / cleanups for the new floors
- `verify_rebalance.py` hard-codes `breach = capital_ratio ≤ 0.08`; under Option A the BCM floor is
  4.25% and the NBCM ratio is cash/IM, so that diagnostic count is stale (cosmetic — fix when reused).
- Add `LR_FLOOR_BCM`, `REG117_FLOOR_NBCM`, `CLIENT_FREEZE_FLOOR` to the `model.md` §4 parameter table
  and the `clearing_writeups.tex` margin section, with the basel_lr / us_eslr_2025 / cftc_reg117 cites.
- The accurate-floor result supersedes the "tiered resilient to −56%" line in `model.md` §6.1 /
  `AGENT.md` — regenerate H1 as a severity curve after the C1 fix.

## Sources
- Basel III LR 3% + G-SIB buffer (50% of surcharge): https://www.bis.org/publ/bcbs270.htm
- US eSLR final rule (eff. 2026-04-01, ~3.5–4.25%): https://www.federalreserve.gov/newsevents/pressreleases/files/bcreg20251125b2.pdf
- CME E-mini S&P 500 SPAN margins: https://www.cmegroup.com/markets/equities/sp/e-mini-sandp500.margins.html
- BIS Dec-2018 — Lehman/LCH & Nasdaq-Aas defaults: https://www.bis.org/publ/qtrpdf/r_qt1812x.htm
