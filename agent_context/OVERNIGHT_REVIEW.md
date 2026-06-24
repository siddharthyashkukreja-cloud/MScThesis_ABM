> **HISTORICAL SNAPSHOT (2026-06-18).** This document records an earlier audit/state and is NOT the current model. For the current committed model and results see AGENT.md and THESIS_SYNTHESIS.md. Notably superseded since: BCM prop cap 6×→2×, house margin 20%→15%, leverage-balanced client→clearer assignment, client balance-sheet rebalance, and the close-out/GBM-ensemble findings.

# Overnight review — full audit, fixes, results, and thesis-presentation plan

*Prepared overnight (2026-06-18). Seven specialist audits (Almgren–Chriss, all margin, calibration,
ODD fidelity, CCP protocols vs. real-world, code correctness, results-presentation) were run in
parallel; their findings are synthesised below, the clear bugs are already fixed, the hypotheses were
re-run under the fixed model, and the docs were updated. Read §0 first.*

---

## 0. TL;DR — what you need to know this morning

**The model is in good shape.** The clearing layer matches real CCP protocols on every structural
check (waterfall order, cover-2 sizing, IM-first, porting, recovery), the Almgren–Chriss trajectory is
a genuine (not faked) AC schedule, and the calibration math is correct. The audits found **3 code bugs
(all now fixed)**, **2 documentation overclaims (now corrected)**, and a list of **defensibility
gaps** to decide on (none fatal).

**Three bugs I fixed tonight** (all were affecting the escrow model you just committed):
1. **Fire-sale queue overwrite (CRITICAL).** When ≥2 of a member's clients defaulted in the same
   cycle, only the *last* client's position was scheduled for liquidation — the rest stayed un-liquidated
   on the member's book. This corrupted exactly the client→member contagion channel the thesis studies.
   Fixed (`agents.py`: `extend` not overwrite, clamp against already-queued).
2. **Excess-IM destruction on default (CRITICAL for cash conservation).** `_seize_im` released a
   defaulter's full posted IM from the CCP account but applied only the part covering the loss — the
   excess vanished from the economy (~$2.8B in a crash run). Fixed: excess returns to the defaulter's
   estate; the escrow identity `im_account == Σ posted_im` now holds to float precision.
3. **Porting capacity used the wrong basis under escrow (MAJOR).** Receiver headroom was computed as
   `cash/(floor·im_frac)`, overstating spare capacity ~8× under the new `cash/exposure` ratio, so
   clients were ported to members already over their floor. Fixed to `cash/floor` under escrow.

**Two doc overclaims, now corrected:** (a) the close-out was described as an "auction to the surviving
member with the largest opposing position" — the code never selects a counterparty; it closes out at
recovery and mutualises the haircut (a simplification of the ODD's PositionAuction). (b) `Vuillemey
2023` → `Vuillemey 2020` (the correct JF paper). Both fixed in `model.md`; see §6 for the bib follow-ups.

**Your overnight run:** if you launched `run_thesis_final.sh` *before* these fixes, the experiments
ran on the buggy code — **re-launch them** (calibration is unaffected, so `EXPERIMENTS_ONLY=1` is
enough). I ran a reduced-seed (6) version under the fixed code; results in §4. They are strong:
- **H1 — tiering contains contagion.** Direct clearing drives losses into the mutualised waterfall
  (L3, ~12 direct-member defaults); tiering's member tier absorbs them (waterfall L0.5, ~0 member
  defaults). The member balance sheet is a shock absorber.
- **H2 — the procyclicality trade-off now has teeth.** Reactive (procyclical) margin is ~1.8× cheaper
  in calm ($8.6B vs $15.7B IM) but **doubles stressed client defaults (16.8 vs 8.3)** vs through-the-cycle
  margin — because escrow makes margin a funding-liquidity channel. The escrow change is what unlocked
  the contagion side of H2 (previously it was collateral-cost-only).
- **NET — netting helps twice.** Net client margining cuts collateral (~$15.2B vs $18.6B) *and* slightly
  reduces contagion (14.5 vs 16.8 defaults, L0 vs L0.5).

**Highest-value things still to decide** (none are blockers; ranked in §3): the calm calibration
window artifact, adding the Franke J-test p-value, and how to justify the Almgren–Chriss urgency
parameter. Everything else is documentation or reporting polish.

---

## 0b. Follow-up changes applied (your morning decisions)

Acting on your replies to §3:
- **#3 AC urgency now DERIVED, not hardcoded.** `globals.ac_urgency(regime) = √(λ·σ²/η)·H` (AC 2000
  Eq. 19), λ=`LAMBDA_RISK`=2.0. Calm κH = **1.03** (≈TWAP), stressed κH = **2.92** (≈2.85× calm) — the
  regime σ/η drive it, so a crash liquidates more aggressively by construction. Wired into `_margin_cycle`.
- **#11 IM seized before the waterfall against the FULL loss** (VM shortfall + close-out haircut), for
  member defaults too — IM is the defaulter's first-loss resource. **This changes the result:** stressed
  deepest waterfall drops to **L0** (defaulter IM + 0.80 recovery absorb the loss before mutualisation),
  vs L0.5–L3 before. This is realistic (real CCPs rarely touch the mutualised fund at actual severity);
  the H1/H2 contrasts now live in **default counts and loss allocation**, not waterfall depth (waterfall
  depth would need the deferred reverse-stress amplifier to separate the arms).
- **#10 EOD liquidation flush** — `_flush_liquidations()` completes any open AC liquidation at each
  session boundary, so no assumed book is carried across the overnight gap.
- **#8 reporting** — `client_history["im_posted"]` now logs the actual escrowed IM, not the 20% house figure.
- **#7 dead code** — the legacy `CAP_DELEVERAGE_TARGET` deleverage path is removed; members freeze at the
  floor (documented as a deliberate ODD deviation).
- **#1 calm window identified:** the calmest contiguous 20-session 2019 stretch is **Nov 5 – Dec 2, 2019**
  (mean daily vol 0.00401, **2.32× calmer** than the current first-20-session window 0.00931, and below the
  full-year 0.00669). Recommended fix: point the calm regime (sim *and* target) at this window and
  re-calibrate in the final run — see §3.1.

Escrow conservation re-verified after all changes (`im_account == Σ posted_im`, $0 drift); calm still
0 defaults; stressed still shows the client funding-squeeze.

## 0c. Final-run setup (ready)

All changes for the definitive run are committed:

| Item | Change |
|---|---|
| DF floor | `DF_STRESS_FLOOR` 0.15 → **0.10** (only binds calm; fixes the calm DF/IM anomaly) |
| Stressed window | new **long** window (Feb 17–May 28 2020, ~75 sessions) — `fv_stressed.csv` + processed rebuilt from your upload (short version backed up) |
| Calm window | sliced to the quiet **2019-09-13 → 12-27** (~75 sessions), `fv_calm.csv` regenerated (full-year backed up) — fixes the #1 artifact AND matches the stressed length |
| Regime σ | `SIGMA_V` calm 2.97e-4→**2.40e-4**, stressed 1.82e-3→**1.24e-3**; `V0` updated to each window's open |
| AC urgency | derived: calm **0.83**, stressed **1.98** (re-derived on new σ) |
| Calibration | `N_DAYS=75` (matched length → comparable D across regimes); `N_RUNS` shared by grid+surrogate (set the value for the final run, e.g. 8) |
| J-test | added to `calibrate.py mcr` — reports Franke bootstrapped J + p-value alongside the MCR |
| Run script | `EXPERIMENTS_ONLY=1` now runs **MCR/J-test + experiments** (skips only relock), for the post-relock stage |

**Final-run recipe (on your machine):**
```bash
cd ~/Desktop/GitHub/mscthesis_abm
mv output/thesis_final output/thesis_final_PREFIX_preview      # keep last night's preview
# (optional) raise N_RUNS in calibrate.py for the definitive run, e.g. 8

# STAGE 1 — re-fit theta on the new windows (~24-36h at N_DAYS=75; the windows ~doubled)
nohup ./run_thesis_final.sh > output/console.log 2>&1 &
tail -f output/thesis_final/run.log
# When it finishes: the VERIFY step WILL say "DIFFERS" — that is EXPECTED (new windows).
# Copy the relock's fresh-seed-chosen optimum (output/relock/) into globals.CALIBRATED.

# STAGE 2 — MCR/J-test + experiments on the final theta
EXPERIMENTS_ONLY=1 N_SEEDS=40 nohup ./run_thesis_final.sh > output/console2.log 2>&1 &
```
Reported at the end: `relock/` (θ + identifiability), `mcr_{calm,stressed}.json` (MCR + **J-test p-value**), `experiments/` (H1/H2/NET, 40 seeds).

**Still flagged (not blocking the run):** `ETA_TEMP` couldn't be auto-re-measured (`data/impact.py` threw an out-of-bounds on the new windows — needs a small lookahead-clip fix); the existing η is kept and only feeds AC urgency, so it's second-order. `P_ZI`/`VOLUME_LOT` for calm are structural and left as-is (re-run `data/p_zi.py` if you want them window-matched). The experiment stress sub-window is still sessions 10–20 of the stressed data (the crash onset) — fine, but say the word if you'd rather run experiments over the full ~75-session episode.

## 1. Bugs found and FIXED tonight

| # | Severity | Where | What was wrong | Fix applied |
|---|---|---|---|---|
| 1 | CRITICAL | `agents.py` BCM+NBCM `start_firesale` | `self._liq_slices = ac_slices(...)` **overwrote** the queue; clustered client defaults left positions un-liquidated | `.extend(...)` + clamp `qty` against `abs(inventory) − already-queued` |
| 2 | CRITICAL (conservation) | `simulation.py` `_seize_im` | excess posted IM over the loss was destroyed (not returned) | excess → defaulter estate (`agent.cash += im − used`); identity verified |
| 3 | MAJOR | `simulation.py` `_port_clients` | headroom `cash/(floor·im_frac)` overstates capacity ~8× under escrow | `cash/floor` under `IM_ESCROW`, legacy formula otherwise |
| 4 | MAJOR (doc) | `model.md §3.3`, code comments | "auction to surviving member with largest opposing position" — never implemented | reworded to "closed out at recovery, haircut mutualised; no counterparty receives the book (simplification of ODD PositionAuction)" |
| 5 | MINOR (doc) | `model.md §3.3/§8.1` | `Vuillemey 2023` does not exist | → `Vuillemey 2020`, *The Value of Central Clearing*, JF 75(4):2021-2053 |
| 6 | MINOR (doc) | `clearing.py` waterfall L1 comment | said "IM is not escrowed" — false under `IM_ESCROW=True` | corrected to note IM is seized first via `_seize_im` |

A note on the "IM double-count on member default" that the margin audit flagged as CRITICAL: I traced
it by hand. It is **not** a double-count — the deficit `−cash` is computed on cash already net of
escrowed IM, and `_seize_im` then applies that IM once; the mutualised loss equals
`max(0, loss − total_resources)`, which is correct. The real issue in that code path was the excess-IM
destruction (bug #2), now fixed.

---

## 2. Verified CORRECT against real-world protocols (no action needed)

The CCP-protocols audit benchmarked the clearing layer against EMIR, CPMI-IOSCO PFMI, and CME/LCH/Eurex
rulebooks. All six structural checks pass:

- **Default-waterfall order** = EMIR Art. 45 exactly: defaulter IM → defaulter DF contribution → CCP
  skin-in-the-game → mutualised DF → surviving-member assessments → CCP capital. SITG correctly sits
  *before* the mutualised fund.
- **Cover-2 default fund** = PFMI Principle 4 / EMIR Art. 43(2): two largest members' stress losses net
  of IM (SLOIM), extreme-but-plausible stress move, +10% buffer. Matches Eurex's SLOM method.
- **Initial margin** = 99% confidence, 2-day MPOR (EMIR-conservative; CFTC allows 1-day), 6% floor ≈
  CME ES live margin, anti-procyclicality cap. Verified numerically (calm ≈6% floored, stressed ≈12%).
- **Skin-in-the-game** = 3% of the fund, in the real 1-5% band (CME ES ≈2.7%); placed at L2 correctly.
- **Porting** = capacity-checked, not-guaranteed transfer with close-out fallback = EMIR Art. 48 +
  realistic fragility (OFR 2026 single-agent dependency).
- **Recovery 0.80** sits correctly between Lehman/LCH (losses < IM) and Nasdaq 2018 (blew the fund).

The **Almgren–Chriss trajectory is genuine**: `ac_slices` implements `xⱼ/X = sinh(κ(1−j/H))/sinh(κ)`
(AC 2000 Eq. 17), with the κ→0 TWAP limit handled. Execution is endogenous market-order book-walk
(temporary impact via LOB depth; permanent impact ≈0, empirically floored). This is defensible and
well-cited.

The **calibration core is bug-free**: the 5-component loss (KS + variance + ACF₁ + ACF₂ + Hill), the
inverse-SD block-bootstrap weights, the XGBoost surrogate + grid + fresh-seed re-rank, and the MCR are
all correctly implemented and faithful to Gao et al. (2022) / Franke–Westerhoff (2012).

---

## 3. Open issues to decide (flagged, NOT changed — your call)

Ranked by value-to-the-thesis. None is a blocker; several are cheap.

### High value
1. **Calm calibration window artifact (calibration C1).** The calm sim runs the *first 20 sessions* of
   2019 (≈1.40× the full-year volatility) but is scored against full-2019 targets, so calm D≈44 and calm
   MCR≈0% are largely an artifact, not a misfit. **Fix:** score calm against *matched-window* empirical
   moments/CIs (cheap), and report calm MCR both ways. This is the single most useful calibration change
   and turns a weak-looking calm result into an honest one. Stressed does not have this problem.
2. **Add the Franke J-test p-value (calibration C2).** You report the point loss D and the MCR but not a
   bootstrapped goodness-of-fit p-value, which is the headline specification test in Franke–Westerhoff
   (2012, eq. 9). Without it you can say "9/10 moments covered," not "the model is not rejected at 5%."
   ~30 lines on the existing bootstrap machinery (build Σ̂, `J = d'·pinv(Σ̂)·d`, bootstrap null). Adds
   real validation weight for an examiner.
3. **Almgren–Chriss urgency is a hard-coded `2.0`, not derived (AC C1).** AC's whole point is that the
   trajectory curvature κ is set by σ, the impact coefficient η, and risk-aversion λ (Eq. 19). You
   already calibrate σ and η per regime, but the schedule ignores them and pins κT=2 in both regimes.
   **Either** derive `urgency` from your calibrated σ/η (closes the gap and uses the orphaned `ETA_TEMP`),
   **or** keep 2.0 but call it "a front-loaded liquidation, urgency sensitivity-tested" and drop the
   AC-fidelity claim for the parameter. A referee who knows AC will ask.

### Medium value
4. **D not comparable across seed counts (calibration M1).** The KS standardiser is sim-sized, so D at 3
   seeds ≠ D at 6 seeds; the grid (3-seed) and surrogate (6-seed) are tabled together as a cross-check.
   Run both at identical seed counts for any side-by-side D table, and report KS's loss share (≈43%
   calm / 61% stressed) explicitly.
5. **Bootstrap B=200 is thin for stressed (calibration M3).** Only ~29 non-overlapping day-blocks in the
   stressed window → coarse moment SDs. Bump to B≥1000 for the final lock.
6. **Maintenance-margin call indicator is a flow test, not a level test (margin).** `call_indicator`
   fires when `|vm| > 5%·IM` (a large mark move, symmetric in sign), not when account equity falls below
   the maintenance level. It is diagnostic-only (no dynamics depend on it), but rename it
   (`large_mark_move`) or implement the level test so the thesis column means what it says.
7. **Document the escrow design choices that deviate from the ODD (ODD C2).** Under escrow a solvent BCM
   *freezes* at the floor instead of fire-sale-deleveraging — removing the ODD's BCM-vs-NBCM asymmetry —
   and the capital ratio is `cash/exposure` not the ODD's `cash/|tradePosition|`. Both are deliberate
   and now in `model.md §3.3`; just make sure the thesis prose states them as choices, not omissions.
   `CAP_DELEVERAGE_TARGET` is now dead code under escrow — delete it or guard it.

### Low value / reporting
8. **`client_history["im_posted"]` logs the 20% house figure, not the escrowed procyclical IM** — ~2×
   off in calm; reporting-only, fix before drawing any "IM held vs notional" figure.
9. **Calm DF/IM ≈ 44%** (vs realistic 3-5%) because `DF_STRESS_FLOOR`=15% sits well above `IM_FLOOR`=6%.
   Already disclosed; stressed DF/IM is realistic. Optionally lower the DF stress floor.
10. **AC liquidation can be left unfinished near the window end (AC C2)** — tail slices never execute,
    understating close-out loss. Add an end-of-run flush + assertion that no member carries un-disposed
    assumed inventory.
11. **Member-vs-client IM-seizure asymmetry** — a client default seizes IM against shortfall+haircut; a
    member default seizes IM against the VM shortfall only (haircut → waterfall). Defensible
    (SLOIM-consistent) but inconsistent; add a one-line justification.
12. **Bib follow-ups:** rename `vuillemey_2023`→`vuillemey_2020`; add `cftc_fcm_data` (FCM capital
    scale, replaces the wrong `fia_tracker`), `haynes_mcphail_zhu_2019`, `acosta_smith_2018`.
    `CLEARING_LAYER_REVIEW.md` and `CLEARING_METHODOLOGY_PROMPT.md` still say "Vuillemey 2023" / old
    open-market framing (working docs, not swept).

---

## 4. Hypothesis results under the fixed escrow model (6 seeds; directional)

*Full 40-seed numbers come from your overnight `run_thesis_final.sh`. Re-launch it on the fixed code.
These 6-seed runs establish direction and sign. Stressed = the COVID window (~−18.8% drawdown);
90 cleared clients; values are per-seed means.*

### H1 — tiered vs direct clearing (where losses land)

| arm | client defaults | member defaults | deepest waterfall | default fund | IM held |
|---|---|---|---|---|---|
| **tiered** | 16.8 | **0.17** | **L0.5** | $1.49B | $18.6B |
| **direct** | 12.2 | **12.2** | **L3.0** | $1.47B | $12.8B |

**Reading:** in direct clearing the failing clients *are* the CCP's counterparties, so client failures
go straight into the mutualised waterfall (reaches L3, ~12 direct-member defaults). Tiering interposes
member balance sheets that absorb the same client stress — the waterfall barely moves off the floor
(L0.5) and almost no member defaults. **The member tier is a shock absorber that keeps losses out of
the mutualised layer**, at the cost of concentrating them on members. (Tiered shows *more* client
defaults — plausibly the member's stop-out freezing its clients, or seed noise; confirm at 40 seeds.
The robust, headline signal is the waterfall-depth gap L0.5 vs L3.) Calm: zero defaults in both arms —
a clean benign baseline.

### H2 — procyclical (reactive VaR) vs through-the-cycle (flat) margin

| arm | regime | client defaults | deepest waterfall | IM held (collateral cost) |
|---|---|---|---|---|
| reactive | calm | 0.0 | L0 | **$8.6B** |
| flat-12 | calm | 0.3 | L0 | **$15.7B** |
| reactive | stressed | **16.8** | L0.5 | $18.6B |
| flat-12 | stressed | **8.3** | L0.5 | $16.0B |

**Reading:** reactive (procyclical) margin is ~1.8× cheaper to fund in calm ($8.6B vs $15.7B) but
**doubles stressed client defaults (16.8 vs 8.3)** — because escrow turns the margin call into a real
funding-liquidity drain: when reactive IM spikes in the crash, the most leveraged clients can't post and
fail. This is the canonical "procyclical margin converts credit risk into liquidity risk" trade-off
(Murphy–Vause; ESRB; Cont), and it is now *quantified on both axes*. **This is the headline payoff of
the escrow redesign** — H2 previously showed only the cost axis.

### NET — gross vs net client margining

| arm | client defaults | deepest waterfall | IM held |
|---|---|---|---|
| gross | 16.8 | L0.5 | $18.6B |
| net | 14.5 | L0.0 | $15.2B |

**Reading:** net client margining cuts collateral (~$15.2B vs $18.6B — the Duffie–Zhu netting benefit)
*and* slightly reduces contagion (14.5 vs 16.8 defaults, waterfall to L0 not L0.5), because the freed
cash leaves clients better able to meet calls. Modest but consistently in the "netting helps" direction.

---

## 5. Full model summary (every aspect)

**Market layer.** A 1-minute limit-order-book market in ES front-month futures (0.25 tick), cleared by a
call auction each step. Three calibrated trader types: **fundamental** (Chiarella–Iori–Perelló;
reservation `R = V_t·(1 + z·ft_sigma_c·σ_t)` around an exogenous fair value, heterogeneous fixed beliefs
`z` that build concentrated inventories — the load-bearing input to clearing), **momentum** (Majewski
EWMA chartist, fixed `mt_lambda=0.05`), and **zero-intelligence** (Cont–Stoikov noise; geometric depth
`p_zi` fit from real MBP-10 book). Order lifetime via replace-on-new (FT/MT) and `zi_delta` cancellation
(ZI). A market maker and a volatility trader were trialled and rejected on parsimony.

**Fundamental process.** Exogenous `V_t` taken from real ES data: a Kalman/empirical-mid path (calm
2019, stressed COVID 2020) for the headline, with an offline SV-MJD generator (OU stochastic vol +
Merton jumps + overnight gaps) as a synthetic robustness alternative. `σ_t` is EWMA realised vol of
`V_t` (30-min half-life, reset per session), feeding both belief width and procyclical margin.

**Clearing tier.** A CCP, 10 BCMs (FT-subclass: own book + clients), 5 NBCMs (pure client clearers),
and 90 clients (30 FT + 20 MT + 40 ZI) clearing through 5 client-carrying BCMs + the NBCMs, plus an H1
counterfactual where clients clear directly.

**Margin & solvency (committed escrow model).**
- *Initial margin* — procyclical VaR scan `f^IM = clip(z₉₉·σ_day·√2, 6%, 30%)`, physically **escrowed**
  as cash at the CCP: members post own-book IM, clients post their own IM through their member. Returned
  as positions shrink; seized first on default; excess returns to the estate.
- *Variation margin* — hourly mark-to-(average-fill)-price, settled in cash.
- *Capital ratio* — net-position **leverage** ratio `κ = cash/exposure ≥ 8%` (cash already net of posted
  IM), i.e. the Basel leverage ratio, motivated by the leverage-cycle literature. A member that can't
  fund a call defaults on **liquidity**; one breaching the floor **freezes**.
- *Position limit* — static `|own notional| ≤ 2·cash` on all BCMs (no volatility feedback), replacing the
  old σ-VaR house cap.
- *Default fund* — cash-prefunded **cover-2 SLOIM** (two largest members' stress-over-IM losses, +10%
  buffer); CCP skin-in-the-game = 3% of the fund.

**Default management.** A defaulted **client** is liquidated **open-market** by its member (Almgren–Chriss
book-walk — the client-level price-impact channel). A defaulted **member** (and a direct client) is
**closed out at 0.80 recovery by the CCP**, the haircut mutualised (no counterparty auction is modelled).
Client positions are **ported** to members with capacity (EMIR Art. 48), unported ones closed out. Losses
flow through the EMIR Art. 45 waterfall: defaulter IM → defaulter DF → SITG → mutualised DF → survivor
assessments → CCP cash.

**Calibration.** Bare-market (clearing off), so invariant to the clearing redesign. SMM on a 5-component
loss (KS + variance + ACF₁ + ACF₂ + Hill), inverse-SD block-bootstrap weighted, optimised by an XGBoost
surrogate and an exhaustive grid, with a fresh-common-seed re-rank to kill winner's-curse. Two regimes
(calm 2019, stressed COVID).

**Validation.** Stylised facts (fat tails via Hill, no linear ACF, short-horizon vol clustering) plus the
**Moment Coverage Ratio**: stressed covers 9/10 moments within empirical sampling error (87% mean), calm
collapses against full-year CIs (window artifact — see §3.1). Long-horizon vol clustering beyond ~30 min
is a documented single-timescale limit.

---

## 6. Thesis presentation plan (from the sample-paper audit)

**Closest templates:** Paddrik & Zhang (2020, OFR WP 20-04) for H1 loss-allocation; Murphy, Vasios &
Vause (BoE 2014/2016) + Murphy & Vause (2021 "CBA of APC") for the H2 cost-vs-contagion trade-off;
Heath–Kelly–Manning (2016) for the tiered-vs-direct framing; Aymanns & Farmer (2015) for the VaR-driven
endogenous-dynamics framing; Bardoscia et al. (2019) for "is cover-2 enough"; Bookstaber–Paddrik–Tivnan
(2018) for ABM legitimacy.

**Figures.**
- *H1 headline* — stacked loss-allocation bar per `{arm × regime}`: client → member → mutualised DF →
  CCP capital. *Plus* a **waterfall-depth distribution** (share of seeds reaching each level) — depth is
  ordinal, never report its mean alone. Optional Sankey of loss flow (tiered vs direct, stressed).
- *H2 headline* — the **cost-vs-contagion frontier**: IM cost on x, stressed defaults (or waterfall
  depth) on y, one point cloud per arm across seeds, arm means connected. *Plus* the **margin path through
  the crash** (reactive vs flat, with the price path), annotated with peak-to-trough and n-day-max margin
  jumps.
- Default-count bars with seed dispersion (error bars + strip plot) for both hypotheses.

**Tables.** One master `arm × regime` table, columns = client defaults, member defaults, deepest waterfall
(modal level + % reaching it), IM cost, DF size, drawdown — all **mean ± SD over seeds**. A separate H2
trade-off table pairing the cost metric, the procyclicality metric, and the contagion metric. A
calibration/scale table so readers can sanity-check IM/DF/SITG against real CCPs.

**Reporting the ensemble (this is what separates a strong ABM thesis):** never means alone — report
mean ± SD or percentile CIs; show **distributions** for skewed/ordinal metrics (losses, waterfall depth)
via box/violin/strip plots; **fan charts** for time series (margin path, price); tail metrics ("% of
seeds with ≥1 member default", "% breaching the mutualised layer", 95th-pct drawdown); and state where the
**between-arm difference exceeds within-arm seed dispersion** (your credibility move).

**Framing/vocabulary:** waterfall layers in canonical order (IM → DF → SITG → mutualised DF → assessments
→ recovery tools); "loss mutualisation / cross-subsidisation", "cover-1/cover-2", "through-the-cycle vs
reactive margin", "MPOR", "procyclicality converts credit risk into liquidity risk", "a convex
loss-vs-stress curve as evidence of contagion (vs additivity)". Cast H1 as *where a regulator wants losses
to land* and H2 as *how much collateral cost buys how much resilience*.

---

## 7. Suggested order of attack (when you're back)

1. **Re-launch the experiments** on the fixed code (`EXPERIMENTS_ONLY=1 ./run_thesis_final.sh`) — the
   bugs fixed tonight change the contagion numbers.
2. Decide the three high-value calibration/AC items (§3.1–3.3): matched-window calm scoring, the J-test
   p-value, and the AC-urgency justification.
3. Bib follow-ups (§3.12) and delete dead `CAP_DELEVERAGE_TARGET`.
4. Build the H1/H2 figures per §6 once the 40-seed results land.

*Artifacts: fixes in `model/{agents,simulation,clearing,globals}.py`; docs in `model.md`/`AGENT.md`;
corrected write-ups in `clearing_writeups.tex`; reduced-seed results in `output/hyp/` and
`output/escrow/`; this report in `OVERNIGHT_REVIEW.md`.*
