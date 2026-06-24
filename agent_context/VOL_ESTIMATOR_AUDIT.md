# IM / DF volatility estimator — audit vs real CCP practice

*How the model estimates the volatility that drives initial margin and the default fund, checked
against CCP methodology disclosures. Numbers measured on the real ES path this session
(`vol_audit.py`). Date 2026-06-18.*

---

## What the model does

`globals.im_fraction(σ) = clip(z · σ_day · √MPOR, 6%, 30%)`, `z = 2.326` (99% one-tailed **Gaussian**),
`MPOR = 2` days, `σ_day = σ_1min · √390`. The driving `σ_1min` (`simulation._im_var`) is a
**RiskMetrics EWMA of squared 1-min sim returns, half-life 2 trading days, overnight gaps excluded**,
seeded at the regime `SIGMA_V`. The DF uses the *same* σ with `z = 3.0` (≈99.87% Gaussian), clipped
to [10%, 35%], cover-2 SLOIM.

## What real CCPs do (verified)

The industry standard for CCP IM is **filtered historical simulation (FHS)**: an **EWMA conditional
volatility** is used to scale a long history of returns, and the margin is the **empirical quantile**
of those scaled returns (LCH, ICE, Eurex). CME's **SPAN 2** is an **HVaR** framework on ≥10-year
history with volatility scaling, plus an explicit **Stress-VaR** component for the fund. Confidence
99% (ETD) / MPOR 1–2 days, with an EMIR Art. 28 anti-procyclicality floor.

**Verdict: the *family* is correct** — EWMA conditional vol is exactly what CCPs use, and 99% / MPOR-2
/ APC-floor are the right regulatory settings. The deviations are (i) **Gaussian quantile instead of
the empirical/FHS quantile**, (ii) a **far-too-short EWMA half-life**, and (iii) **overnight gaps
excluded** from the estimator. (i) and (iii) make the model *under*-margin; (ii) makes it *over*-react.

---

## Measured deviations (real ES path)

| Issue | Measured | Implication |
|---|---|---|
| **Gaussian vs FHS quantile** | empirical 99% of EWMA-standardised ES returns = **3.04** (stressed) / **3.22** (calm) vs Gaussian **2.326** | IM **under-margined ~31–39%** at a given vol — the fat-tail gap FHS exists to close |
| **EWMA half-life** | model 2-day → stressed IM peak **17.8%**, peak/mean **2.19×**; RiskMetrics ~11-day (λ≈0.94) → peak **12.6%**, peak/mean **1.51×** | 2-day is ~5× more reactive than any real CCP EWMA → **overstates the peak procyclical spike ~40%** |
| **Overnight gaps excluded** | gaps are **45%** (stressed) / 29% (calm) of daily variance; worst stressed gap **−6.5%** | margin marks the book *across* the gap but the vol estimate ignores it → **under-covers the close-to-close move** it is exposed to |

Note the calm IM is pinned at the 6% APC floor regardless of half-life (the VaR is far below 6% in
calm), so these effects bite in stress — which is where they matter.

The two understatements (fat tail × gap) compound: a proper FHS, close-to-close estimator would set
stressed IM materially higher than the model's Gaussian-intraday number. The 6% floor and the reactive
EWMA partly mask this, but the methodology as written systematically under-charges relative to a real
FHS CCP.

---

## Status — all three implemented 2026-06-18

**#1, #2 and #3 are done** (`globals.py` / `simulation.py`): `IM_VOL_HALFLIFE = 11*390` (RiskMetrics
λ≈0.94); `IM_CONF_Z = 3.0` (empirical FHS 99% quantile, was Gaussian 2.326); `DF_STRESS_Z = 3.9`
(raised from 3.0 to stay above IM so the SLOIM coefficient stays positive); **`IM_INCLUDE_GAPS = True`**
— a separate EWMA of squared session-boundary gap returns is added to the daily variance
(`σ_daily² = σ_intraday²·390 + σ_gap²`, seeded from the regime's historical gaps), so margin covers the
close-to-close move. LaTeX (`clearing_writeups.tex` IM paragraph: FHS `α₉₉=3.0`, √MPOR scaling,
close-to-close σ) and `model.md` §3.1/§3.2/§4 synced.

**Measured effect** (Option A + C1, seed 42): DF>IM at every σ (SLOIM 3–6%); calm still pinned at the
6% floor (clean, 0 defaults); client IM share ~71–75%. Procyclicality (stress/calm posted IM): **1.52×**
with #1+#2 (intraday-only), **~2.0×** once gaps are included (#3) — and ~2× matches CME's *actual* ES
margin increase through March 2020, so the close-to-close estimator is the more empirically faithful
one. The richer (higher, fat-tailed, gap-inclusive) margin strengthens the funding channel: a handful
of client defaults now appear at COVID (≈7–12, localised — 0 member defaults, waterfall 0), and
contagion onsets a little shallower in the reverse-stress sweep; banks still rarely default and direct
still mutualises (H1 intact, inventory/cash conserved). Stressed IM rises from ≈10.3% (intraday-only)
to ≈14% (gap-inclusive), covering the overnight move.

## Recommended changes (ranked: simple → most correct)

**1. [DONE] Lengthen the EWMA half-life: 2 days → ~11 days (RiskMetrics λ≈0.94).** One constant
(`IM_VOL_HALFLIFE = 11 * 390`). The current 2-day window is far more reactive than any real CCP and
inflates the peak collateral spike ~40%; at 11 days the procyclicality is ~1.5× (still the H2 story,
now realistic). Keep the reactive-vs-flat contrast for H2. *Simplest, clear correctness basis.*

**2. Use the FHS quantile instead of the Gaussian z (the CCP-standard fix).** Either:
   - *(cheap)* set the VaR multiplier to the **empirical 99% of EWMA-standardised returns (~3.0)**
     rather than 2.326, and cite it as a fat-tail / FHS-consistent adjustment — a one-constant change
     that removes the ~30% understatement; or
   - *(proper)* maintain a rolling buffer of standardised returns and take the empirical 99% quantile
     each cycle — this makes the model a genuine FHS engine (the LCH/ICE/Eurex method), directly
     citable, at the cost of a little state.

**3. Include overnight gaps in the estimator (close-to-close).** Feed the day-boundary return into the
IM/DF EWMA (don't re-anchor it away) so the margin vol reflects the gap it must cover. A few lines;
raises stressed IM toward the close-to-close level. (Keep the gap *excluded* from the calibration
moments — that exclusion is correct for matching intraday stylised facts, but not for margin.)

**On the default fund:** the 3σ-Gaussian stress move inherits the same fat-tail understatement, but it
is clipped to a historical extreme-but-plausible band [10%, 35%], and in stress the clip/historical
floor dominates — so it is a defensible parametric proxy for the scenario-based Stress-VaR CCPs use.
Lower priority; if you change #2, the DF z benefits automatically.

**Minimal high-value combo:** do #1 and the cheap form of #2 (two constants). Together they make the
estimator behave like a real CCP EWMA-FHS margin (realistic reactivity + fat-tailed quantile) with no
structural code change, and they're individually citable (RiskMetrics λ=0.94; FHS empirical quantile).

## Sources
- FHS with EWMA conditional vol is the CCP IM standard: https://www.risk.net/media/download/954861/download ; https://www.systemicrisk.ac.uk/sites/default/files/images/Gurrola-Perez_Model%20risk%20in%20OTC%20regulation%20SLIDES%20(website).pdf
- CME SPAN 2 (HVaR ≥10yr + Stress-VaR): https://www.cmegroup.com/clearing/files/cme-span-2-margin-framework.pdf ; https://www.cmegroup.com/clearing/risk-management/span-overview/span-2-methodology.html
- RiskMetrics EWMA λ≈0.94 standard; EMIR Art. 24/28/41 (99%, MPOR, APC floor).
- Measurements: `vol_audit.py` (this session).
