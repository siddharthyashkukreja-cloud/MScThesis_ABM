# Calibration campaign — FINDINGS

Interim screening resolution (NOT thesis-final). Surrogate-assisted SMM (`calibrate.py run`),
both regimes, fast args (`48 20 2 1 12 8`; E5 high-dim `96 20 2 1 16 12`). Each experiment is
gated behind an env flag (default OFF), so E0 is the recoverable baseline. Campaign ran in
**2.56 h**, 8 experiments × 2 regimes + E6, **zero errors**. Full per-run θ\*, moments and configs
are in `output/campaign/<EID>_<regime>.json`; the comparison table is `SUMMARY.md`; the
gaps-vs-shock sweep is `E6_gaps_vs_shock.csv`.

D = validated true loss (lower better); component deltas are Franke-standardised (sampling-SDs
off). Acceptance held throughout: surrogate held-out R² ≥ 0.58 every run, book depth bounded
(~197–373, never near the 5000 runaway flag), no errors.

## Headline: ΔD vs the E0 baseline

E0 baseline: **calm D = 44.02**, **stressed D = 13.84**.

| EID | config | calm D (ΔD) | stressed D (ΔD) | verdict |
|-----|--------|-------------|-----------------|---------|
| E0  | baseline control | 44.02 (—) | 13.84 (—) | reference |
| E2  | `mt_lambda` in the loop | **41.70 (−2.32)** | **11.93 (−1.91)** | **improves BOTH** — best stressed, cheap (1 extra param) |
| E3  | `mt_lambda` pinned ~3h half-life | 41.23 (−2.79) | 14.38 (+0.54) | calm total down but **Hill blows up** (see below) — reject |
| E4a | +10 short-λ MT (30 total) | 43.11 (−0.91) | 16.68 (+2.84) | calm marginal (R²=0.58), stressed worse |
| E4b | two-cohort MT (15 long + 15 short) | 44.46 (+0.44) | 16.80 (+2.96) | no gain either regime |
| E5  | FT/MT Bernoulli gate + cancellation | **38.41 (−5.61)** | 14.70 (+0.86) | **best calm**; biggest clustering gain; stressed ~flat |
| E1  | clearing/CCP tier active in the loop | 44.45 (+0.43) | 27.98 (+14.14) | calm ≈ E0 (inert); **stressed materially altered** (see below) |

## Per-experiment reading

**E2 — `mt_lambda` calibrated (the safe win).** Improves both regimes with clean component
deltas and only one extra dimension. The data prefers a *slower* trend than the pinned 0.05:
λ\* ≈ 0.018 calm (half-life ≈ 38 min) and ≈ 0.041 stressed (≈ 17 min). Calm gain is in ACF1
(4.60 → 2.87); stressed gain is in KS (6.72 → 5.52) and Hill (2.56 → 0.91). This is the
lowest-risk improvement found — Majewski et al. (2018) fix λ externally, but here the data
mildly disagrees and the fit improves without distorting any other component.

**E3 — `mt_lambda` pinned at a 3-h half-life (rejected on the tail).** The calm *total* drops
to 41.23, but the decomposition is a bad trade: KS (16.8 → 8.56) and V (7.12 → 3.25) improve
while **Hill explodes (1.13 → 8.08)** and ACF1 worsens (4.60 → 7.39). A single slow-timescale
trend corrupts the return tail. Stressed is flat-to-worse. A long horizon does **not** help the
|return| ACF (clustering) here. Do not adopt.

**E4a / E4b — more / two-cohort momentum (rejected; revives a D43/D44-dropped design).**
Re-tested as instructed. Neither reviving the count (E4a) nor the two-timescale cohort (E4b)
improves the loss: calm is marginal-to-worse, stressed is clearly worse (+2.8 to +3.0). The
two-cohort design does **not** revive long-horizon volatility clustering — ACF2 is essentially
unchanged vs E0. This **confirms the documented structural limit** (AGENT.md: a single
momentum timescale can't produce multi-scale memory, and a second cohort at this LOB doesn't
either). The D43/D44 rejection stands.

**E5 — FT/MT activation gate + cancellation (the promising one; revives the D36-rejected
gate).** Re-tested as instructed, now that the LOB TTL is gone (D58) so cancellation is the
sole order-lifetime mechanism. **Best calm D of the campaign (38.41, −5.61)** and the largest
clustering improvement anywhere: ACF2 14.37 → 9.17, with a near-perfect Hill (Δ 0.019) and the
lowest calm kurtosis (5.9). Fitted gates are genuinely intermittent (calm `ft_alpha`≈0.77,
`mt_alpha`≈0.27, `ft_delta`≈`mt_delta`≈0.33). Contrary to D36 ("the gate corrupts the ACF"),
under the no-TTL regime it *helps* the |return| ACF. Costs: V worsens (7.12 → 9.14), depth
rises (~373, still bounded), and it adds four parameters (defensibility cost; high-dim). Stressed
is ~flat (+0.86). A real candidate for the calm regime, to validate at higher resolution.

**E1 — clearing in the calibration loop (the caveat).** Hypothesis confirmed. **Calm ≈ E0**
(D 44.45 vs 44.02; θ within noise) — no client freezes fire, so the clearing tier is inert and
the bare-market calibration carries over. **Stressed is materially altered: D 13.84 → 27.98**,
kurtosis 35 → 94, Hill Δ 2.56 → 6.77, KS 6.72 → 14.28. Freezes / deleverage / defaults during
the COVID crash feed back into the price the moments are matched on, and the loop compensates by
thinning ZI flow (`zi_alpha` 0.196 → 0.089, `zi_delta` 0.16 → 0.10). **Implication for the
thesis-final calibration:** the stressed bare-market θ is *not* invariant to the clearing tier —
the market layer calibrated without clearing understates the dynamics the clearing-active model
actually produces under stress. This is a methodology flag, not a "better fit."

**E6 — gaps vs single shock (contagion scenario; Euronext A9 §5).** The clean structural
finding of the campaign. At **equal total drawdown**, a single concentrated shock is far more
destructive than the gapped overnight-limit-down path:

| | gapped reaches | shock reaches |
|---|---|---|
| first client defaults | c≈2.0 (−34%) | **c=1.0 (−19%)** |
| first CM defaults | c≈2.5 (−40%) | **c=1.0 (−19%)** |
| L4 mutualisation | c≈3.0 (−46%) | **c=1.0 (−19%)** |
| client defaults @ −47% | ~17.5 | ~35.5 |

Contagion is driven by the **concentration/speed of the move, not its total magnitude**. The
gapped path delivers the drawdown in overnight steps each marked by the 60-min variation-margin
cycle, so the book de-risks incrementally between jumps; the single shock delivers the whole move
at once, before VM can collect, catching clients and CMs at full exposure → mass default and
mutualisation. The gap *structure* is a mitigant, not an amplifier, relative to an equal-size
instantaneous shock.

## Recommendation

Carry **E2 (`mt_lambda` in the loop)** into the thesis-final calibration: it is the only change
that improves both regimes, it is cheap (one parameter), its component deltas are clean, and it
is well-motivated (the data prefers a marginally slower trend than the pinned 0.05). Treat **E5
(FT/MT gate + cancellation)** as a genuine *calm-specific* candidate worth re-testing at thesis
resolution — it delivers the campaign's best clustering and tail fit, and the D36 rejection no
longer applies post-TTL-removal — but weigh its four-parameter, higher-depth cost against
defensibility before adopting. **Reject E3, E4a, E4b** (no robust gain; E3 wrecks the tail; the
two-cohort design does not revive multi-scale clustering — the documented limit stands). Note for
the record that **E1 shows the stressed calibration is not clearing-invariant** (D 13.8 → 28.0):
the thesis-final stressed run should at minimum report this sensitivity, and ideally calibrate
with the clearing tier active. **E6** is a standalone result for the contagion chapter: cascade
severity tracks move concentration, not total magnitude.

Per the brief, no model defaults were changed — `globals.CALIBRATED` is untouched and nothing was
pushed to git. These are exploration screens; every θ\* lives in its own per-experiment JSON.
