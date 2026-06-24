# Leverage-floor and house-margin analysis

> **Superseded floor (2026-06-19):** this doc recommends the type-differentiated "Option A" floor
> (BCM Basel-LR `cash/exposure≥4.25%`; NBCM Reg 1.17). That was an intermediate step; the **committed**
> scheme unifies client-clearing solvency to CFTC Reg 1.17 `cash/IM≥8%` for **both** tiers, keeps the
> Basel-LR as a **bank-only deleverage trigger**, and resizes NBCMs to $0.5–3B (see
> `FLOOR_BALANCE_RESULTS.md`). The house-margin (15%) analysis below stands.

*Follow-up to `SECOND_READER_REVIEW.md` items H2 (the "8% Basel" floor) and H3 (the 15% house
margin). All numbers measured this session (sandbox, committed config, seed 42, real COVID window).
Regulatory facts verified June 2026.*

---

## TL;DR

1. **The 8% `cash/exposure` floor is the single load-bearing trigger for the COVID-window
   contagion**, not a cosmetic mislabel. It sits just above the endogenous stress κ_min (~0.04–0.06).
   Lowering it to the regulatorily-accurate G-SIB value (~4%) makes the COVID window **completely
   benign** (0 client defaults, 0 breaches). Sid's proposed 5.5% does the same (2 defaults).
2. **That is more realistic, not less:** no major CCP clearing member defaulted in March 2020 — COVID
   was a margin/funding ("dash for cash") event, not a solvency cascade. The 8%-floor result (20–37
   client defaults, ~22–41% of the client book) is arguably *too* severe.
3. **Recommendation:** use accurate, type-differentiated floors and let COVID be benign;
   demonstrate contagion/mutualisation under the deeper GBM/reverse-stress arms (where κ naturally
   falls below ~4%) — this is exactly the "mutualisation is conditional" framing the synthesis
   already recommends. If you want COVID-window contagion to remain, move the trigger from the
   *floor* to *member capital* (thinner, dedicated clearing capital), not a high floor.
4. **15% house margin is a fine value** — it is the clean-calm boundary that best approaches the
   CME ~82% client share — but **re-cite it** (FCM house add-on ≈2× the ES SPAN margin), not the
   security-futures statutory minimum, which does not apply to a broad-based index future.

---

## 1. Current regulatory facts (verified June 2026)

| Constraint | Who | Current value | Form |
|---|---|---|---|
| Basel III leverage ratio (base) | all banks | **3%** | Tier1 / total exposure |
| Basel III G-SIB LR buffer | G-SIBs | +50% of the G-SIB surcharge → total **~3.5–4.25%** | Tier1 / exposure |
| US eSLR (**reformed**, final rule eff. **1 Apr 2026**) | US G-SIB BHCs | **~3.5–4.25%** (was a fixed 5% BHC / 6% IDI) | Tier1 / exposure |
| CFTC Reg 1.17 | **non-bank** FCMs | **8% of risk margin (IM)** ≈ 0.5–1.2% of notional | capital / IM |

So the accurate floor for a **bank** clearing member (G-SIB) is **~4%** on `cash/exposure` — not 8%,
and not "the Basel III leverage ratio" at 8% (that label is ~2.7× the real minimum). The "8%" in the
model actually originates in **Reg 1.17**, which is **8% of IM** (a non-bank rule on a different,
much smaller base) — it was applied to the wrong base. Note also that the Basel LR is a *bank-wide*
constraint (Tier1 / total bank exposure ~$3–4T), whereas the model computes it on the *clearing book
alone* — a further reason to treat the floor as a stylised leverage limit rather than the literal LR.

---

## 2. Floor sweep — the floor is the contagion trigger

Uniform `cash/exposure` floor, seed 42. Only client-carrying CMs ever approach the floor, so a
uniform sweep ≈ the differentiated case for the breach behaviour.

**Full COVID window (sessions 10–30, −25.6% drawdown):**

| floor | breach-cycles | bcm κ_min | client defaults | member defaults | waterfall |
|------:|---:|---:|---:|---:|---:|
| **0.04** (accurate G-SIB) | 0 | 0.058 | **0** | 0 | 0 |
| **0.08** (committed) | 36 | 0.040 | **20** | 0 | 0 |

**Shorter window (sessions 10–22), finer grid:**

| floor | breach-cyc | bcm κ_min | client def |
|------:|---:|---:|---:|
| 0.03 | 0 | 0.060 | 0 |
| 0.04 | 0 | 0.060 | 0 |
| 0.05 | 0 | 0.060 | 1 |
| 0.055 | 0 | 0.061 | 2 |
| 0.06 | 0 | 0.063 | 2 |
| **0.08** | **6** | **0.040** | **15** |

**Reading.** There is a sharp threshold between 0.06 and 0.08. Below ~0.06 the member never breaches,
never deleverages, and the client-default cascade does not occur. At 0.08 the member breaches →
Almgren–Chriss-deleverages its own book into the falling LOB → amplifies the drop (and the
floor-triggered client freezes thin the book) → client VM losses spike → 15–20 client defaults.
**Calm is clean at every floor** (κ stays ≥0.082), so calm-cleanliness does not constrain the choice.

Two consequences:
- The headline "stressed fires the leverage cycle (~37 client defaults)" is an **8%-floor artifact**.
  At the accurate ~4% floor the COVID window produces **zero** member breaches and **zero** client
  defaults.
- This also revises review item **M3**: the forced-deleverage amplification is *not* weak when it
  fires — it is the dominant driver of the cascade (0 → 20 defaults). It is simply gated by the floor.

---

## 3. House-margin sweep — 15% is the clean-calm boundary

Position-opening cap `|notional| ≤ cash / im_percent` (committed 8% floor), seed 42, sessions 10–22:

| house margin | client leverage | calm client share | calm breaches | stress client def |
|---:|---:|---:|---:|---:|
| 12% | 8.3× | **79.4%** | **10** (calm not clean) | 26 |
| **15%** | 6.67× | 74.6% | **0** | 16 |
| 20% | 5.0× | 69.1% | 0 | 1 |

**Reading.** 15% is the lowest house margin (highest client leverage, hence highest client IM share,
closest to the CME ~82%) that still keeps **calm perfectly clean**. At 12% the share rises to ~79%
but calm starts breaching; at 20% calm is clean but the share drops to ~69% and contagion nearly
vanishes. So 15% is a **calibrated boundary**, exactly as the `globals.py` comment hints ("~70% here
is the clean-calm maximum") — its justification is the clean-calm/share trade-off, **not** the
security-futures statute.

---

## 4. Recommendations

### 4.1 Leverage floor — three options, ranked

**Option A (recommended) — accurate, type-differentiated floors; COVID is benign.**
- BCM (bank G-SIB client clearer): `cash/exposure ≥ 4.25%` (Basel G-SIB top bucket / current US
  eSLR upper) — or 5% if you want a hair of conservatism (old eSLR).
- BCM (non-G-SIB, own-account): `cash/exposure ≥ 3%` (Basel base). [Moot — these don't breach.]
- NBCM (non-bank FCM): `cash/IM ≥ 8%` (CFTC Reg 1.17 — the *correct* non-bank rule, and the code
  already supports this form via the legacy `cash/IM` branch). Makes NBCMs liquidity-defaulters
  (default on cash, not ratio) — which is what Nasdaq 2018 / LME 2022 actually were.
- Client freeze: decouple from the member floor — freeze a client at **maintenance-margin** breach
  (`mm_percent`), the FCM cut-off, not a leverage ratio (clients are not banks).
- *Consequence:* COVID is a funding-squeeze (IM ↑1.4–1.6×) but not a solvency cascade — matching
  March 2020. Contagion/mutualisation is then demonstrated in the deeper GBM/reverse-stress arms,
  where κ genuinely falls below ~4%. This is the most defensible path and aligns with the synthesis's
  "resilient baseline, conditional mutualisation."

**Option B — keep 8%, but reframe honestly as a prudential limit.** Present 8% as a
risk-management leverage limit a conservative dealer enforces (a buffer above the ~4% minimum), drop
the "Basel III leverage ratio" and Reg-1.17-on-notional labels, and **disclose the floor sensitivity**
(this §2 table) showing the COVID cascade is a function of the floor. Defensible but weaker — the
headline becomes contingent on a tuned trigger, and a clearing-literate examiner will press on 8%.

**Option C — accurate floor + thin member capital (keeps COVID contagion, honestly).** Use Option
A's floors but move the trigger to the balance sheet: size the client-clearing capital to dedicated
clearing-unit capital (thinner than the parent's $5–10B) so κ_min naturally falls below ~4% in
stress. Then COVID contagion is real *and* the floor is accurate. Requires re-checking calm-clean and
re-tuning cash; target a **realistic handful** of client defaults (not 22–41% of the book).

On Sid's specific proposal (client-BCM 5.5% / target 8%, others lower): the differentiation is the
right idea, but **5.5% makes COVID benign** (2 client defaults) — it behaves like Option A, not like
a behaviour-preserving tweak. If the goal is to keep the COVID cascade, 5.5% won't do it; use Option
C. If the goal is accuracy, 5.5% is fine (slightly conservative vs the ~4% current requirement) and
you should embrace the benign-COVID result.

### 4.2 House margin

- **Keep 15%.** It is the clean-calm boundary that best approaches the CME client share.
- **Re-cite** as a representative FCM **house-margin add-on** — FCMs set house margin above the
  exchange SPAN minimum (NFA Rule 2-26; FCM customer agreements), here ≈2× the ES SPAN maintenance
  margin (~5–7%). **Drop** the security-futures statutory citation (17 CFR 242.400-406) — ES is a
  broad-based index future, not a security future, so that rule does not apply.
- **Add the §3 sensitivity table** to the thesis: it both justifies 15% and shows the
  share/contagion trade-off.
- *Optional, more realistic:* tie the house margin to a multiple of the CCP VaR margin
  (`house_im = m · im_fraction(σ)`, m≈2), making the client cap procyclical (it already rises in
  stress in reality). Cleaner citation (FCM margin-multiplier practice) and removes the free 15%
  parameter — but it is procyclical, so re-check the calm-clean/contagion behaviour before adopting.
- *Note the redundancy:* the static 15% opening cap and the procyclical escrowed IM (6–30%) are two
  margin concepts for one client. The simplest conceptually-clean alternative is a single house
  margin = `max(SPAN VaR, house floor)` used both to cap the position and as the posted IM. Keeping
  them split is fine and has a purpose (static max leverage vs the dynamic funding drain) — just
  disclose it.

---

## 5. The one decision for Sid

**Does the COVID window need to show contagion, or is a benign COVID acceptable?**

- *Benign COVID acceptable* → **Option A** (accurate ~4% BCM floor + Reg-1.17 NBCM). Most defensible,
  matches history, strengthens the conditional-mutualisation framing. Re-run the suite; expect the
  contagion to live in the deeper GBM/reverse-stress arms.
- *COVID contagion wanted* → **Option C** (accurate floor + thinner clearing capital), tuned to a
  realistic handful of client defaults. Avoid keeping 8% dressed as Basel (Option B) unless you fully
  own it as a tuned prudential limit with the sensitivity disclosed.

Either way: fix the citations (floor ≠ "Basel LR at 8%"; house margin ≠ security-futures statute),
and report the §2 and §3 sensitivity tables — they convert two "magic numbers" into defensible,
evidenced choices.

## Sources
- [Basel III leverage ratio framework (BIS) — 3% base + G-SIB buffer = 50% of surcharge](https://www.bis.org/publ/bcbs270.htm)
- [Federal Reserve — final eSLR rule, effective 1 Apr 2026 (surcharge-linked ~3.5–4.25%)](https://www.federalreserve.gov/newsevents/pressreleases/files/bcreg20251125b2.pdf)
- [OCC Bulletin 2025-41 — eSLR final rule](https://www.occ.treas.gov/news-issuances/bulletins/2025/bulletin-2025-41.html)
- [CME — E-mini S&P 500 margins (SPAN ~3–12% of notional)](https://www.cmegroup.com/markets/equities/sp/e-mini-sandp500.margins.html)
- [CME — Customer margining (gross customer margining; house/customer segregation)](https://www.cmegroup.com/education/articles-and-reports/customer-margining-at-cme-clearing.html)
- Sandbox measurements: `floor_sweep.py`, `verify_rebalance.py` (HOUSE_MARGIN sweep), this session.
