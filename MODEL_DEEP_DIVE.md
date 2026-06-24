# Model deep-dive — adequacy, H1 interpretation, robustness, design

Critical review of the ABM and the H1 (margin-procyclicality) result. Everything
below is checked against the 40-seed `output/thesis_final/experiments/rows.csv`,
the descriptive 60-seed run, and the clearing code (`model/simulation.py`,
`model/clearing.py`). Numbers are reproducible from those files.

**Bottom line.** The model is well-calibrated and the *core* H1 trend is statistically
solid, but two things in the current write-up overclaim: (1) "reactive out-performs
flat-8% on **every** risk metric" is **not** statistically supported — the only robust
reactive advantage is cost; and (2) the stated mechanism for rising member fragility —
"over-margining drains member liquidity" (Brunnermeier–Pedersen) — is **mechanically
wrong for this model**. The real channel is close-out loss on large assumed books, and
the rise with margin is a *tail-concentration* (freeze-trap) effect. These are fixable in
the text without new runs.

---

## 1. Is H1 robust?

Per-arm, stressed, n=40/arm:

| metric | flat-4% | flat-8% | flat-12% | reactive |
|---|---|---|---|---|
| client defaults | 15.2 (sd 3.8) | 11.9 (4.9) | 6.5 (4.1) | 10.2 (4.8) |
| member defaults | 0.00 (0.00) | 0.38 (0.54) | 0.57 (0.64) | 0.23 (0.48) |
| reach fund (L≥3) | 0% | 35% | 47% | 20% |
| peak-to-trough | −35.7% | −35.7% | −35.6% | −35.6% |

**What is robust.** The flat-ladder trend is highly significant. flat-4% → flat-12%:
client defaults fall (Welch p < 0.0001), member defaults rise (p < 0.0001), mutualisation
rises (p < 0.0001). The central H1 statement — *more standing margin moves loss from
clients up the waterfall, at rising cost* — holds firmly. The cost ladder ($16B → $28B →
$38B standing) is deterministic, not noisy.

**What is NOT robust — "reactive beats flat-8% on every risk metric."** Welch tests,
reactive vs flat-8%:

| | reactive | flat-8% | diff | p | verdict |
|---|---|---|---|---|---|
| client defaults | 10.2 | 11.9 | −1.7 | 0.124 | n.s. |
| member defaults | 0.23 | 0.38 | −0.15 | 0.193 | n.s. |
| reach fund | 0.20 | 0.35 | −0.15 | 0.137 | n.s. |

None of the three risk-metric advantages clears significance at n=40. Reactive and
flat-8% are **statistically indistinguishable on risk.** The defensible claim is:
*reactive achieves the same stress-time protection as flat-8% at roughly half the
standing collateral cost ($16B vs $28B)* — the cost saving is the robust win, the
risk-metric "domination" is within sampling noise. **→ Reword the conclusion (writing.tex
~l.51) and §5.2 accordingly: "matches flat-8% protection at flat-4% standing cost," not
"out-performs on every risk metric."**

**Exogeneity confirmed.** The realised path is identical across all four arms
(−35.65%, SD ≤ 0.08). Margin has *zero* feedback into the price — structurally enforced
because fundamentalists anchor the mid to the exogenous $V_t$. This is the right basis for
the "collateral-demand, not amplification" caveat, but it also means H1 *cannot* speak to
the procyclical feedback loop that is the actual systemic concern (see §4).

**Small-N caveat on the systemic metrics.** Member defaults and mutualisation are rare
events over only 5 NBCMs; per-arm means sit at 0.2–0.6 with SDs of similar size. The
*direction* across the flat ladder is significant; individual arm-to-arm gaps (esp.
reactive vs flat-8%) are not. Report these as event frequencies with care, not point
estimates. (This is the same lesson as the netting "doubling" that vanished from 12 → 40
seeds.)

---

## 2. How H1 is interpreted — the mechanism is mis-attributed

The conclusion (writing.tex l.47–49) reads: *"the posted collateral is cash the member
must lock up and cannot deploy, so over-margining drains the very balance-sheet liquidity
it needs to absorb a client failure — the funding-liquidity channel of Brunnermeier–
Pedersen."* **The model does not work this way.** Three findings, in increasing force:

**(a) NBCMs post no own margin.** Member IM is charged on the member's *own* book:
`_post_im(cm, im_frac * own_notional)` (simulation.py l.685). An NBCM has no own
account, so `own_notional = 0` and it posts **zero** IM from its own cash, at every margin
level. Higher flat margin cannot "lock up" NBCM liquidity — there is none to lock up.
BCMs *do* post own IM and *do* feel the funding drain, but BCMs **never default** in any
arm (all member defaults are NBCM). So the funding-drain → default chain is absent.

**(b) Under high margin, NBCMs absorb ≈ $0 of client VM shortfall.** In rows.csv,
`client_loss_absorbed` is ≈ 0 at flat-8/12 (client IM fully covers the VM shortfall) yet
member defaults are *highest* there; at flat-4 the absorbed loss is positive but member
defaults are *zero*. So absorbed client losses don't drive member defaults either.

**(c) The real channel — close-out loss on assumed books.** On a client default the
carrying member *assumes* the client's position (`cm.inventory += pos`, l.622/634);
next mark-to-market the assumed inventory bleeds VM in the falling market and the deficit
adds a close-out haircut `(1−recovery)·|book|·USD·mid` (l.702). Tracing seed 45:

- **flat-12%:** two NBCMs default carrying **1,173** and **3,902-lot** assumed books;
  deficits **$0.76B** and **$2.58B** — dominated by the close-out haircut, not VM.
- **flat-4%, same seed:** **zero** member defaults.

**Why it rises with margin — a tail-concentration (freeze-trap), not a drain.** Higher
IM pushes clients to the capital floor earlier, so most freeze/default *smaller*, but a few
get trapped frozen holding *very large* positions that default as concentrated losses.
Position-at-default distribution shifts accordingly (same client-default count, different
shape):

| seed | flat-4% median / max | flat-12% median / max | >1,800-lot tail |
|---|---|---|---|
| 45 | 937 / 1,824 | 514 / **3,886** | only at flat-12% |
| 46 | 602 / 1,557 | 352 / **4,856** | only at flat-12% |

High margin **compresses the body and fattens the tail** of the default-size
distribution. The rare giant frozen default (≈3,900–4,900 lots) is assumed by a
thinly-capitalised NBCM, can't be ported (other members are also stressed), and the
one-shot close-out haircut on the warehoused book exhausts it. That — not funding drain —
is what makes member defaults rise with margin.

**→ Re-attribute the mechanism.** This is a loss-allocation / close-out-cost channel
(closer to Cont 2017 risk *transformation* and the auction/porting literature) plus a
margin-induced *concentration* of tail risk. It is arguably a **more interesting** result
than the textbook funding story — higher pre-positioned margin protects the average client
but concentrates the surviving tail into fewer, larger, harder-to-port defaults. But it
must be stated as what it is. Drop or heavily qualify the Brunnermeier–Pedersen attribution
for the member-default result (B–P still legitimately frames the *collateral-demand /
cost* side).

---

## 3. Model adequacies (what is solid)

- **Market-layer calibration.** Surrogate-assisted SMM reproduces ES heavy tails and the
  calm/stressed regimes with good fit; stylised facts validated. This is the model's
  strongest, most defensible component and the genuine contribution.
- **Clearing architecture.** The two-tier BCM/NBCM/client structure with a full five-level
  EMIR waterfall, escrowed IM seized first, SITG, and cover-2 default fund is faithful to
  the ODD/Deloitte references and is, as claimed, an unusual end-to-end coupling of a
  realistic LOB to a complete waterfall with a client tier.
- **Internal consistency under stress.** The descriptive run behaves qualitatively like
  March 2020 — members ride the floor but survive, loss localises at the carrying member,
  mutualisation only via the rare NBCM failure. The risk-*transformation* (counterparty →
  liquidity) narrative is borne out by the mechanics, not just asserted.
- **Clean experimental control.** Identical price path across arms isolates the margin
  treatment cleanly (the flip side of the exogeneity limitation).

---

## 4. Model inadequacies / design

**Central limitation — exogenous price (no margin→price feedback).** Confirmed: drawdown
identical across arms. Real procyclicality is a *loop* (margin spike → forced sales → price
fall → larger spike). The model has the first half (close-outs do walk the LOB) but the
fundamental anchor pins the path, so the loop never closes. H1 therefore measures
collateral *demand* and loss *allocation*, and cannot adjudicate the destabilisation
question that motivates the procyclicality literature. The write-up says this — but it
bounds the *policy* reach of H1 more than the current framing admits.

**Single fixed path → no scenario uncertainty.** Seeds vary only the agent RNG against one
replayed crash. The findings are conditional on *this* COVID path; "probability of a member
default" is not yet a meaningful quantity. (The synthetic-path ensemble in future work is
the right fix and would convert these into probabilistic statements.)

**The member-default result is design-sensitive.** It hinges on mechanics that are set, not
estimated: the assume-and-warehouse close-out, finite porting capacity, the `CLOSEOUT_
RECOVERY` haircut applied as one lumpy mark at the crash mid, and the **fixed-percentage
ODD default fund** used for H1 (the SLOIM fund mechanically vanishes once flat margin
exceeds the stress move, which is *why* the driver forces the fixed fund). The
mutualisation frequencies are conditional on these choices and should be presented as
sensitivity levers, not point predictions. A `CLOSEOUT_RECOVERY` / porting-capacity sweep
would show how much of the "more margin → more member failure" result survives.

**Small NBCM tier (n=5) + rare events.** Systemic metrics are inherently noisy; arm-to-arm
gaps below the flat-4→flat-12 span are within sampling error (see §1). More NBCMs or more
seeds would tighten this.

**Single-asset netting.** Net margining can't expose fellow-customer contamination (the
real risk of omnibus netting), so the efficiency gain (≈0.6×) is an upper bound and the
resilience cost is out-of-model — correctly flagged, but it means the netting paragraph is
a *capital-efficiency* result, not a resilience verdict.

**Framing/labelling issues to reconcile (from the thesis scan):**
- κ = cash/exposure is labelled a Basel capital-adequacy ratio but cited to leverage-ratio
  papers — pick one framing and make the constraint match it.
- Five-level waterfall vs six-tranche descriptions appear in different chapters.
- Client share ≈ 57% vs the ≈ 67–70% real-world split — the house/client margin balance is
  slightly off and worth a sentence.
- Volatility clustering is matched less tightly than the tails (single-timescale SV
  fundamental + LOB attenuation) — already noted; keep as a calibration caveat.

---

## 5. Recommendations

**Fix in the text now (no new runs):**
1. Soften "reactive beats flat-8% on every risk metric" → "matches flat-8% protection at
   ~half the standing cost" (writing.tex l.51; §5.2). Risk-metric gaps are n.s.
2. Re-attribute the member-default mechanism: close-out loss on large assumed books +
   margin-induced tail concentration, **not** Brunnermeier–Pedersen funding drain
   (writing.tex l.47–49). Keep B–P for the cost/collateral-demand side only.
3. Add one sentence that member-default and mutualisation gaps below the flat-4→flat-12
   span are within sampling noise over 5 NBCMs.
4. State that the high-margin mutualisation result is conditional on the fixed-% default
   fund and the close-out/porting mechanics (sensitivity levers).

**Verify when convenient (cheap):**
5. Pool position-at-default over more seeds to make the tail-fattening claim quantitative
   (median ↓, p99/max ↑ with margin) — two seeds already show it cleanly.
6. One `CLOSEOUT_RECOVERY` ∈ {0.8, 0.9, 0.95} sweep to bound the member-default result.

**Already correct / keep:** the flat-ladder trend, the exogeneity caveat, the netting
efficiency-as-upper-bound framing, the "risk relocated not removed" headline.
