# Variation-margin design check + VOLUME_LOT empirical-volume audit

*Date 2026-06-18. Measurements this session.*

---

## 1. Variation margin — the average-fill-price design is correct

The per-cycle VM (`simulation.py` Phase 0/1/2) is
```
vm = ( pos·(mid − last_mark) + last_mark·fill_qty − fill_cost ) · usd
```
with `_fill_qty`/`_fill_cost` the signed quantity and signed cash of fills since the last mark
(reset each cycle), `last_mark` the prior cycle's mid. This is **algebraically exact** M2M P/L:
decompose the book into the position held at `last_mark` and the fills during the cycle. The held
part earns `(pos − fill_qty)·(mid − last_mark)`; each fill earns `Σ q·(mid − fill_price) =
mid·fill_qty − fill_cost`. Summing:
```
(pos − fill_qty)(mid − last_mark) + mid·fill_qty − fill_cost
   = pos·(mid − last_mark) + last_mark·fill_qty − fill_cost.   ✓ (matches the code exactly)
```
So a position is marked from the last mark to the new mid, and **each fill is marked from its own
fill price** — i.e. the average fill price is handled correctly even with multiple fills at different
prices in one cycle (it is `fill_cost / fill_qty`). Spot check: flat, buy `q` at `p`, mid → `m`:
`vm = q(m − last_mark) + last_mark·q − p·q = q(m − p)` ✓. Held position, no fills:
`pos·(m − last_mark)` ✓. **No change needed** — the design is sound and matches the ODD's
`P/L = N·(P_market − P_filled)`. (Cadence: marks at the hourly cycle mid; fills within the cycle are
marked to that mid — standard periodic VM.)

---

## 2. VOLUME_LOT vs empirical volume — it does NOT currently hold; the target is also inconsistent

`VOLUME_LOT` (calm 30 / stressed 60) relabels one model lot as N ES contracts so simulated
per-minute contract volume matches empirical ES. It is a **reporting/notional scale only** — it does
not enter the matching engine, so returns/ACFs/Hill are invariant.

**Measured (seed 42):** sim contract volume = mean model-lots/min × VOLUME_LOT:

| regime | sim lots/min | × VOLUME_LOT | sim contracts/min | `EMP_VOL` const | real RTH front-month | 24h OHLCV |
|---|---:|---:|---:|---:|---:|---:|
| calm | 113 | ×30 | **3 395** | 2 524 | **2 118** (ES_front) | 1 620 |
| stressed | 110 | ×60 | **6 580** | 4 979 | — (file volume = 0) | 2 602 |

Two problems:
1. **The mapping overshoots.** Sim runs ~1.35× the `EMP_VOL` constant and ~1.6× the real RTH
   front-month (calm 3 395 vs ES_front 2 118). The agent flow rose ~35% since 30/60 was set (the D67
   relock raised `zi_alpha`/`p_zi`, so more limit orders → more fills), and the lot map was not
   re-matched.
2. **The empirical target itself is inconsistent.** `EMP_VOL` (2 524 / 4 979 in
   `analyze_volume_margin.py`) sits ~20% above the cleaned RTH front-month series
   (`data/processed/ES_front_calm_1m.csv` ≈ 2 118/min) and well above the 24h OHLCV mean. And the
   **stressed `ES_front` file has its volume column all zeros** (a data-pipeline bug) — so the
   stressed empirical target can't currently be verified from the front-month series at all.

**Why this is safe to fix (the key invariance).** The clearing tier is leverage-based: every position
is capped in *notional* (clients by the house margin, BCMs by `POSITION_LIMIT_X`), VM = notional·return,
IM = `im_frac`·notional, κ = cash/notional. Notional = lots·`VOLUME_LOT`·`CONTRACT_USD`·mid. So if you
scale **`VOLUME_LOT` and every cash band by the same factor α**, leverage, κ, IM share, defaults and
waterfall depth are all **identical** (every dollar amount scales by α, every ratio is unchanged), and
only the reported contract volume scales by α. So matching volume is a behaviour-neutral relabeling
*provided cash is scaled with it*.

**Resolved (2026-06-18).** Targets set to the clean RTH front-month round numbers **calm 2,000 /
stressed 3,500** per minute. Sim runs **113 / 108 model-lots/min** (stable across seeds), so
**`VOLUME_LOT = 18 / 32`** contracts per lot (`globals.py`; `EMP_VOL` in `analyze_volume_margin.py`
updated to 2,000 / 3,500). Verified: sim contract volume **2,119 / 3,592 per min** (within ~3–6% of
target).

**No cash rescale was needed — the change is behaviour-neutral in practice.** The position caps are in
*notional* (max client notional = cash / `im_percent`, independent of `VOLUME_LOT`; same for the BCM
own-book cap), so the cap-bound positions that dominate the margin simply hold the same notional in
*more* lots. Measured before/after (seed 42): total IM ≈ \$22B calm / \$42B stressed (was \$22B / \$44B),
client IM share ≈ 73% (unchanged), calm clean (0 defaults), stressed contagion and procyclicality
(~1.9×) essentially unchanged. So `VOLUME_LOT` matches empirical volume while leaving every clearing
result intact — exactly the behaviour-neutral relabeling the invariance above predicts (the cash bands
keep their cited, realistic FCM/asset-manager scale). (The stressed `ES_front` volume column is still
zeroed — a data-pipeline bug worth fixing if you want to re-derive the stressed target from the
front-month series rather than the round 3,500 used here.)

---

## 3. Where to write up VOLUME_LOT (suggested placement)

The institutional-lot map already appears implicitly in `clearing_writeups.tex` Eq.~(1) as the factor
$\ell$ in $E_\tau=(|N^{\mathrm{own}}|+\sum_c|N^c|)\,\ell\,\chi\,m_\tau$ but is never defined. The
cleanest home is a one-sentence definition (or footnote) right where $E_\tau$ is introduced in §A —
it is a notional-scaling convention, so it belongs with the notional definition, not the margin
mechanics. Suggested snippet:

```latex
% footnote or sentence at E_tau in Eq (1):
Here $\ell$ is the \emph{institutional-lot map}: one model order unit is read as a block of $\ell$
E-mini S\&P~500 contracts ($\ell$ set per regime), $\chi=\$50$ is the CME point multiplier, and
$m_\tau$ the index level, so one model-lot point of exposure is $\ell\,\chi\,m_\tau$ dollars. The
map enters only the reported volume and the USD notional (hence margin and the leverage ratio), never
the matching engine, so the calibrated price dynamics --- returns, autocorrelations, tail index ---
are invariant to it; $\ell$ is fixed per regime ($\ell=18$ calm, $32$ stressed) so that the simulated
per-minute contract volume matches the empirical RTH front-month ES volume ($\approx2{,}000$ calm,
$\approx3{,}500$ stressed).
```

(Concrete values for your write-up: order size is uniform $U[1,10]$ model units; lot size $\ell=18$
calm / $32$ stressed; simulated volume $2{,}119 / 3{,}592$ per minute vs empirical $2{,}000 / 3{,}500$.)

(Update the empirical numbers in the prose once the target is settled per §2.)
