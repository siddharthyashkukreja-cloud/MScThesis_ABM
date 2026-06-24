# Rebalancing the CM floor/capital scheme — who defaults, and the fix

*Addresses the concern that the same tier (small NBCMs) kept defaulting, that NBCMs should clear real
volume, and that the BCM leverage measure was confounded by the house book. Investigated by
simulation. Numbers measured this session (`floor_balance.py`). Date 2026-06-19.*

---

## The problem

Under the type-differentiated floors, deep stress only ever toppled the **NBCMs**. Two suspects:
(i) the BCM floor was the Basel/eSLR leverage ratio on **own + client** exposure — it includes the
house book, so it is not a clean client-clearing measure; (ii) NBCMs were tiny ($50M–1B), so a
client fire-sale loss was a large fraction of their capital. We disentangled the two by simulation.

## The experiment

Four schemes, deep reverse-stress (c=2.2 on the COVID path, ≈ −45%), seeds {42, 7, 123}, member
defaults counted by tier (summed over seeds):

| scheme | BCM client-clearing floor | NBCM capital | **BCM def** | **NBCM def** | reading |
|---|---|---:|---:|---:|---|
| A_current | Basel/eSLR cash/exposure (incl. house) | $50M–1B | 1 | 8 | NBCM-only (biased) |
| C_unified | **Reg 1.17 cash/IM on client book** | $50M–1B | 1 | 8 | **same as A** |
| B_nbcm_big | Basel/eSLR cash/exposure | $0.5–3B | 0 | 1 | nobody defaults (too safe) |
| **D_unified_big** | **Reg 1.17 cash/IM on client book** | **$0.5–3B** | **2** | **2** | **balanced** ✓ |

**The decisive finding:** the de-confounding alone (A→C) does **not** change who defaults — the bias
is **capital size**, not the ratio form. Small NBCMs run out of cash absorbing a client loss; $5–10B
BCMs do not, regardless of how their floor is written. Raising NBCM capital alone (B) over-corrects
(no one defaults). Only **both together (D)** balances it: comparable BCM/NBCM default counts, with
contagion still present (waterfall to L3).

## The committed fix (scheme D + bank Basel LR)

Three changes, all in `globals.py` (reversible via the flags):

1. **`UNIFIED_CLIENT_FLOOR = True`** — both tiers' **client-clearing solvency** is the same FCM rule,
   **CFTC Reg 1.17** (`cash/IM ≥ 8%` on the client book). This de-confounds the bank's house account
   from its client-clearing measure and makes the two tiers directly comparable.
2. **`NBCM_CASH_RANGE = $0.5–3B`** (was $50M–1B) — the substantial non-bank clearers that actually
   carry volume (Marex, ABN AMRO Clearing, Clear Street, StoneX). NBCMs now clear large client books
   (leverage-balanced assignment routes more to them) **and** fail in balance with BCMs.
3. **`BASEL_LR_BCM = True`** — a banking member is still a bank: on top of the Reg 1.17 client floor it
   is held to the **Basel III / US-eSLR leverage ratio** (`cash/exposure(own+client) ≥ 4.25%`), which
   triggers its own-book **deleverage** (the leverage cycle, Haynes-McPhail-Zhu). NBCMs (non-banks)
   are exempt. This **keeps** the bank leverage cycle the unification alone would have dropped, without
   re-confounding the client-clearing solvency measure — so we get comparability *and* the HMZ channel.
   Verified: the BCM Basel LR fires under deep stress (`cash/exposure → 0.03`, banks deleverage); banks
   then mostly deleverage-and-survive (the realistic response) rather than defaulting first.

## What is preserved (the realism anchors hold)

Measured under D (seed 42): **calm clean** (0 defaults); **actual COVID benign** (2 client defaults,
0 member defaults, waterfall 0); **client IM share ≈ 70–75%**. So the fix changes *who fails under
deep stress* (now balanced across tiers) without disturbing the validated baseline.

## Why this is the right call

- It removes a genuine **artifact** (one tier always failing because it was tiny), confirmed to be
  size-driven, not rule-driven.
- It makes NBCMs realistic volume-clearers, as in the real market.
- It **simplifies** the story: both are FCMs under Reg 1.17 for client clearing; banks additionally
  run a house book. Cleaner than the two-different-ratios framing.
- Defaults are now **path-dependent across tiers** — so H1 ("where does the loss land?") can study
  *both* a bank member and a non-bank member failing, not just NBCMs.

## Reproduce
`python3 floor_balance.py` (env `SEEDS`, `C`; schemes A/B/C/D). Toggle in `globals.py`:
`UNIFIED_CLIENT_FLOOR`, `NBCM_CASH_RANGE`.
