# Thesis writing — working notes (by section)

_**writing.md** (this file) + **writing.tex** are the only two writing files. Both are now ordered by thesis
chapter so you can go through them one section at a time; the matching draft prose lives under the same
chapter headers in writing.tex. **Cross-cutting items are collected at the end** (editorial, independence,
priority, bib keys, status/runs). Deeper reference: `MODEL_DEEP_DIVE.md` (H1 mechanism audit),
`RESULTS_METHODOLOGY.md` (results + methodology)._

_Priority tags: **P1** grade-limiting · **P2** band-lifter · **P3** polish. Chapter word counts: intro 844 ·
related 1933 · model 3809 · calibration 2441 · results 1178 · **conclusion 3 (stub)** · abstract 3
(placeholder) · appendix 0._

---

# Abstract — `chapters/00_abstract.tex`  **(P1)**

- [ ] Write ≤250 words: problem → model → the H1 result → contribution. Uncomment
  `\input{chapters/00_abstract}` in `main.tex`.


# Ch.5 — Results — `chapters/05_results.tex`

**Criterion 4 — Description & analysis of results** *(gap: ch.05 is 1178 words with NO robustness /
alternative-interpretation / limitations text — the rubric's good/very-good bands explicitly want it)* **(P1)**

- [ ] Sync the developed H1 material from writing.tex into ch.05 (the chapter lags the pad).
- [ ] Add a **Robustness & limitations** subsection: seed-count sensitivity (rare member-default /
  mutualisation counts over ~5 members swing with seed count), significance of the H1 trend (p<0.0001) vs the
  within-noise reactive-vs-flat-8 gaps (Welch p≈0.12–0.19), the small NBCM tier, the exogenous-path caveat,
  the fixed-DF/close-out sensitivity levers. Most of this is in `MODEL_DEEP_DIVE.md`.
- [ ] State the corrected H1 mechanism (close-out / loss-concentration, **not** Brunnermeier–Pedersen funding
  drain) — already fixed in writing.tex, mirror into ch.05; strip any "every risk metric" wording too.
- [ ] Remove the stray in-body "Disclaimer" sentence in ch.05.
- Citation: **CME client IM share** — use ~67–70% (record 67.7% cleared-IRS, Jan-2025), NOT ~82%.

**Results to-do / replace** *(folded from the old results.tex checklist):*
```
  [x] H1 prose + table updated to 40-seed numbers; netting finding corrected; mechanism re-attributed
      (close-out, not B-P drain; traced seeds 45/46); reactive overclaim softened (Welch p=0.12-0.19).
  [ ] MIRROR the mechanism + overclaim corrections into thesis ch.05/ch.06 on merge.
  [ ] THESIS ch.05 descriptive numbers — refresh from the overnight run (the pad numbers are pre-cap).
  [ ] THESIS ch.05 "never reaches the mutualised fund": WRONG — NBCM failures reach L3 in ~20% of stressed
      seeds. Scope to "client defaults alone".
  [ ] Re-verify the NBCM-failure waterfall split now that waterfall_events.csv logs L1-L5 across all seeds.
  [ ] Regenerate figures from the refreshed CSVs; copy into Figures/.
  [ ] Synthetic-ensemble section (commented in writing.tex) — fill from output/overnight_joint/ or cut.
```

**Output CSVs → new figures/tables** *(core + H1 now emit these; the base ODD has no client observables):*
- `client_defaults.csv` — per default: client TYPE (FT/MT/ZI), carrying CM + type, position-at-default
  (`assumed_pos`), IM, loss, timing → defaults by type/CM + the tail-concentration mechanism (median ↓ / p99 ↑).
- `waterfall_events.csv` — per default: level + L1–L5 amounts + mutualised → waterfall-level distribution.
- `client_freezes.csv` — freeze onsets: reason (`own_distress` vs `cm_contagion`), κ, type, CM, timing → the
  client→CM contagion channel quantified.
- `porting_events.csv` — per member default: `n_ported` / `n_unported` / unported position (EMIR Art. 48).
- `member_balance_band.csv` (core) — cash / maintenance-margin / DF-contribution bands by member type.
- **Build:** stacked waterfall L1–L5 distribution · client-defaults-by-type bar · contagion-vs-distress freeze
  timeline · porting success rate · member balance-sheet bands.

---

# Ch.6 — Conclusion — `chapters/06_conclusion.tex`

**Criterion 5 — Conclusion & relevance discussion** *(gap: ch.06 is a 3-word stub — the single biggest hole;
the rubric weights "relevance for practice" / "implications for theory, methodology and practice" heavily,
for this thesis read practice = regulatory/industry)* **(P1)**

- [ ] Write the full conclusion (draft scaffold is in writing.tex): summary → each RQ answered → contribution
  to literature → limitations → future work.
- [ ] Add an explicit **relevance / implications** passage ("who wants to know"): CCP risk managers and
  regulators (ESMA/CFTC) — margin-procyclicality buffer design and the client-protection-vs-member-fragility
  trade-off.
- [ ] Reconcile the implications chapter: add a short Discussion/Implications chapter, or fold it into the
  conclusion **and fix the intro `\ref{ch:implications}`**.

---

# Appendix — `chapters/A_appendix.tex`  **(P3)**

- [ ] Populate (robustness tables, the synthetic-ensemble figures, full calibration moments) and uncomment it,
  or drop the appendix refs. Populating it also helps criterion 4.

---
---

# General / cross-cutting

## Editorial quality (criterion 6)  **(P1 for refs, P3 for prose)**
- [ ] Fix broken cross-refs: `\ref{ch:implications}`, `\ref{ch:appendix}`, `\ref{sec:model-default}`.
- [ ] Resolve two terminology inconsistencies: κ labelled a Basel **capital-adequacy** ratio but cited to
  **leverage-ratio** papers (pick one); **five-level** vs **six-tranche** waterfall described differently.
- [ ] Consistent table/figure captions and number formatting across chapters.
- (The abstract is under "Abstract" above; bib hygiene under "Bib keys" below.)

## Degree of independence (criterion 7)  *(process-graded, not a document section)*
Keep the supervisor informed of plans/progress; be ready to **defend your design choices** (the band rewards
"willing to defend own choices"); disclose tool/AI assistance per VU policy (the fail band penalises
*undisclosed* third-party help) — a brief methods/acknowledgement note keeps you safe.

## Priority order (grade-limiting first)
1. **Conclusion chapter** (ch.06 stub → full) + relevance/implications — criterion 5.
2. **Results robustness & limitations** subsection + sync H1 — criterion 4.
3. **Abstract** + **broken refs** + the two terminology inconsistencies — criterion 6.
4. **Explicit research question** in the intro — criterion 1.
- P2 band-lifters: lit gap→contribution paragraph; design-choices reflection.
- P3 polish: bib hygiene, captions, appendix population, prose pass.

## Bib keys to add to references.bib
```
  cont_2001                       — Cont (2001), "Empirical properties of asset returns…", Quant. Finance 1(2).
  adrian_shin_2010                — Adrian & Shin (2010), "Liquidity and leverage", J. Fin. Intermediation 19(3).
  cgfs_2010                       — CGFS (2010), "The role of margin requirements and haircuts in procyclicality", BIS.
  murphy_vasios_vause_2014        — Murphy, Vasios & Vause (2014), BoE Financial Stability Paper No. 29.
  glasserman_wu_2018              — Glasserman & Wu (2018), "Persistence and procyclicality in margin requirements", Mgmt Sci.
  brunnermeier_pedersen_2009      — Brunnermeier & Pedersen (2009), "Market liquidity and funding liquidity", RFS 22(6).
  cont_2017                       — Cont (2017), "Central clearing and risk transformation", Banque de France FSR No. 21.
  biais_heider_hoerova_2016       — Biais, Heider & Hoerova (2016), "Risk-sharing or risk-taking?", J. Finance 71(4).
  duffie_scheicher_vuillemey_2015 — Duffie, Scheicher & Vuillemey (2015), "Central clearing and collateral demand", JFE 116(2).
  menkveld_vuillemey_2021         — Menkveld & Vuillemey (2021), "The economics of central clearing", ARFE 13.
  barndorff_shephard_2001         — Barndorff-Nielsen & Shephard (2001), "Non-Gaussian OU-based models…", JRSS B 63(2). [supOU]
(Already in bib: duffie_zhu_2011, galbiati_soramaki_2013, almgren_chriss_2000, franke_westerhoff_2012.
 WATCH OUT: `bns_2004` is a DIFFERENT Barndorff-Nielsen–Shephard paper — add barndorff_shephard_2001 separately.)
```

## Confirmed accurate — keep as written
Fundamental = empirical mid (ch.3, not Kalman); discrete call-auction LOB; stressed 73 / calm 75 sessions
(~29k obs); loss weighting `w_c = 1/s_c` (per-moment); KS excluded from MCR (with the stated reason); EMIR
Art. 28 APC floor; SPAN/FHS 99% IM; five-level EMIR waterfall ordering; agent counts (30/20/40 + 10 BCM /
5 NBCM / 90 clients).

## Status + overnight runs (operational)
- **Parameters LOCKED.** House-only BCMs follow the 8% capital-adequacy floor under a finite 5× cap
  (`POSITION_LIMIT_X_HOUSE=5.0`, `POSITION_LIMIT_CLIENTS_ONLY=True`, κ ~0.10–0.20) — they run the leverage
  cycle and can fail in deep stress, without the uncapped overshoot. Client-clearing BCMs keep the 2× cap.
  On the re-run the H1 story is clean (member defaults NBCM-only, reactive attractive).
- **House/client split ~50/50** left as a limitation (clients are strategy- not cash-constrained; the order-
  size lever is calibrated). Sentence in writing.tex; doesn't affect loss allocation.
- **Netting DROPPED everywhere** (driver, notebook, writing.tex). One contribution (model) + one result (H1).
- **New per-run CSV logging** added (see Ch.5 "Output CSVs"). **Robustness sweeps (severity/closeout)
  SKIPPED** for time (consistent with the writing).
- **Overnight: 3-parallel** — `nohup bash scripts/run_parallel.sh > output/parallel.log 2>&1 &`
  (defaults `DESC_SEEDS=100 HYP_SEEDS=150 TWOF_HOURS=12`):

  | # | job | script | → output dir |
  |---|---|---|---|
  | 1 | CORE | `run_model_descriptive.py` | `output/results/descriptive/` |
  | 2 | H1 | `run_thesis_experiments.py` | `output/thesis_final/experiments/` |
  | 3 | SYNTH | `overnight_joint.py --regime stressed --log-vol` | `output/overnight_joint/` (D80 log-vol two-factor: clustering + fat-tail repair vs baseline + clearing generalisation) |

  **Morning:** refresh the H1 table + §5.1 numbers (pad numbers are pre-cap); confirm member defaults
  NBCM-only / reactive attractive / client defaults fall; build the new figures; report the synthetic 2f-vs-1f
  repair (`validation.csv`) + ensemble (`clearing.csv`).

- **Synth calibration LOCKED** (`output/overnight_joint_logvol/`): joint 9-D log-vol optimum **D=6.73 vs the
  1-factor baseline 20.86 (−68%)**; out-of-sample (80 paths, 42d) **9/10 moments in the 95% CI**, mean gap
  0.64 SD — clustering + tail repaired; only the lag-1 bounce (`acf_r_1`, 2.7 SD) stays out, from `zi_delta`
  calibrating to its 0.32 ceiling. Written into writing.tex Ch.4 (`tab:cal-synth`).
- **Optional wide-bound re-run** (set up, NOT launched; SEPARATE dir, won't overwrite the locked run): unpins
  `zi_delta` and warm-starts from the locked surface — short, may shave D (likely trading against the bounce).
  `overnight_joint.py` gained opt-in `--out-tag` / `--zi-delta-hi` / `--warm-start` (defaults unchanged).
  ```
  nohup python3 scripts/overnight_joint.py --regime stressed --log-vol \
    --out-tag wide --zi-delta-hi 0.50 \
    --warm-start output/overnight_joint_logvol/calibration.json \
    --max-hours 2.5 --calib-frac 0.8 --skip-clearing \
    --sobol-init 40 --n-path-seeds 3 --path-seed 7 \
    > output/overnight_joint_logvol_wide.out 2>&1 &
  ```
  → `output/overnight_joint_logvol_wide/`. Keep `--n-path-seeds 3 --path-seed 7` to match the warm-start loss;
  compare its `validation.csv` to the locked one before adopting (watch the lag-1 bounce).
