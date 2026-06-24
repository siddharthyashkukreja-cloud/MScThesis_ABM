# Model parameter inventory

Complete list of every parameter, its role, whether it is **calibrated** (estimated to data /
stylised facts) or **fixed** (set from regulation, the Simudyne ODD, public CCP disclosures, or a
direct measurement), and — for fixed parameters — the value and its interpretation. Source of truth:
`model/globals.py`, `model/globals.CALIBRATED`, `output/v_gbm_params.json`, and the joint log-vol
calibration `output/overnight_joint_logvol/calibration.json`.

---

## 1. Market layer — behavioural agent parameters (calibrated)

Seven parameters per regime, estimated by surrogate-assisted SMM (XGBoost surrogate → grid/active
learning) against the empirical ES stylised-facts loss. Source: `globals.CALIBRATED`.

| Parameter | Description | Status | Calm | Stressed |
|---|---|---|---|---|
| `ft_sigma_c` | FT belief-width scale (× daily V_t σ); dominant tail / clustering lever | Calibrated | 0.642 | 0.453 |
| `zi_alpha` | ZI limit-order arrival rate (per step) | Calibrated | 0.544 | 0.364 |
| `zi_mu` | ZI market-order arrival rate | Calibrated | 0.016 | 0.144 |
| `zi_delta` | ZI per-resting-order cancellation rate | Calibrated | 0.067 | 0.232 |
| `p_zi` | geometric limit-placement depth, k ∼ Geom(p_zi) | Calibrated | 0.460 | 0.512 |
| `mt_lambda` | MT EWMA decay (trend horizon; small = long memory) | Calibrated | 0.023 | 0.181 |
| `mt_gamma` | MT tanh-activation strength (share of MTs acting vs trend) | Calibrated | 0.310 | 0.324 |

---

## 2. Market layer — fixed structural / population parameters

| Parameter | Description | Status | Value | Interpretation |
|---|---|---|---|---|
| `n_fundamental / n_momentum / n_zi` | LOB population | Fixed | 40 / 20 / 40 | 100-trader book (calibration); in the clearing run the 40 FT = 30 FT-clients + 10 BCM |
| clearing population | members + clients | Fixed | 10 BCM (5 client-clearing), 5 NBCM, 90 clients | ODD star topology extended with a client tier |
| `ft_alpha` / `mt_alpha` | FT/MT trade every step | Fixed | 1.0 | replace-on-new, no dead-band |
| `ft_delta` / `mt_delta` / `mt_mu` | FT/MT cancellation & MT market branch | Fixed | 0.0 | limit-only, replace-on-new |
| `mt_eps` | \|M_t\| floor | Fixed | 1e-6 | skips EWMA warm-up |
| `qty_min` / `qty_max` | order size U[1,10] | Fixed | 1 / 10 | ODD §Stochasticity |
| `tick_size` | minimum price increment | Fixed | 0.25 | CME ES tick (= \$12.50/contract) |
| `dt_minutes` | step cadence | Fixed | 1.0 | 1-min ODD-native frequency |
| `VOLUME_LOT` | ES contracts per model lot | Fixed | 18 (calm) / 32 (stressed) | scales sim volume to empirical ES (≈2,000 / 3,500 contracts·min⁻¹) |
| `CONTRACT_USD` | ES contract multiplier | Fixed | 50.0 | \$50 per index point (CME) |
| `V0` | window start price | Fixed | 3016.6 / 3387.4 | first mid of the calm / stressed window |
| `SIGMA_V` | per-min realised vol of V_t (also MT reference σ) | Measured | 2.4e-4 / 1.24e-3 | 1-min intraday return std per regime |

---

## 3. Fundamental-value process (SV-MJD synthetic path)

### 3a. Measured baseline (realized-measures estimation; `output/v_gbm_params.json`)

Used when the path is generated without overrides. Estimated directly from ES 1-min data
(Barndorff-Nielsen–Shephard bipower split + Mancini jump count). Stressed values shown.

| Parameter | Description | Status | Stressed value | Interpretation |
|---|---|---|---|---|
| `sigma_d` | diffusion (bipower) vol, per min | Measured | 1.224e-3 | jump-robust 1-min vol; anchors E[σ²]=σ_d² |
| `mean_return` (μ) | mean intraday log-return | Measured | −3.3e-6 | (superseded by the drift target below) |
| `jump_lambda_day` | Merton jump intensity | Measured | 1.32 /day | Mancini (2009) threshold count |
| `jump_delta` | jump-size std | Measured | 4.32e-3 | jump magnitude |
| `overnight_pool` | empirical overnight gaps | Measured (bootstrap) | n=72 | nonparametric resample at session boundaries |
| `alpha`, `sigma_vol` | single-factor OU vol | Measured | 1.78e-3, 4.04e-5 | method-of-moments on bipower σ series |

### 3b. Synthetic-fundamental calibration (current best: log-vol two-factor, jointly with agents)

The synthetic fundamental is jointly calibrated with the agents (`scripts/overnight_joint.py
--log-vol`). Stressed optimum from `output/overnight_joint_logvol/calibration.json` (D 20.0→15.5).

**Free (jointly calibrated):**

| Parameter | Description | Stressed optimum |
|---|---|---|
| `s_x²` (`f`) | total log-vol variance (vol-of-vol amplitude) → drives clustering **and** tail | 0.80 |
| `w` | slow factor's share of the log-vol variance | 0.11 |
| `mt_lambda` | momentum horizon (re-calibrated jointly) | 0.18 |
| `mt_gamma` | momentum activation (re-calibrated jointly) | 1.37 |
| `ft_sigma_c` | FT belief width (re-calibrated jointly) | 0.54 |
| `delta_mult` | Merton jump-size multiplier | 1.42 → **recommend 0 (drop jumps)** |

**Fixed in the synthetic-fundamental calibration:**

| Parameter | Status | Value | Interpretation |
|---|---|---|---|
| `alpha_fast` | Fixed | 1.78e-3 (measured) | fast OU mean-reversion rate |
| `ratio` | Fixed | 0.10 | slow factor 10× slower than fast (α_slow = ratio·α_fast) |
| `lam_mult` | Fixed | 1.0 | jump intensity at the measured Mancini value |
| `log_vol` | Fixed | True | OU on log-volatility (strictly positive, no clamp) |
| drift `μ` | Fixed | −2.13e-5 /min | set so the path declines ~30% net over a ~2-month (42-session) window, **incl. overnight** |
| `sigma_d` | Fixed | 1.224e-3 | anchors E[σ²] = σ_d² |
| ZI rates (`zi_alpha/zi_mu/zi_delta/p_zi`) | Fixed | `CALIBRATED` | held at the empirical-path calibration (L2-identified) |

---

## 4. Clearing tier — fixed regulatory / ODD / disclosure constants

None of these is SMM-calibrated; each is set from regulation, the Simudyne ODD, or public CCP data.

### 4a. Initial margin (procyclical VaR / SPAN)

| Parameter | Description | Value | Interpretation |
|---|---|---|---|
| `IM_CONF_Z` | IM VaR multiplier | 3.0 | empirical FHS 99% quantile of standardised ES returns (fat-tail; > Gaussian 2.33). EMIR Art. 41 |
| `IM_MPOR_DAYS` | margin period of risk | 1 day | CME/CFTC liquidation horizon for liquid ETD (EMIR 2-day = OTC min) |
| `IM_FLOOR` | anti-procyclicality IM floor | 0.04 | ≈ CME ES maintenance margin / notional; EMIR Art. 28 APC floor |
| `IM_CAP` | IM ceiling | None | no cap (reactive VaR self-bounds ~11% at 1-day MPOR); EMIR APC are floors, not caps |
| `IM_MODE` | margin driver | "reactive" | EWMA-VaR baseline; flipped to flat 4/8/12% in the H1 experiment |
| `IM_FLAT_FRAC` | flat-IM comparator | 0.05 | H1 flat arm (also 0.04/0.08/0.12 in the experiment) |
| `IM_DAILY` / `IM_DAILY_LAMBDA` | daily close-to-close EWMA | True / 0.94 | RiskMetrics convention; warmed on pre-window history |
| `IM_VOL_HALFLIFE` | EWMA half-life | 11 days | RiskMetrics λ≈0.94 (real-CCP reactivity) |
| `IM_ESCROW` | physical IM escrow | True | IM moved as cash to the CCP; funding-squeeze channel |
| `IM_INCLUDE_GAPS` | overnight gap coverage | True | adds a gap-EWMA to the daily variance |
| `im_percent` | client house-margin position cap | 0.15 | 15% of notional (~6.7× leverage); 17 CFR 242 security-futures minimum |
| `mm_percent` | maintenance-margin threshold | 0.95 | CME methodology |

### 4b. Default fund (cover-2) and waterfall

| Parameter | Description | Value | Interpretation |
|---|---|---|---|
| `cover_number` | cover-N standard | 2 | two most-exposed groups (EMIR / Dodd-Frank) |
| `df_buffer` | DF buffer on cover-2 | 0.10 | Euronext Clearing A9 §3 (×1.10) |
| `ex_df_ratio` | SITG as share of the fund | 0.03 | CME/ICE empirical (~2–3% of prefunded fund) |
| `DF_STRESS_Z` | cover-2 stress quantile | 3.9 | empirical FHS ~99.87% (extreme-but-plausible) |
| `DF_STRESS_FLOOR` | cover-2 stress-move floor | 0.08 | calm floor; Euronext A9 §2.1 |
| `DF_STRESS_CAP` | cover-2 stress-move ceiling | 0.35 | 2-day extreme ES move (2008/COVID scale) |
| `DF_ODD_FIXED` | fixed-DF sizing toggle | False (H1: True) | ODD Mech #4 IM-independent fund for the margin arms |
| `DF_ODD_PERCENT` | ODD fixed-DF move | 0.10 | COVID-class 1-day ES move (Simudyne ODD) |
| `DF_DECOUPLE_IM` | size DF on fixed reference IM | False | H2 isolation lever |
| `df_interval` | DF recalculation cadence | 390 | one RTH day |
| `margin_interval` | VM cadence | 60 | hourly (ODD §Scales) |

### 4c. Member / client capital and solvency

| Parameter | Description | Value | Interpretation |
|---|---|---|---|
| `CCP_CASH` | CCP own capital | \$1.5B | SITG source + L5 backstop; major-CCP equity scale |
| `BCM_CASH_RANGE` | bank-FCM capital | U[\$5B, \$10B] | JPM/GS-class FCM net capital (CFTC) |
| `NBCM_CASH_RANGE` | non-bank FCM capital | U[\$0.5B, \$3B] | Marex/ABN/StoneX-class |
| `FT_CLIENT_CASH` | FT-client capital | U[\$0.5B, \$3B] | asset managers (largest books) |
| `MT_CLIENT_CASH` | MT-client capital | U[\$0.2B, \$1B] | CTA / managed futures |
| `ZI_CLIENT_CASH` | ZI-client capital | U[\$0.2B, \$0.5B] | smaller noise accounts |
| `LR_FLOOR_BCM` | BCM capital-adequacy floor | 0.08 | cash/exposure ≥ 8% (Basel III Pillar-1; deleverage trigger) |
| `REG117_FLOOR_NBCM` | NBCM floor | 0.08 | cash/exposure ≥ 8% (CFTC Reg 1.17; stop-out trigger) |
| `CLIENT_FREEZE_FLOOR` | client distress freeze | 0.04 | cash/exposure ≥ 4% (FCM freezes the client) |
| `DELEVERAGE_TARGET_BCM` | BCM deleverage target | 0.10 | breaching BCM sheds own book to 10% (2pp above floor) |
| `cap_ratio_floor` | legacy single CM floor | 0.08 | CFTC Reg 1.17 (fallback when floors not differentiated) |

### 4d. Close-out, liquidation and risk limits

| Parameter | Description | Value | Interpretation |
|---|---|---|---|
| `CLOSEOUT_RECOVERY` | member-default auction recovery | 0.80 | conservative stressed recovery (vs Lehman ~par, ODD 0.60) |
| `CLOSEOUT_MODE` | member book resolution | "transfer" | CCP auctions to largest-offsetting member |
| `CLIENT_CLOSEOUT` | client book resolution | "firesale" | carrying member liquidates open-market (Almgren-Chriss) |
| `CLIENT_MARGIN_NETTING` | margin/capital netting basis | "gross" | US/CME gross (vs EU net-omnibus — the netting experiment axis) |
| `POSITION_LIMIT_X` | client-clearing BCM house cap | 2.0 | own notional ≤ 2× cash (dealer prop book) |
| `POSITION_LIMIT_X_HOUSE` | house-only BCM cap | 5.0 | own notional ≤ 5× cash (looser; ~30–35% house IM share) |
| `POSITION_LIMIT_CLIENTS_ONLY` | scope of the 2× cap | True | 2× binds client-clearing BCMs only |
| `LAMBDA_RISK` | Almgren-Chriss risk aversion | 2.0 | the single free liquidation parameter (κH≈1 calm, ≈2.9 stressed) |
| `AC_HORIZON` | AC liquidation horizon | 30 min | slices a distressed book |
| `ETA_TEMP` | AC temporary-impact coeff | 1.32e-4 / 3.78e-4 | Measured (impact regression, calm / stressed) |
| `GAMMA_PERM` | AC permanent-impact coeff | 0.0 | empirically negative (reverts) → floored to transient-only |

### 4e. Experiment / sensitivity toggles (held at baseline, varied in specific runs)

`REANCHOR_ON_GAP=True` (clean overnight reprice), `DIFFERENTIATED_FLOORS=True`,
`UNIFIED_CLIENT_FLOOR=True`, `BASEL_LR_BCM=True` — wiring switches for the member-solvency design,
held at baseline. `IM_MODE`, `IM_FLAT_FRAC`, `DF_ODD_FIXED`, `DF_DECOUPLE_IM`,
`CLIENT_MARGIN_NETTING`, `CLOSEOUT_RECOVERY` are the levers flipped in the H1 (margin) / netting /
robustness experiments.

---

### Summary of what is calibrated vs fixed

- **Calibrated (to stylised facts):** the 7 behavioural agent parameters per regime; and, for the
  synthetic fundamental, the log-vol amplitude `s_x²`, slow share `w`, and the three agent levers
  (`mt_lambda`, `mt_gamma`, `ft_sigma_c`) jointly (jump size `delta_mult` recommended dropped).
- **Measured (from data, not stylised-facts-fitted):** `SIGMA_V`, the SV-MJD realized measures
  (`sigma_d`, jumps, OU rates), `ETA_TEMP`, `p_zi` (calm, L2), `V0`, `VOLUME_LOT`.
- **Fixed (regulation / ODD / disclosure):** the entire clearing tier — margin, default fund,
  waterfall, capital floors, close-out, position limits — plus the structural LOB constants.
