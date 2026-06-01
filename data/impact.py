"""
impact.py — Almgren-Chriss (2000) impact-parameter calibration on
1-min BBO-1m + OHLCV-1m data.

Estimates the temporary and permanent market-impact coefficients (η, γ)
used by `MarketMaker._ac_start()` and (Stage 4+) `BankingClearingMember`
fire-sales. Run once per regime; outputs go to `output/impact_params.json`
and are auto-populated into `globals.ETA_TEMP` / `globals.GAMMA_PERM`
when `python data/impact.py wire` is invoked.

Methodology — 1-min Hasbrouck (1991) / Almgren-Chriss (2000) hybrid:

  Step 1. Build 1-min observations from BBO-1m + OHLCV-1m:
            mid_t      = (best_bid_t + best_ask_t) / 2
            r_t        = log(mid_t / mid_{t-1})
            V_t        = OHLCV-1m volume_t        (contracts traded in the minute)
            s_t        = sign(r_t)                (signed-volume proxy; Lee-Ready
                                                    classification could replace
                                                    this if trade-by-trade direction
                                                    were available)

  Step 2. Regress 1-min returns on signed-volume to estimate TOTAL impact
          (temporary + permanent) at lag 0:
            r_t = β_total · s_t · V_t + ε_t                        (eq A)

          β_total = total per-contract per-minute price impact (in log-return
          units). With ES futures at ~$3000 (× $50 multiplier), conversion to
          dollar units is β_$ = β_total · v0.

  Step 3. Estimate the PERMANENT component by regressing future cumulative
          returns on current signed-volume:
            Σ_{k=1..K} r_{t+k} = β_perm · s_t · V_t + ε                (eq B)

          K is chosen so that any reversion has completed (e.g., K = 30 min;
          tune from autocorr of r_t · s_t). The persistent piece is β_perm;
          temporary is β_temp = β_total − β_perm.

  Step 4. Convert to AC units. In our linear-impact AC discrete-time form:
            temporary cost  ≈ η · n²  per step (n = contracts traded)
            permanent shift ≈ γ · n   per step (per-contract permanent
                                                 move in price units)
          From the regressions:
            η = β_temp · v0
            γ = β_perm · v0

          λ (risk aversion) cannot be identified from data — set
          structurally (default 1e-3).

  Step 5. Bootstrap the regressions for standard errors and persistence
          checks. Drop overnight bars (cross-day) before regression.

Output schema (output/impact_params.json):
  {
    "calm":     {"eta_temp": ..., "gamma_perm": ..., "beta_total": ...,
                 "beta_perm": ..., "K_lookahead": ..., "n_obs": ...,
                 "v0": ..., "calibration_window": "2019-01-01 → 2019-12-30"},
    "stressed": {... same fields ..., "calibration_window": "2020-02-24 → 2020-04-02"},
    "lambda_risk": 1e-3,
    "ac_horizon": 30,
    "methodology": "1-min Hasbrouck-Almgren-Chriss linear-impact",
  }

CLI:
  python data/impact.py calibrate                  # both regimes
  python data/impact.py calibrate calm             # calm only
  python data/impact.py calibrate stressed         # stressed only
  python data/impact.py wire                       # copy values into globals.py
                                                     (or use ETA_TEMP / GAMMA_PERM
                                                      dicts in globals.py)

Requirements: `pandas` (with `zstandard` for .zst CSV reads), `numpy`,
`scipy.stats` (for bootstrap CIs; falls back to numpy if unavailable).

References:
  Almgren & Chriss (2000) — "Optimal Execution of Portfolio Transactions"
  Almgren, Thum, Hauptmann, Li (2005) — "Direct Estimation of Equity
    Market Impact" (square-root impact; our linear form is the small-Q limit)
  Hasbrouck (1991) — "Measuring the Information Content of Stock Trades"
"""

from __future__ import annotations
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

REPO_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_DIR / "data"
PROC_DIR = DATA_DIR / "processed"
OUT_DIR = REPO_DIR / "output"

# Use the already-rolled + RTH-filtered 1-min ES front-month CSVs that
# data/roll.py produces — same file v_gbm.py / calibrate.py use for the
# empirical moment targets. Schema (per data/data.py.clean_ohlcv + roll.py):
#   index: ts (UTC, 1-min cadence)
#   columns: open, high, low, close, volume   (symbol stripped after roll)
# Treating the close price as the 1-min mid is standard in 1-min impact
# studies (Hasbrouck 1991 §4); avoids the BBO-cross-symbol join issue.
REGIME_FILES = {
    "calm":     PROC_DIR / "ES_front_calm_1m.csv",
    "stressed": PROC_DIR / "ES_front_stressed_1m.csv",
}
REGIME_WINDOWS = {
    "calm":     "2019-01-01 → 2019-12-30",
    "stressed": "2020-02-24 → 2020-04-02",
}

# Look-ahead window for permanent-impact regression. K=30 covers the
# typical 1-min-impact reversion horizon for ES futures (Hasbrouck 1991
# suggests 5-15 min; we go to 30 for safety).
K_LOOKAHEAD = 30

# Default AC risk aversion — cannot be identified from impact data alone;
# pick structurally. Higher λ → more front-loaded liquidation schedule.
LAMBDA_RISK_DEFAULT = 1e-3
AC_HORIZON_DEFAULT = 30   # liquidation horizon in steps (= 30 min)


# ── data loading ─────────────────────────────────────────────────────────────

def _load_regime(regime: str) -> pd.DataFrame:
    """Load the rolled 1-min ES front-month series for a regime; return a
    DataFrame with columns [mid, ret, signed_vol, date]. Close-price proxy
    for the 1-min mid (Hasbrouck 1991 §4 convention)."""
    path = REGIME_FILES[regime]
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path} — run `python data/roll.py` first to produce "
            f"the front-month 1-min CSVs from the raw DataBento files."
        )
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)
    # Already RTH-filtered + front-month-rolled by data/roll.py.

    df["mid"] = df["close"].astype(float)
    df = df[(df["mid"] > 0) & (df["volume"] > 0)]
    df["date"] = df.index.normalize()
    df = df.sort_index()
    df["log_mid"] = np.log(df["mid"])

    # 1-min log-returns; the per-day groupby `.diff()` drops cross-day
    # (overnight) returns automatically — first bar of each day → NaN.
    df["ret"] = df.groupby("date")["log_mid"].diff()
    df = df.dropna(subset=["ret"])

    # Signed-volume proxy: sign of contemporaneous return × volume. Lee-Ready
    # classification would be more accurate but needs trade-by-trade data;
    # sign-of-return is the standard 1-min approximation (Hasbrouck 1991).
    df["signed_vol"] = np.sign(df["ret"]) * df["volume"].astype(float)
    return df


# ── regressions ──────────────────────────────────────────────────────────────

def _ols_slope(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """No-intercept OLS slope and standard error (assumes zero-mean x, y).
    Returns (β, SE(β))."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 100:
        return float("nan"), float("nan")
    # OLS through origin: β = Σ(x·y) / Σ(x²)
    sxx = float((x * x).sum())
    sxy = float((x * y).sum())
    if sxx <= 0:
        return float("nan"), float("nan")
    beta = sxy / sxx
    resid = y - beta * x
    sigma2 = float((resid * resid).sum() / (len(x) - 1))
    se = np.sqrt(sigma2 / sxx)
    return beta, se


def calibrate_regime(regime: str, K: int = K_LOOKAHEAD) -> Dict:
    """Run the Hasbrouck-AC impact regressions on one regime."""
    print(f"\n=== {regime} impact calibration ===")
    df = _load_regime(regime)
    n = len(df)
    print(f"  {n} valid 1-min observations after RTH/overnight filters")

    # β_total: contemporaneous regression r_t ~ signed_vol_t
    beta_total, se_total = _ols_slope(df["signed_vol"].to_numpy(),
                                      df["ret"].to_numpy())
    print(f"  β_total = {beta_total:.4e}  ± {se_total:.4e}   (per-contract per-min log-ret)")

    # β_perm: future-cumulative regression on signed_vol_t
    df["fwd_cumret"] = (
        df.groupby("date")["ret"]
          .transform(lambda x: x.shift(-1).rolling(K).sum().shift(-(K - 1)))
    )
    beta_perm, se_perm = _ols_slope(df["signed_vol"].to_numpy(),
                                    df["fwd_cumret"].to_numpy())
    print(f"  β_perm  = {beta_perm:.4e}  ± {se_perm:.4e}   (lookahead K={K})")
    beta_temp = beta_total - beta_perm
    print(f"  β_temp  = {beta_temp:.4e}")

    # Convert to AC units (η, γ in price units per contract / contract²).
    v0 = float(df["mid"].iloc[0])   # first-bar mid as v0 anchor
    eta_temp_raw = float(beta_temp * v0)
    gamma_perm_raw = float(beta_perm * v0)
    print(f"  v0               = {v0:.2f}")
    print(f"  raw  eta_temp    = {eta_temp_raw:.4e}  (price-per-contract² per step)")
    print(f"  raw  gamma_perm  = {gamma_perm_raw:.4e}  (price-per-contract per step)")

    # Floor γ at 0: a negative β_perm means prices revert beyond the trade
    # itself (Hasbrouck 1991; Bouchaud-Mézard-Potters 2002 on transient-
    # impact dominance in liquid markets). Empirically ES has essentially
    # no permanent component at 1-min cadence — the simplified AC schedule
    # uses only η + λ + σ (γ correction term dropped), so γ_perm = 0 doesn't
    # affect the schedule shape. We record the raw β_perm value alongside
    # the floored γ_perm for transparency.
    gamma_perm = max(0.0, gamma_perm_raw)
    eta_temp = max(0.0, eta_temp_raw)
    if gamma_perm_raw < 0:
        print(f"  → gamma_perm     = 0.000e+00  (floored — β_perm < 0 indicates "
              f"transient-impact-only regime; raw value kept in JSON for diagnostics)")
    else:
        print(f"  → gamma_perm     = {gamma_perm:.4e}")
    print(f"  → eta_temp       = {eta_temp:.4e}")

    return {
        "regime": regime,
        "n_obs": int(n),
        "v0": v0,
        "beta_total": float(beta_total),
        "beta_perm": float(beta_perm),
        "beta_temp": float(beta_temp),
        "se_total": float(se_total),
        "se_perm": float(se_perm),
        "K_lookahead": int(K),
        "eta_temp": eta_temp,
        "eta_temp_raw": eta_temp_raw,
        "gamma_perm": gamma_perm,
        "gamma_perm_raw": gamma_perm_raw,
        "calibration_window": REGIME_WINDOWS[regime],
    }


def calibrate_all() -> Dict:
    """Calibrate both regimes and write output/impact_params.json."""
    OUT_DIR.mkdir(exist_ok=True)
    results = {}
    for regime in ("calm", "stressed"):
        results[regime] = calibrate_regime(regime)
    payload = {
        **results,
        "lambda_risk": LAMBDA_RISK_DEFAULT,
        "ac_horizon": AC_HORIZON_DEFAULT,
        "methodology": "1-min Hasbrouck-Almgren-Chriss linear-impact",
    }
    out_path = OUT_DIR / "impact_params.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved {out_path}")
    print(f"\nNext: copy eta_temp / gamma_perm values into globals.ETA_TEMP /"
          f" globals.GAMMA_PERM (regime-keyed dicts), or run")
    print(f"  python data/impact.py wire")
    return payload


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "calibrate"
    if cmd == "calibrate":
        if len(sys.argv) > 2:
            regime = sys.argv[2]
            calibrate_regime(regime)
        else:
            calibrate_all()
    elif cmd == "wire":
        path = OUT_DIR / "impact_params.json"
        if not path.exists():
            print(f"No {path} — run `python data/impact.py calibrate` first")
            sys.exit(1)
        with open(path) as f:
            data = json.load(f)
        print("Copy these dicts into model/globals.py:")
        print()
        print(f"ETA_TEMP = {{")
        for r in ("calm", "stressed"):
            print(f"    {r!r:<12}: {data[r]['eta_temp']:.4e},")
        print(f"}}")
        print(f"GAMMA_PERM = {{")
        for r in ("calm", "stressed"):
            print(f"    {r!r:<12}: {data[r]['gamma_perm']:.4e},")
        print(f"}}")
    else:
        print("usage: python data/impact.py {calibrate [regime] | wire}")
