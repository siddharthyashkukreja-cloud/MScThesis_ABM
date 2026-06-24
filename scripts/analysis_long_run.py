import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
Long-horizon simulator run + multi-frequency moment comparison, at the baseline
θ (`globals.CALIBRATED`) on the calibration population (bare market, 40 FT-equiv
/ 20 MT / 40 ZI, no clearing tier — the configuration the θ was fit on). Reports
return moments and ACFs at 1-min and daily aggregation, alongside the empirical
ES targets.

Daily mid is the last 1-min mid of each simulated trading day (end-of-RTH
close). Empirical daily close is taken the same way from the rolled
front-month ES tape (`data/processed/ES_front_{regime}_1m.csv`).

CLI:
    python analysis_long_run.py [regime] [n_days]
        regime  ∈ {calm, stressed}   default: calm
        n_days                        default: 100  (V_t paths cover 120)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from model.globals import ModelParams, V0, CALIBRATED, day_start_steps
from model.simulation import Simulation
from model.run_simulation import build_traders, BARS_PER_DAY

ROOT = Path(__file__).resolve().parent.parent   # scripts/ -> repo root
EMP_PATH = {
    "calm":     ROOT / "data" / "processed" / "ES_front_calm_1m.csv",
    "stressed": ROOT / "data" / "processed" / "ES_front_stressed_1m.csv",
}


def _acf(x: np.ndarray, k: int) -> float:
    if len(x) <= k:
        return float("nan")
    x = x - x.mean()
    var = float((x * x).sum())
    return 0.0 if var == 0.0 else float((x[:-k] * x[k:]).sum() / var)


def _acf_smooth(x: np.ndarray, c: int) -> float:
    """Forward 3-lag avg — matches calibrate._acf_smoothed_fwd so the long-run
    validation uses the same estimator as the calibration loss (XGB-Chiarella
    §3.2.2: lag-c = mean of {c, c+1, c+2})."""
    lags = (c, c + 1, c + 2)
    vals = [v for v in (_acf(x, l) for l in lags) if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def _hill(r: np.ndarray, frac: float = 0.05) -> float:
    a = np.abs(np.asarray(r, float))
    a = a[np.isfinite(a) & (a > 0)]
    n = len(a)
    if n < 100:
        return float("nan")
    k = max(int(frac * n), 20)
    if k >= n:
        return float("nan")
    a_sorted = np.sort(a)[::-1]
    thr = a_sorted[k]
    if thr <= 0:
        return float("nan")
    xi = float(np.mean(np.log(a_sorted[:k]) - np.log(thr)))
    return 1.0 / xi if xi > 0 else float("nan")


def _moments(r: np.ndarray, lags_r=(1, 5, 10), lags_a=(1, 5, 10, 20, 30, 60)) -> dict:
    r = r[np.isfinite(r)]
    sd = float(r.std()) if len(r) else float("nan")
    kurt = (float((((r - r.mean()) / sd) ** 4).mean() - 3.0)
            if sd > 0 else float("nan"))
    a = np.abs(r)
    out = {"n": len(r), "ret_std": sd, "kurt": kurt, "hill": _hill(r)}
    for L in lags_r:
        out[f"acf_r_{L}"] = _acf_smooth(r, L)
    for L in lags_a:
        out[f"acf_a_{L}"] = _acf_smooth(a, L)
    return out


def _print_row(label: str, m: dict, lags_r=(1, 5, 10), lags_a=(1, 5, 10, 20, 30, 60)):
    print(f"\n  {label}  (n={m['n']:,d}):")
    print(f"    ret_std = {m['ret_std']:.3e}    kurt = {m['kurt']:>7.1f}    "
          f"Hill = {m['hill']:.3f}")
    print(f"    acf(r):    " + "  ".join(
        f"lag{L}={m[f'acf_r_{L}']:+.4f}" for L in lags_r))
    print(f"    acf(|r|):  " + "  ".join(
        f"lag{L}={m[f'acf_a_{L}']:+.4f}" for L in lags_a))


def _theta(regime: str) -> dict:
    """The baseline θ — `globals.CALIBRATED`; the surrogate JSON is a cross-check,
    not the source of truth."""
    print(f"  using globals.CALIBRATED[{regime}] (the locked grid headline)")
    return dict(CALIBRATED[regime])


def empirical_returns(regime: str):
    df = pd.read_csv(EMP_PATH[regime], index_col=0, parse_dates=True)
    mid = df["mid"].to_numpy(dtype=float)
    log_p = np.log(mid)
    # 1-min log-returns, drop cross-day boundary
    r_1m = np.diff(log_p)
    dates = pd.DatetimeIndex(df.index).date
    keep = dates[1:] == dates[:-1]
    r_1m = r_1m[keep]
    # Daily close = last mid of each calendar day
    df_idx = pd.DatetimeIndex(df.index)
    last_per_day = df["mid"].groupby(df_idx.date).last().to_numpy()
    r_d = np.diff(np.log(last_per_day))
    return r_1m, r_d


def main():
    regime = sys.argv[1] if len(sys.argv) > 1 else "calm"
    n_days = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    if regime not in V0:
        print(f"unknown regime '{regime}'; valid: calm, stressed")
        sys.exit(1)
    print(f"long-run simulation: regime={regime}, n_days={n_days}")
    theta = _theta(regime)

    # Calibration population (calibrate.POP): bare market, no clearing tier —
    # the configuration the θ was actually fit on.
    params = ModelParams(
        n_fundamental=40, n_momentum=20, n_zi=40,
        n_bcm=0, n_nbcm=0, n_bcm_with_clients=0,
        v0=V0[regime], tick_size=0.25, dt_minutes=1.0,
        stressed=(regime == "stressed"),
        **theta,
    )
    traders = build_traders(params, seed=42)
    hist = Simulation(params, traders, seed=42, ccp=None).run(n_days * BARS_PER_DAY)
    mid = pd.Series(hist["mid_price"]).ffill().bfill().to_numpy()
    mid = mid[mid > 0]

    # Real RTH-session opens within the run (data-driven, ~405 bars/session and
    # variable, not a fixed 390).
    opens = [d for d in day_start_steps(regime) if 0 < d < len(mid)]
    r_1m_mdl = np.diff(np.log(mid))
    # Drop cross-day (overnight) returns — the sim opens each RTH day at the
    # gapped V_t, so the day-boundary return is an overnight gap; match the
    # empirical convention (intraday returns only) for the 1-min comparison.
    r_1m_mdl = np.delete(r_1m_mdl, [d - 1 for d in opens])
    # daily close = last mid of each session (the bar before the next open, plus
    # the final bar); the overnight gap is in the daily return — that is the
    # correct cross-day return.
    closes = sorted(set([d - 1 for d in opens] + [len(mid) - 1]))
    daily_mid = mid[closes]
    r_d_mdl = np.diff(np.log(daily_mid))

    r_1m_emp, r_d_emp = empirical_returns(regime)

    print(f"\n{'='*68}")
    print(f"  MODEL  vs  EMPIRICAL  —  {regime}")
    print(f"{'='*68}")

    LA_1m = (1, 5, 10, 20, 30, 60)
    LA_d  = (1, 2, 5, 10, 20)
    _print_row(f"MODEL 1-min",    _moments(r_1m_mdl, lags_a=LA_1m), lags_a=LA_1m)
    _print_row(f"EMP   1-min",    _moments(r_1m_emp, lags_a=LA_1m), lags_a=LA_1m)
    _print_row(f"MODEL daily",    _moments(r_d_mdl,  lags_a=LA_d),  lags_a=LA_d)
    _print_row(f"EMP   daily",    _moments(r_d_emp,  lags_a=LA_d),  lags_a=LA_d)

    out_csv = ROOT / "output" / f"long_run_{regime}.csv"
    pd.DataFrame({"mid_1m": mid}).to_csv(out_csv, index=False)
    print(f"\n  saved {out_csv}")


if __name__ == "__main__":
    main()
