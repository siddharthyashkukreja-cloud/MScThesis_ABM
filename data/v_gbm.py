"""
data/v_gbm.py — Stein-Stein-style stochastic-volatility Merton jump-diffusion:
calibration and path generation for the synthetic V_t fundamental (the
scenario-generator alternative to the Kalman path).

V_t evolves as a GBM with a stochastic, mean-reverting volatility sigma_t
(Vasicek / Ornstein-Uhlenbeck on sigma — Stein & Stein 1991), Poisson Merton
(1976) jumps, and empirical overnight gaps at every session boundary. Overnight
gap risk is the primary CCP default driver: the gaps carry ~40% of ES
close-to-close variance in both regimes, and in the COVID window the crash
itself was overnight while intraday returns netted positive (Lou, Polk &
Skouras 2019).

    dsigma_t = alpha (theta - sigma_t) dt + sigma_vol dW    (Vasicek-OU, 1-min)
    dlog V   = mu + sigma_t * Z + J          intraday steps (Z ~ N(0,1);
               J = N(0, delta^2) w.p. lambda_step)
    dlog V   = G                             session boundaries (G drawn from the
               regime's EMPIRICAL overnight-return pool — nonparametric bootstrap,
               no invented parameters)

Calibration — every parameter is measured from the ES data via a realized-
measures decomposition. The jump intensity is the Mancini (2009) threshold
count on ES itself (the ODD §Calibration pins lambda = 3/day from FTSE):

  1. Jump/diffusion split — per-day realized variance RV_d = sum r^2 vs
     bipower variation BV_d = (pi/2) sum |r_i||r_{i-1}| (Barndorff-Nielsen &
     Shephard 2004): BV is jump-robust, so
         jump variance / day = mean(max(RV_d - BV_d, 0))
         sigma_d^2 (per min) = mean(BV_d) / bars_per_day.
     Measured jump share: ~6.8% calm / ~3.9% stressed of intraday variance —
     consistent with the ~5-7% S&P 500 literature benchmark (Huang & Tauchen
     2005; Andersen, Bollerslev & Diebold 2007).
  2. Jump intensity / size — Mancini (2009) threshold counts: a 1-min return
     is a jump if |r| > THRESH_C * sigma_loc, sigma_loc = sqrt(BV_d / n_d);
         lambda_hat = count / n_days        (≈1.9/day calm, ≈1.0/day stressed)
         delta^2    = jump variance per day / lambda_hat.
  3. Stochastic vol (alpha, theta, sigma_vol) — method of moments on a
     JUMP-ROBUST 30-min bipower sigma series, with errors-in-variables-robust
     moments (the 30-bar window sigma_hat carries ~13% sampling noise, which
     inflates the naive variance and attenuates the lag-1 autocorrelation):
     for an OU with iid measurement noise, gamma_k = Var(sigma) e^{-alpha k D},
     so   e^{-alpha D} = gamma_2 / gamma_1,   Var(sigma) = gamma_1^2 / gamma_2
     — both free of the noise term. theta = mean(sigma_hat).
  4. theta is re-anchored at generation so E[sigma_t^2] = sigma_d^2 (the SV
     layer redistributes the measured diffusion variance over time; it does
     not add to it).
  5. Drift — mu = the empirical mean INTRADAY 1-min log-return, used directly
     (no -1/2 sigma^2 Ito correction: mu is already a log-return mean). The
     overnight pool carries the empirical overnight drift (negative in the
     COVID regime) at the boundaries.

Excess kurtosis is recorded as a validation diagnostic only — the generated
path carries the measured jump/SV share of the tails; the remainder is the
agent layer's to produce (consistent with the thesis convention that the
stylised facts are earned by the agents, not injected).

Usage:
    python data/v_gbm.py calibrate                          # writes output/v_gbm_params.json
    python data/v_gbm.py generate <regime> <seed> [n_days]  # writes data/fv_gbm_{regime}.csv
    python data/v_gbm.py generate-all <seed> [n_days]       # both regimes

n_days defaults to 120; the path length is decoupled from the empirical
sample, so ensemble episodes of any horizon can be generated (vary <seed>).
The output is fv_gbm_{regime}.csv — it does NOT touch the Kalman fundamentals
fv_{regime}.csv.
"""

from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent
PROC_DIR = DATA_DIR / "processed"
OUT_DIR = DATA_DIR.parent / "output"

REGIMES = ("calm", "stressed")

BARS_PER_DAY = 390   # 6.5-hour RTH day at the 1-min cadence
RV_WINDOW_MIN = 30   # realized-vol window for the SV calibration
THRESH_C = 4.0       # Mancini (2009) jump threshold, in local (bipower) sigmas


# ── data loading ─────────────────────────────────────────────────────────────

def load_returns(regime: str) -> dict:
    """Returns dict with ts, mid, returns, is_open_bar. `mid` is the 1-min
    (best_bid+best_ask)/2 — the series the agent calibration matches. Open-bar
    returns cross a date boundary (the overnight gap)."""
    df = pd.read_csv(PROC_DIR / f"ES_front_{regime}_1m.csv",
                     index_col=0, parse_dates=True)
    mid = df["mid"].to_numpy(dtype=float)
    log_p = np.log(mid)
    returns = np.concatenate([[np.nan], np.diff(log_p)])
    dates = pd.DatetimeIndex(df.index).date
    is_open_bar = np.zeros(len(df), dtype=bool)
    is_open_bar[0] = True
    is_open_bar[1:] = (dates[1:] != dates[:-1])
    return {"ts": df.index, "mid": mid, "dates": dates,
            "returns": returns, "is_open_bar": is_open_bar}


# ── calibration ──────────────────────────────────────────────────────────────

def _daily_measures(data: dict):
    """Per-day intraday return arrays and (RV, BV) realized measures."""
    r = data["returns"]
    valid = (~data["is_open_bar"]) & np.isfinite(r)
    day_of = pd.factorize(data["dates"])[0]
    out = []
    for d in np.unique(day_of):
        x = r[(day_of == d) & valid]
        if len(x) < 30:
            continue
        rv = float(np.sum(x * x))
        bv = float((np.pi / 2.0) * np.sum(np.abs(x[1:]) * np.abs(x[:-1])))
        out.append((x, rv, bv))
    return out


def calibrate_sv(data: dict, window_min: int = RV_WINDOW_MIN) -> dict:
    """Vasicek-OU SV calibration on a JUMP-ROBUST bipower sigma series with
    EIV-robust moments. Non-overlapping `window_min`-bar windows give
    sigma_hat_w = sqrt((pi/2) sum |r_i||r_{i-1}| / window_min); then
        theta        = mean(sigma_hat)
        e^{-alpha D} = gamma_2 / gamma_1     (autocovariance ratio — immune to
        Var(sigma)   = gamma_1^2 / gamma_2    the iid window-sampling noise)
        sigma_vol    = sqrt(2 alpha Var(sigma)).
    Returns (alpha, theta, sigma_vol) in 1-min units."""
    r = data["returns"].copy()
    valid = (~data["is_open_bar"]) & np.isfinite(r)
    r = np.where(valid, r, 0.0)
    n = len(r)
    n_wins = n // window_min
    rc = r[: n_wins * window_min].reshape(n_wins, window_min)
    vc = valid[: n_wins * window_min].reshape(n_wins, window_min).all(axis=1)
    bpv = (np.pi / 2.0) * np.sum(np.abs(rc[:, 1:]) * np.abs(rc[:, :-1]), axis=1)
    sig = np.sqrt(bpv[vc] / window_min)
    sig = sig[sig > 0]
    theta = float(sig.mean())
    x = sig - sig.mean()
    g1 = float(np.mean(x[:-1] * x[1:]))
    g2 = float(np.mean(x[:-2] * x[2:]))
    if g1 > 0 and g2 > 0 and g2 < g1:
        rho = g2 / g1                      # = e^{-alpha * window}
        var_s = g1 * g1 / g2               # noise-free Var(sigma)
    else:                                  # degenerate ACF — fall back to naive
        rho = float(np.clip(np.corrcoef(sig[:-1], sig[1:])[0, 1], 1e-3, 0.999))
        var_s = float(np.var(sig, ddof=1))
    alpha = -np.log(np.clip(rho, 1e-3, 0.999)) / window_min
    sigma_vol = float(np.sqrt(2.0 * alpha * var_s))
    return {"alpha": float(alpha), "theta": theta, "sigma_vol": sigma_vol,
            "var_sigma": float(var_s), "acov_ratio": float(rho),
            "window_min": int(window_min), "n_windows": int(len(sig))}


def calibrate(verbose: bool = True) -> dict:
    """Per-regime SV-MJD calibration from realized measures — see the module
    docstring. All parameters measured from the ES data; lambda is the Mancini
    threshold count (the ODD pins lambda=3/day from FTSE)."""
    results = {}
    for regime in REGIMES:
        data = load_returns(regime)
        days = _daily_measures(data)
        n_days = len(days)
        rv = np.array([d[1] for d in days])
        bv = np.array([d[2] for d in days])
        bars = float(np.mean([len(d[0]) for d in days]))
        jv_day = float(np.mean(np.maximum(rv - bv, 0.0)))
        n_jumps = 0
        for x, _rv, _bv in days:
            sloc = np.sqrt(_bv / len(x))
            n_jumps += int(np.sum(np.abs(x) > THRESH_C * sloc))
        lam_day = n_jumps / n_days
        delta = float(np.sqrt(jv_day / max(n_jumps / n_days, 1e-9)))
        sigma_d = float(np.sqrt(np.mean(bv) / bars))
        # Diagnostics + drift on the pooled intraday returns.
        valid = (~data["is_open_bar"]) & np.isfinite(data["returns"])
        r = data["returns"][valid]
        z = (r - r.mean()) / r.std()
        gaps = data["returns"][data["is_open_bar"] & np.isfinite(data["returns"])]
        results[regime] = {
            "sigma_d": sigma_d,
            "jump_lambda_day": float(lam_day),
            "jump_delta": delta,
            "jump_var_share": float(np.sum(np.maximum(rv - bv, 0)) / np.sum(rv)),
            "mean_return": float(np.mean(r)),
            "v0": float(data["mid"][0]),
            "sigma_total": float(np.std(r, ddof=1)),
            "excess_kurtosis_diag": float((z ** 4).mean() - 3.0),
            "n_valid": int(valid.sum()), "n_days": int(n_days),
            "overnight_pool": [float(g) for g in gaps],   # bootstrap pool
            "sv": calibrate_sv(data),
        }
        p = results[regime]; sv = p["sv"]
        if verbose:
            print(f"[{regime}] {n_days}d  sigma_d={sigma_d:.3e}  "
                  f"lambda={lam_day:.2f}/day  delta={delta:.3e}  "
                  f"(jump share {p['jump_var_share']*100:.1f}% of RV; "
                  f"ex_kurt diag {p['excess_kurtosis_diag']:.0f})")
            print(f"          SV: alpha={sv['alpha']:.4f}/min  theta={sv['theta']:.3e}  "
                  f"sigma_vol={sv['sigma_vol']:.2e}  (acov ratio {sv['acov_ratio']:.3f}, "
                  f"n={sv['n_windows']} @ {sv['window_min']}m)")
            print(f"          overnight pool: n={len(p['overnight_pool'])}  "
                  f"mean={np.mean(p['overnight_pool']):+.4f}  "
                  f"std={np.std(p['overnight_pool']):.4f}")
    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / "v_gbm_params.json"
    out_path.write_text(json.dumps(results, indent=2))
    if verbose:
        print(f"Saved {out_path}")
    return results


# ── generator ────────────────────────────────────────────────────────────────

REGIME_START = {"calm": "2019-01-02", "stressed": "2020-02-24"}
# Default path length per regime: calm is stationary (any horizon); stressed is
# the COVID episode, whose gap pool and drift describe a ~29-session crash, so
# episodes default to the empirical window length. Compounding crash-severity
# overnight gaps for 120 days gives a -79% path, far beyond any calibrated
# meaning.
DEFAULT_N_DAYS_REGIME = {"calm": 120, "stressed": 29}
DEFAULT_N_DAYS = 120   # for explicit calls


def generate(regime: str, seed: int, n_days: int | None = None,
             out_path: Path | None = None, mu_override: float | None = None,
             sv_override: dict | None = None, jump_override: dict | None = None) -> np.ndarray:
    """Forward-simulate the SV-MJD V_t path: OU sigma_t around theta_eff
    (anchored so E[sigma_t^2] = sigma_d^2), intraday returns
    mu + sigma_t Z + J, and a bootstrapped empirical overnight gap at every
    session boundary. Writes ts / V_smooth / sigma_t; the ts day boundaries
    drive the simulator's session reprice and the calibration's overnight-return
    exclusion, exactly as for the Kalman path.

    mu_override (per 1-min step) replaces the calibrated mean return when set --- used
    for a stress-scenario drift (e.g. the empirical period return taken down to the
    minute) so an ensemble reproduces a target overall decline rather than the mild
    calibrated net drift. The vol/jump/gap engine (the stylised-fact calibration) is
    unchanged."""
    if n_days is None:
        n_days = DEFAULT_N_DAYS_REGIME[regime]
    params_path = OUT_DIR / "v_gbm_params.json"
    if not params_path.exists():
        raise FileNotFoundError(f"Run calibration first: {params_path}")
    p = json.loads(params_path.read_text())[regime]
    sigma_d, v0 = p["sigma_d"], p["v0"]
    mu = float(p["mean_return"])               # mean intraday log-return (calibrated)
    if mu_override is not None:                 # stress-scenario per-step drift override
        mu = float(mu_override)
    delta = float(p["jump_delta"])
    lam_day = float(p["jump_lambda_day"])
    if jump_override is not None:                 # reverse-calibration: jump intensity / size override
        delta = float(jump_override.get("jump_delta", delta))
        lam_day = float(jump_override.get("jump_lambda_day", lam_day))
    lam_step = lam_day / BARS_PER_DAY
    pool = np.asarray(p["overnight_pool"], dtype=float)
    sv = p["sv"]
    if sv_override is not None:                  # reverse-calibration: inject OU vol params
        sv = {**sv, **sv_override}               # (alpha, sigma_vol, ...)
        if "sigma_d" in sv:                      # optional vol-LEVEL override (joint calibration)
            sigma_d = float(sv["sigma_d"])
    n_steps = n_days * BARS_PER_DAY
    rng = np.random.default_rng(seed)

    if sv_override is not None and sv_override.get("two_factor"):
        # Two-factor superposition-OU volatility (Barndorff-Nielsen & Shephard 2001): two
        # zero-mean OU factors u_fast + u_slow with rates a_f/a_s and stationary variances
        # v_f/v_s, so the vol autocorrelation is a SUM of exponentials (two timescales).
        # log_vol=True puts the OU on LOG-volatility (exp-OU / Scott): sigma_t=exp(m+u_f+u_s),
        # strictly positive (NO clamp), so a large vol-of-vol fattens the return tails THROUGH
        # the volatility clustering rather than needing i.i.d. jumps. Both anchor E[sigma^2]=sigma_d^2.
        a_f, a_s = float(sv["alpha_fast"]), float(sv["alpha_slow"])
        v_f, v_s = float(sv["v_fast"]), float(sv["v_slow"])
        ef, es = float(np.exp(-a_f)), float(np.exp(-a_s))
        sd_f = float(np.sqrt(v_f * (1.0 - ef * ef)))   # innovation sd -> stationary var v_f
        sd_s = float(np.sqrt(v_s * (1.0 - es * es)))
        z_f, z_s = rng.standard_normal(n_steps), rng.standard_normal(n_steps)
        u_f, u_s = np.empty(n_steps), np.empty(n_steps)
        u_f[0] = u_s[0] = 0.0
        for t in range(n_steps - 1):
            u_f[t + 1] = ef * u_f[t] + sd_f * z_f[t]
            u_s[t + 1] = es * u_s[t] + sd_s * z_s[t]
        if sv.get("log_vol"):
            m = float(np.log(sigma_d)) - (v_f + v_s)        # lognormal anchor: E[sigma^2]=sigma_d^2
            sigma_path = np.exp(m + u_f + u_s)
            theta_eff = float(np.exp(m))
        else:
            theta_eff = float(np.sqrt(max(sigma_d ** 2 - (v_f + v_s), 1e-12)))
            sigma_path = np.maximum(theta_eff + u_f + u_s, 1e-10)
    else:
        alpha, sigma_vol = sv["alpha"], sv["sigma_vol"]
        sv_var = sigma_vol * sigma_vol / (2.0 * alpha)
        theta_eff = (float((sigma_d ** 2 - sv_var) ** 0.5)
                     if sigma_d ** 2 > sv_var else float(sv["theta"]))
        e_alpha = float(np.exp(-alpha))
        ou_sd = float(sigma_vol * np.sqrt((1.0 - np.exp(-2.0 * alpha)) / (2.0 * alpha)))
        eps_s = rng.standard_normal(n_steps)
        sigma_path = np.empty(n_steps)
        sigma_path[0] = theta_eff
        for t in range(n_steps - 1):
            sigma_path[t + 1] = (theta_eff + e_alpha * (sigma_path[t] - theta_eff)
                                 + ou_sd * eps_s[t])
        sigma_path = np.maximum(sigma_path, 1e-10)

    eps_r = rng.standard_normal(n_steps)
    if jump_override is not None and jump_override.get("hawkes"):
        # Self-exciting (Hawkes) jumps: the intensity jumps up after each jump and decays, so
        # jumps CLUSTER in time (Ait-Sahalia-Cacho-Diaz-Laeven 2015) - a fat tail that
        # REINFORCES the |r| autocorrelation rather than diluting it (the Poisson failure mode).
        # Long-run rate kept at lam_step; branching ratio n_br sets the endogenous clustering.
        n_br = float(jump_override.get("branching", 0.5))
        g = float(np.exp(-float(jump_override.get("beta", 0.02))))    # per-step intensity decay
        mu_h = lam_step * (1.0 - n_br)                                # baseline keeps the avg rate
        jo = n_br * (1.0 - g) / g                                     # excitation added per jump
        uj, zj = rng.random(n_steps), rng.normal(0.0, delta, n_steps)
        jumps = np.zeros(n_steps); e = 0.0
        for t in range(n_steps):
            if uj[t] < mu_h + e:
                jumps[t] = zj[t]; e += jo
            e *= g
        jump_mask = jumps != 0.0
    else:
        jump_mask = rng.random(n_steps) < lam_step
        jumps = jump_mask * rng.normal(0.0, delta, n_steps)
    log_increments = mu + sigma_path * eps_r + jumps
    log_increments[0] = 0.0
    # Overnight gaps: a bootstrap draw from the regime's empirical
    # overnight-return pool at each session boundary (first bar of days 1..n).
    gap_draws = rng.choice(pool, size=n_days - 1, replace=True)
    log_increments[BARS_PER_DAY::BARS_PER_DAY] = gap_draws
    V = np.exp(np.log(v0) + np.cumsum(log_increments))

    days = pd.bdate_range(start=REGIME_START[regime], periods=n_days)
    ts = [pd.Timestamp(d) + pd.Timedelta(hours=13, minutes=30)
          + pd.Timedelta(minutes=k) for d in days for k in range(BARS_PER_DAY)]
    ts = pd.DatetimeIndex(ts[:n_steps])

    if out_path is None:
        out_path = DATA_DIR / f"fv_gbm_{regime}.csv"   # never the Kalman fv_{regime}.csv
    out_path = Path(out_path)
    pd.DataFrame({"ts": ts, "V_smooth": V,
                  "sigma_t": sigma_path}).to_csv(out_path, index=False)
    dd = float(V.min() / V[0] - 1.0)
    print(f"[{regime}] seed={seed} n_days={n_days} n={n_steps}: "
          f"{int(jump_mask.sum())} jumps, {n_days - 1} gaps "
          f"(gap mean {gap_draws.mean():+.4f}), V0={V[0]:.2f} Vend={V[-1]:.2f} "
          f"maxDD={dd:+.1%}; sigma_t mean={sigma_path.mean():.3e} "
          f"(theta_eff={theta_eff:.3e}) -> {out_path}")
    return V


# ── CLI ──────────────────────────────────────────────────────────────────────

def _main():
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    cmd = sys.argv[1]
    if cmd == "calibrate":
        calibrate()
    elif cmd == "generate":
        if len(sys.argv) < 4:
            print("Usage: generate <regime> <seed> [n_days]"); sys.exit(1)
        generate(sys.argv[2], int(sys.argv[3]),
                 int(sys.argv[4]) if len(sys.argv) >= 5 else None)
    elif cmd == "generate-all":
        if len(sys.argv) < 3:
            print("Usage: generate-all <seed> [n_days]"); sys.exit(1)
        seed = int(sys.argv[2])
        n_days = int(sys.argv[3]) if len(sys.argv) >= 4 else None
        for r in REGIMES:
            generate(r, seed, n_days)
    else:
        print(f"Unknown command: {cmd}\n"); print(__doc__); sys.exit(1)


if __name__ == "__main__":
    _main()
