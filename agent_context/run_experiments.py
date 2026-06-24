"""
Overnight contagion-experiment driver — H1, H2, E6, SV-MJD robustness, open-disorder.

Builds seed ENSEMBLES over a reverse-stress amplifier grid (Euronext A9 §5) on the
standard cleared population (no engineered client) and writes one row per run, then
aggregates per hypothesis. All experiments reuse the tested building blocks in
`covid_contagion.py` / `run_simulation.py`; nothing here changes model behaviour
(the open-disorder arm only toggles the existing `globals.REANCHOR_ON_GAP` flag).

Design notes
------------
* One deduplicated MASTER spec set is run once; each hypothesis is recovered in
  `aggregate()` by SLICING the rows on their config columns (so reactive+tiered is
  computed once and shared by H1 and H2). Rows are checkpointed to `rows.csv` and
  the run is RESUMABLE (already-present (config|c|seed) keys are skipped).
* H2 margin regimes flip the module global `globals.IM_MODE` / `IM_FLAT_FRAC`; the
  worker sets these at the start of every task, so multiprocessing worker reuse is
  safe (each task re-sets its own globals before building the sim).
* The SV-MJD overlay points the sim at `data/fv_gbm_stressed.csv` with the GBM
  re-lock theta (output/relock_gbm/grid_stressed.json); the Kalman arm uses the
  real COVID window + globals.CALIBRATED["stressed"].

Usage
-----
  python3 run_experiments.py smoke      # tiny pre-flight (2 seeds, 2 c) — run this FIRST
  python3 run_experiments.py all        # the full overnight matrix
  python3 run_experiments.py aggregate  # (re)build summaries from rows.csv only
Env overrides: SEEDS (e.g. 42:82), NPROC, OUTDIR, CGRID ("1,1.5,2,..."), QUICK=1.
"""
from __future__ import annotations
import os, sys, json, time, traceback, itertools, csv
from pathlib import Path
import numpy as np
import pandas as pd
import multiprocessing as mp

from model import globals as G
from model.globals import ModelParams, CALIBRATED, CCP_CASH, day_start_steps
from model.simulation import Simulation
from model.run_simulation import build_traders, build_clearing_tier
import covid_contagion as cc

# ── Config (env-overridable) ──────────────────────────────────────────────────
def _seeds_from_env() -> list:
    s = os.environ.get("SEEDS", "42:82")          # 40 seeds, 42..81
    a, b = s.split(":"); return list(range(int(a), int(b)))

def _cgrid_from_env() -> list:
    s = os.environ.get("CGRID", "")
    if s:
        return [float(x) for x in s.split(",")]
    return [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5, 4.0]

SEEDS    = _seeds_from_env()
C_GRID   = _cgrid_from_env()
C_GRID_E6 = [1.0, 1.5, 2.0, 2.5, 3.0]             # E6 subset (all are in C_GRID)
CALM_SEEDS = list(range(42, 52))                  # 10 seeds for the H2 calm-cost
NPROC    = int(os.environ.get("NPROC", str(max(1, (os.cpu_count() or 2) - 1))))
OUTDIR   = Path(os.environ.get("OUTDIR", "output/experiments"))
GBM_WINDOW_SESSIONS = 10                          # mirror the ~10-session Kalman COVID window
IM_REGIMES = {                                    # (im_mode, im_flat_frac)
    "reactive": ("reactive", 0.05),
    "static":   ("static",   0.05),
    "flat12":   ("flat",     0.12),
    "flat05":   ("flat",     0.05),
}

def _gbm_theta() -> dict:
    p = Path("output/relock_gbm/grid_stressed.json")
    if not p.exists():
        raise FileNotFoundError("GBM re-lock theta missing — run the SV-MJD re-lock first "
                                "or drop the gbm fund (output/relock_gbm/grid_stressed.json)")
    t = json.loads(p.read_text())["results"]["stressed"]["theta_grid"]
    keys = ("ft_sigma_c", "zi_alpha", "zi_delta", "p_zi", "zi_mu")
    return {k: float(t[k]) for k in keys if k in t}

# Fundamental specs: (fv csv, behavioural theta, window kind)
def _fundamentals() -> dict:
    f = {"kalman": dict(fv="data/fv_stressed.csv", theta=dict(CALIBRATED["stressed"]),
                        window="covid")}
    if Path("data/fv_gbm_stressed.csv").exists() and Path("output/relock_gbm/grid_stressed.json").exists():
        f["gbm"] = dict(fv="data/fv_gbm_stressed.csv", theta=_gbm_theta(), window="gbm")
    return f

# ── One run ───────────────────────────────────────────────────────────────────
def _fv_window(fund: str, fspec: dict):
    """Return (V, SIG, w0, wn) for the fundamental's stress window."""
    df = pd.read_csv(fspec["fv"])
    V = df["V_smooth"].to_numpy(); SIG = df["sigma_t"].to_numpy()
    if fspec["window"] == "covid":
        return V, SIG, cc.WINDOW_START, cc.WINDOW_STEPS
    # gbm: first GBM_WINDOW_SESSIONS sessions from the synthetic stressed episode
    d = pd.to_datetime(df["ts"]).dt.normalize().to_numpy()
    opens = list(np.flatnonzero(np.concatenate(([True], d[1:] != d[:-1]))))
    w0 = 0
    wn = (int(opens[GBM_WINDOW_SESSIONS]) if len(opens) > GBM_WINDOW_SESSIONS else len(V)) - w0
    return V, SIG, w0, wn


def _metrics(sim, ccp) -> dict:
    ch = pd.DataFrame(sim.clearing_history)
    cl = pd.DataFrame(sim.client_history)
    mid = np.array([m for m in sim.history["mid_price"] if m == m])
    deepest = int(ch["waterfall_level"].max()) if len(ch) else 0
    nbcm = int(ch[ch["has_defaulted"] & (ch["kind"] == "NBCM")]["agent_id"].nunique()) if len(ch) else 0
    bcm = int(ch[ch["has_defaulted"] & (ch["kind"] == "BCM")]["agent_id"].nunique()) if len(ch) else 0
    # IM time series — aggregate posted initial margin across members per cycle
    if len(ch):
        ims = ch.groupby("t")["initial_margin"].sum()
        ims = ims[ims > 0]
    else:
        ims = pd.Series(dtype=float)
    im_open = float(ims.iloc[0]) if len(ims) else float("nan")
    im_peak = float(ims.max()) if len(ims) else float("nan")
    im_trough = float(ims.min()) if len(ims) else float("nan")
    im_mean = float(ims.mean()) if len(ims) else float("nan")
    im_p2t = (im_peak / im_trough) if (len(ims) and im_trough > 0) else float("nan")
    im_jump = float(ims.pct_change().max()) if len(ims) > 1 else float("nan")
    sig = np.asarray(sim.sigma_t_array, dtype=float)
    sig = sig[np.isfinite(sig) & (sig > 0)]
    if len(sig) and len(ims) and im_open > 0 and sig[0] > 0 and sig.max() / sig[0] > 1.0:
        resp = (im_peak / im_open - 1.0) / (sig.max() / sig[0] - 1.0)
    else:
        resp = float("nan")
    return dict(
        drawdown=float(100.0 * (mid.min() / mid[0] - 1.0)) if len(mid) else float("nan"),
        client_defaults=int(len(cl)),
        cm_defaults=int(bcm + nbcm), bcm_defaults=int(bcm), nbcm_defaults=int(nbcm),
        deepest_waterfall=deepest, mutualised_l3=int(deepest >= 3), mutualised_l4=int(deepest >= 4),
        total_df=float(ccp.total_df), ccp_cash_used=float(CCP_CASH - ccp.cash),
        im_open=im_open, im_peak=im_peak, im_trough=im_trough, im_mean=im_mean,
        im_peak_to_trough=im_p2t, im_max_jump_pct=im_jump, im_responsiveness=resp,
    )


def run_one(spec: dict) -> dict:
    """Execute one (fund, mode, im-regime, reanchor, scenario, c, seed) run."""
    G.IM_MODE = spec["im_mode"]; G.IM_FLAT_FRAC = spec["im_flat"]
    G.REANCHOR_ON_GAP = bool(spec["reanchor"])
    row = {k: spec[k] for k in ("family", "fund", "regime", "mode", "im_regime",
                                 "im_mode", "im_flat", "reanchor", "scenario", "c", "seed")}
    try:
        fspec = _FUNDS[spec["fund"]] if spec["regime"] == "stressed" else None
        seed = int(spec["seed"]); c = float(spec["c"])
        if spec["regime"] == "calm":                         # H2 calm-cost
            V = pd.read_csv("data/fv_calm.csv")["V_smooth"].to_numpy()
            w0 = 0; wn = int(day_start_steps("calm")[15])    # first ~15 calm sessions
            p = ModelParams(n_fundamental=30, n_momentum=20, n_zi=40, n_bcm=10, n_nbcm=5,
                            n_bcm_with_clients=5, v0=float(V[w0]), tick_size=0.25,
                            dt_minutes=1.0, **CALIBRATED["calm"], stressed=False)
            traders = build_traders(p, seed=seed)
            ccp = build_clearing_tier(traders, p, seed=seed, direct=False)
            sim = Simulation(p, traders, seed=seed, ccp=ccp, v_start=w0)
            sim.run(wn)
            row.update(_metrics(sim, ccp)); row["error"] = ""
            return row
        # stressed contagion run
        V, SIG, w0, wn = _fv_window(spec["fund"], fspec)
        p = ModelParams(n_fundamental=30, n_momentum=20, n_zi=40, n_bcm=10, n_nbcm=5,
                        n_bcm_with_clients=5, v0=float(V[w0]), tick_size=0.25, dt_minutes=1.0,
                        **fspec["theta"], stressed=True, fv_csv=fspec["fv"])
        traders = build_traders(p, seed=seed)
        ccp = build_clearing_tier(traders, p, seed=seed, direct=(spec["mode"] == "direct"))
        sim = Simulation(p, traders, seed=seed, ccp=ccp, v_start=w0)
        w = V[w0:w0 + wn]
        gapped = w[0] * np.exp(c * (np.log(w) - np.log(w[0])))
        if spec["scenario"] == "shock":
            out = np.full(len(gapped), gapped[0], dtype=float)
            out[len(gapped) // 2:] = gapped.min()
            sim.v_array = out
        else:
            sim.v_array = gapped
        sim.sigma_t_array = SIG[w0:w0 + wn]
        sim.run(wn)
        row.update(_metrics(sim, ccp)); row["error"] = ""
    except Exception as e:                                    # never let one run kill the batch
        row["error"] = f"{type(e).__name__}: {e}"
        row["_trace"] = traceback.format_exc()[-800:]
    return row


# ── Master spec set (deduplicated; sliced per hypothesis in aggregate) ─────────
def _key(s: dict) -> str:
    return "|".join(str(s[k]) for k in ("fund", "regime", "mode", "im_regime",
                                         "reanchor", "scenario", "c", "seed"))

def generate_specs(quick: bool = False) -> list:
    funds = list(_FUNDS.keys())
    seeds = SEEDS[:2] if quick else SEEDS
    cg = C_GRID[:2] if quick else C_GRID
    cg6 = C_GRID_E6[:2] if quick else C_GRID_E6
    calm_seeds = CALM_SEEDS[:2] if quick else CALM_SEEDS
    out, seen = [], set()

    def add(family, fund, mode, im_regime, reanchor, scenario, c, seed, regime="stressed"):
        im_mode, im_flat = IM_REGIMES[im_regime]
        s = dict(family=family, fund=fund, regime=regime, mode=mode, im_regime=im_regime,
                 im_mode=im_mode, im_flat=im_flat, reanchor=int(reanchor), scenario=scenario,
                 c=float(c), seed=int(seed))
        k = _key(s)
        if k not in seen:
            seen.add(k); out.append(s)

    for fund in funds:
        for seed in seeds:
            # H1: tiered vs direct (reactive, clean open, gapped)
            for mode in ("tiered", "direct"):
                for c in cg:
                    add("H1", fund, mode, "reactive", True, "gapped", c, seed)
            # H2: margin regimes (tiered, gapped) — reactive shared with H1
            for im_regime in ("static", "flat12", "flat05"):
                for c in cg:
                    add("H2", fund, "tiered", im_regime, True, "gapped", c, seed)
    # E6: gaps vs shock (Kalman, tiered, reactive). gapped reuses H1; add shock.
    for seed in seeds:
        for c in cg6:
            add("E6", "kalman", "tiered", "reactive", True, "shock", c, seed)
    # Open-disorder: no-reanchor variant (Kalman, tiered+direct, reactive). reanchor=True reuses H1.
    for seed in seeds:
        for mode in ("tiered", "direct"):
            for c in cg:
                add("OPENDIS", "kalman", mode, "reactive", False, "gapped", c, seed)
    # H2 calm-cost: calm regime, each IM regime, no amplification
    for seed in calm_seeds:
        for im_regime in ("reactive", "static", "flat12", "flat05"):
            add("CALMCOST", "kalman", "tiered", im_regime, True, "gapped", 1.0, seed, regime="calm")
    return out


# ── Run loop (resumable, incremental) ─────────────────────────────────────────
_ROW_FIELDS = ["family", "fund", "regime", "mode", "im_regime", "im_mode", "im_flat",
               "reanchor", "scenario", "c", "seed", "drawdown", "client_defaults",
               "cm_defaults", "bcm_defaults", "nbcm_defaults", "deepest_waterfall",
               "mutualised_l3", "mutualised_l4", "total_df", "ccp_cash_used", "im_open",
               "im_peak", "im_trough", "im_mean", "im_peak_to_trough", "im_max_jump_pct",
               "im_responsiveness", "error"]

def _done_keys(rows_csv: Path) -> set:
    if not rows_csv.exists():
        return set()
    df = pd.read_csv(rows_csv)
    done = set()
    for _, r in df.iterrows():
        if str(r.get("error", "")) == "" or pd.isna(r.get("error")):
            done.add("|".join(str(r[k]) for k in ("fund", "regime", "mode", "im_regime",
                                                    "reanchor", "scenario", "c", "seed")))
    return done

def run_all(quick: bool = False):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    rows_csv = OUTDIR / ("rows_smoke.csv" if quick else "rows.csv")
    specs = generate_specs(quick=quick)
    done = _done_keys(rows_csv)
    todo = [s for s in specs if _key(s) not in done]
    print(f"[experiments] funds={list(_FUNDS)} seeds={len(SEEDS if not quick else SEEDS[:2])} "
          f"c-grid={len(C_GRID if not quick else C_GRID[:2])}  total specs={len(specs)} "
          f"done={len(done)} todo={len(todo)}  nproc={NPROC}  -> {rows_csv}", flush=True)
    new = not rows_csv.exists()
    f = rows_csv.open("a", newline="")
    w = csv.DictWriter(f, fieldnames=_ROW_FIELDS, extrasaction="ignore")
    if new:
        w.writeheader()
    t0 = time.time(); n = 0
    pool = mp.Pool(NPROC) if NPROC > 1 else None
    it = (pool.imap_unordered(run_one, todo, chunksize=4) if pool else map(run_one, todo))
    for row in it:
        w.writerow(row); f.flush(); n += 1
        if row.get("error"):
            print(f"  ERR {row.get('family')} {row.get('fund')} c={row.get('c')} "
                  f"seed={row.get('seed')}: {row['error']}", flush=True)
        if n % 50 == 0:
            el = time.time() - t0
            print(f"  {n}/{len(todo)} done  {el/60:.1f} min  ({el/n:.2f}s/run)", flush=True)
    if pool:
        pool.close(); pool.join()
    f.close()
    print(f"[experiments] {n} runs in {(time.time()-t0)/60:.1f} min -> {rows_csv}", flush=True)
    aggregate(rows_csv)


# ── Aggregation (slice the master rows per hypothesis) ────────────────────────
def _breach_table(df: pd.DataFrame, by: list) -> pd.DataFrame:
    """Per-seed smallest c reaching L3 / L4, summarised over seeds within `by` groups."""
    recs = []
    for keys, g in df.groupby(by + ["seed"]):
        keys = keys if isinstance(keys, tuple) else (keys,)
        for lvl, col in ((3, "mutualised_l3"), (4, "mutualised_l4")):
            hit = g[g[col] == 1]["c"]
            recs.append({**dict(zip(by, keys[:len(by)])), "seed": keys[-1],
                         "level": lvl, "breach_c": (hit.min() if len(hit) else np.nan)})
    bt = pd.DataFrame(recs)
    out = []
    for keys, g in bt.groupby(by + ["level"]):
        keys = keys if isinstance(keys, tuple) else (keys,)
        bc = g["breach_c"].dropna()
        out.append({**dict(zip(by, keys[:len(by)])), "level": keys[-1],
                    "frac_seeds_breached": len(bc) / len(g) if len(g) else np.nan,
                    "breach_c_median": bc.median() if len(bc) else np.nan,
                    "breach_c_p10": bc.quantile(.1) if len(bc) else np.nan,
                    "breach_c_p90": bc.quantile(.9) if len(bc) else np.nan})
    return pd.DataFrame(out)

def _ens(df: pd.DataFrame, by: list) -> pd.DataFrame:
    agg = df.groupby(by).agg(
        n=("seed", "nunique"),
        drawdown=("drawdown", "mean"),
        client_def_mean=("client_defaults", "mean"), client_def_med=("client_defaults", "median"),
        client_def_p90=("client_defaults", lambda x: x.quantile(.9)),
        cm_def_mean=("cm_defaults", "mean"), nbcm_def_mean=("nbcm_defaults", "mean"),
        deepest_mean=("deepest_waterfall", "mean"), deepest_max=("deepest_waterfall", "max"),
        frac_l3=("mutualised_l3", "mean"), frac_l4=("mutualised_l4", "mean"),
        df_mean=("total_df", "mean"), ccp_used_mean=("ccp_cash_used", "mean"),
    ).reset_index()
    return agg

def aggregate(rows_csv: Path | None = None):
    rows_csv = Path(rows_csv) if rows_csv else (OUTDIR / "rows.csv")
    if not rows_csv.exists():
        print(f"no rows at {rows_csv}"); return
    df = pd.read_csv(rows_csv)
    df = df[(df["error"].isna()) | (df["error"].astype(str) == "")].copy()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    out = {}

    st = df[df["regime"] == "stressed"]
    # H1 — tiered vs direct (reactive, clean, gapped)
    h1 = st[(st["im_regime"] == "reactive") & (st["reanchor"] == 1) & (st["scenario"] == "gapped")]
    out["summary_h1.csv"] = _ens(h1, ["fund", "mode", "c"])
    out["summary_h1_breach.csv"] = _breach_table(h1, ["fund", "mode"])
    # H2 — margin regimes (tiered, clean, gapped)
    h2 = st[(st["mode"] == "tiered") & (st["reanchor"] == 1) & (st["scenario"] == "gapped")]
    out["summary_h2.csv"] = _ens(h2, ["fund", "im_regime", "c"])
    out["summary_h2_breach.csv"] = _breach_table(h2, ["fund", "im_regime"])
    h2im = h2.groupby(["fund", "im_regime", "c"]).agg(
        im_mean=("im_mean", "mean"), im_peak=("im_peak", "mean"),
        im_peak_to_trough=("im_peak_to_trough", "mean"),
        im_max_jump_pct=("im_max_jump_pct", "mean"),
        im_responsiveness=("im_responsiveness", "mean")).reset_index()
    out["summary_h2_margin_metrics.csv"] = h2im
    # E6 — gaps vs shock (kalman, tiered, reactive)
    e6 = st[(st["fund"] == "kalman") & (st["mode"] == "tiered") & (st["im_regime"] == "reactive")
            & (st["reanchor"] == 1) & (st["c"].isin(C_GRID_E6))]
    out["summary_e6.csv"] = _ens(e6, ["scenario", "c"])
    # Open-disorder — reanchor True vs False (kalman, tiered, reactive, gapped)
    od = st[(st["fund"] == "kalman") & (st["mode"] == "tiered") & (st["im_regime"] == "reactive")
            & (st["scenario"] == "gapped")]
    out["summary_opendisorder.csv"] = _ens(od, ["reanchor", "c"])
    # H2 calm-cost
    calm = df[df["regime"] == "calm"]
    if len(calm):
        out["summary_h2_calmcost.csv"] = calm.groupby("im_regime").agg(
            n=("seed", "nunique"), im_mean=("im_mean", "mean"),
            im_peak=("im_peak", "mean")).reset_index()

    for name, t in out.items():
        t.to_csv(OUTDIR / name, index=False)
    _write_summary_md(out)
    print(f"[aggregate] wrote {len(out)} summaries + SUMMARY.md to {OUTDIR}/")
    # console headline
    if "summary_h1_breach.csv" in out:
        print("\n=== H1 breach multipliers (smallest c reaching a level, over seeds) ===")
        print(out["summary_h1_breach.csv"].to_string(index=False))


def _write_summary_md(out: dict):
    L = ["# Overnight experiment summary", "",
         f"Generated {time.strftime('%Y-%m-%d %H:%M')}. Rows: `{OUTDIR}/rows.csv`. "
         "Per-hypothesis CSVs in the same folder.", ""]
    def tbl(name, title, note=""):
        if name in out and len(out[name]):
            L.append(f"## {title}")
            if note: L.append(note)
            L.append("")
            L.append(out[name].round(3).to_markdown(index=False))
            L.append("")
    tbl("summary_h1_breach.csv", "H1 — tiered vs direct: mutualisation breach multiplier",
        "Smallest amplifier c reaching waterfall L3 (pooled DF) / L4 (survivor cash), over seeds. "
        "Tiered should breach at a HIGHER c than direct (the buffer clause).")
    tbl("summary_h2_breach.csv", "H2 — margin regimes: breach multiplier",
        "Does reactive (procyclical) IM breach earlier than flat-12 (through-the-cycle)?")
    tbl("summary_h2_calmcost.csv", "H2 — calm-period collateral cost",
        "Mean posted IM in the calm regime per margin scheme (the procyclicality vs cost trade-off).")
    tbl("summary_e6.csv", "E6 — overnight gaps vs single shock (matched drawdown)", "")
    tbl("summary_opendisorder.csv", "Open-disorder sensitivity (reanchor 1=clean / 0=disorderly open)", "")
    (OUTDIR / "SUMMARY.md").write_text("\n".join(L))


_FUNDS = _fundamentals()

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    if mode == "smoke":
        run_all(quick=True)
    elif mode == "all":
        run_all(quick=False)
    elif mode == "aggregate":
        aggregate()
    else:
        print(__doc__)
