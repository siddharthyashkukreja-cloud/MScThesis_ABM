"""Descriptive "normal model" runner -> small CSVs for results_figures.ipynb.

Runs N_SEEDS x {calm, stressed} with the clearing tier active (the committed D77/D78
config: IM floor 4% / MPOR 1 / no cap; 8% capital-adequacy floor) and writes
pre-aggregated CSVs to output/results/descriptive/ so the NOTEBOOK only reads + plots
(no in-notebook simulation). This separates the heavy compute (run once, here) from the
presentation (fast, reproducible figures).

    N_SEEDS=40 PYTHONPATH=. python3 scripts/run_model_descriptive.py

Runs the WHOLE calibrated window per regime (calm 75 / stressed 73 sessions, ~30k steps) —
~28s/seed, so N_SEEDS=40 over both regimes is ~35-40 min. Set WINDOW_DAYS="10:30" to focus a
day range (e.g. the stressed crash trough); MAX_STEPS caps steps for a quick smoke.

Outputs (all small; bands are p25/p50/p75 across seeds [x members, for kappa]):
  descriptive/representative_timeseries.csv  one seed, per step: buy/sell vol, mid, V_t  (figs A, E)
  descriptive/capital_ratio_band.csv         regime,t,mtype, kappa p25/p50/p75            (fig B)
  descriptive/im_path.csv                    regime,t, im_frac, im_total p25/p50/p75       (fig C)
  descriptive/im_by_type.csv                 regime,t,type, im p25/p50/p75                 (fig D)
  descriptive/margin_calls.csv               regime,hour, calls p25/p50/p75                (fig F)
  descriptive/member_clientclearing.csv      regime,seed,agent_id,n_clients,min_kappa,breached  (H1')
  descriptive/summary_scalars.csv            regime,seed, defaults / IM / DF / calls / client-IM share
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

from model import globals as G
from model.globals import ModelParams, CALIBRATED, day_start_steps, CONTRACT_USD, im_fraction
from model.agents import (FundamentalTrader, MomentumTrader, ZeroIntelligenceTrader,
                          BankingClearingMember, NonBankingClearingMember)
from model.run_simulation import build_traders, build_clearing_tier
from model.simulation import Simulation

N_SEEDS = int(os.environ.get("N_SEEDS", "40"))
SEED0   = int(os.environ.get("SEED0", "42"))
REGIMES = os.environ.get("REGIMES", "calm,stressed").split(",")
REP_SEED = SEED0                                   # representative seed for per-step figs (A, E)
IM_SNAP_EVERY = 60                                 # posted-IM snapshot cadence (steps)
OUT = os.environ.get("DESC_OUT", "output/results/descriptive")
os.makedirs(OUT, exist_ok=True)


def _typ(a):
    if isinstance(a, BankingClearingMember):    return "BCM"
    if isinstance(a, NonBankingClearingMember): return "NBCM"
    if isinstance(a, MomentumTrader):           return "MT"
    if isinstance(a, ZeroIntelligenceTrader):   return "ZI"
    if isinstance(a, FundamentalTrader):        return "FT"
    return "other"


def _window(regime):
    """Full calibrated window by default (calm 75 / stressed 73 sessions).
    Optional WINDOW_DAYS="start:end" slices a day range (e.g. "10:30" to focus the
    stressed crash trough) — used only if set; otherwise the whole regime runs."""
    fv = pd.read_csv(G.FV_CSV[regime])
    V, SIG = fv["V_smooth"].to_numpy(float), fv["sigma_t"].to_numpy(float)
    wd = os.environ.get("WINDOW_DAYS", "").strip()
    if wd:
        st = list(day_start_steps(regime))
        a, b = (int(x) for x in wd.split(":"))
        s = st[a] if a < len(st) else 0
        e = st[b] if b < len(st) else len(V)
    else:
        s, e = 0, len(V)                                # whole calibrated window
    return V, SIG, s, e - s


rep_rows, cyc_frames, mc_rows, imt_rows, mem_rows, scal_rows, imfrac_rows = [], [], [], [], [], [], []
rep_kappa_frames, rep_call_frames, book_depth_frames = [], [], []   # rep seed: kappa paths (B), call counts (F), book depth (A2)
cli_frames, wf_rows = [], []   # per-client-default detail (type/CM/position/loss) + per-default waterfall L1-L5
frz_rows, prt_rows, balance_frames = [], [], []   # client freezes, porting events, per-CM balance-sheet bands

for regime in REGIMES:
    V, SIG, s, n = _window(regime)
    for seed in range(SEED0, SEED0 + N_SEEDS):
        p = ModelParams(n_fundamental=30, n_momentum=20, n_zi=40, n_bcm=10, n_nbcm=5,
                        n_bcm_with_clients=5, v0=float(V[s]), tick_size=0.25, dt_minutes=1.0,
                        **CALIBRATED[regime], stressed=(regime == "stressed"))
        tr = build_traders(p, seed=seed)
        ccp = build_clearing_tier(tr, p, seed=seed, direct=False)
        n = min(n, int(os.environ.get("MAX_STEPS", n)))      # MAX_STEPS caps the window (smoke only)
        sim = Simulation(p, tr, seed=seed, ccp=ccp, v_start=s)
        sim.v_array, sim.sigma_t_array = V[s:s + n], SIG[s:s + n]
        tymap = {a.agent_id: _typ(a) for a in tr}
        for m in ccp.members.values():
            tymap.setdefault(m.agent_id, _typ(m))
        ncl = {m.agent_id: len(m.client_ids) for m in ccp.members.values()
               if isinstance(m, BankingClearingMember)}
        lot = p.volume_lot
        rep = (seed == REP_SEED)
        prev_mid = float(V[s])
        im_keys = ["FT", "MT", "ZI", "BCM"]

        for t in range(n):
            sim.step()
            mid = sim.history["mid_price"][-1]
            if rep:                                          # per-step order flow + mid/V_t (one seed)
                ref = prev_mid if prev_mid == prev_mid else mid
                b = sv = 0.0
                for f in sim.lob.step_fills:
                    q = f.qty * lot
                    if ref != ref:        b += .5 * q; sv += .5 * q
                    elif f.price > ref:   b += q
                    elif f.price < ref:   sv += q
                    else:                 b += .5 * q; sv += .5 * q
                rep_rows.append((regime, t, b, sv, mid if mid == mid else np.nan, float(V[s + t])))
            if mid == mid:
                prev_mid = mid
            if t % IM_SNAP_EVERY == 0:                       # posted IM by type (all seeds)
                agg = {k: 0.0 for k in im_keys}
                for a in tr:
                    ty = tymap[a.agent_id]
                    if ty in agg:
                        agg[ty] += getattr(a, "_posted_im", 0.0)
                for m in ccp.members.values():
                    if tymap[m.agent_id] == "BCM":
                        agg["BCM"] += getattr(m, "_posted_im", 0.0)
                for k in im_keys:
                    imt_rows.append((regime, seed, t, k, agg[k]))

        ch = pd.DataFrame(sim.clearing_history)
        cl = pd.DataFrame(sim.client_history)
        if len(cl):                                          # per-client-default detail (type, CM, position, loss)
            cli_frames.append(cl.assign(regime=regime, seed=seed))
        for w in getattr(ccp, "_waterfall_log", []) or []:   # per-default waterfall decomposition (L1-L5)
            wf_rows.append({**w, "regime": regime, "seed": seed})
        for f in getattr(sim, "freeze_log", []) or []:       # client freeze onsets (own-distress / CM-contagion)
            frz_rows.append({**f, "regime": regime, "seed": seed})
        for pp in getattr(sim, "porting_log", []) or []:     # client porting on a member default
            prt_rows.append({**pp, "regime": regime, "seed": seed})
        if len(ch):
            ch["capital_ratio"] = pd.to_numeric(ch["capital_ratio"], errors="coerce")
            # per-BCM dose-response (H1'): min kappa + breach by #clients
            for aid, sub in ch[ch.kind == "BCM"].groupby("agent_id"):
                mk = float(np.nanmin(sub["capital_ratio"].to_numpy()))
                nc = ncl.get(aid, 0)
                mem_rows.append((regime, seed, aid, nc, mk, int(mk <= G.LR_FLOOR_BCM)))
            # capital-ratio band by member type (BCMc / BCMo / NBCM)
            cm = ch[ch.kind.isin(["BCM", "NBCM"])].copy()
            cm = cm[np.isfinite(cm["capital_ratio"])]
            cm["mtype"] = np.where(cm.kind == "NBCM", "NBCM",
                          np.where(cm.agent_id.map(lambda a: ncl.get(a, 0)) > 0, "BCMc", "BCMo"))
            cyc_frames.append(cm[["t", "mtype", "capital_ratio"]].assign(regime=regime))
            balance_frames.append(cm[["t", "mtype", "cash", "maintenance_margin",
                                      "df_contribution"]].assign(regime=regime))
            if rep:   # rep seed: individual member kappa trajectories (fig B) + per-cycle simultaneity (fig F)
                rep_kappa_frames.append(cm[["t", "agent_id", "mtype", "capital_ratio"]].assign(regime=regime))
                rep_call_frames.append(ch[ch.kind.isin(["BCM", "NBCM"])].groupby("t")["call_indicator"]
                                       .sum().reset_index(name="n_calls").assign(regime=regime))
                book_depth_frames.append(pd.DataFrame({                       # resting bid/ask depth (fig A2)
                    "regime": regime, "t": range(len(sim.history["bid_depth"])),
                    "bid_depth": sim.history["bid_depth"], "ask_depth": sim.history["ask_depth"]}))
            # margin calls per hour: sum across members within each 60-step bin, per seed
            calls = ch.groupby("t")["call_indicator"].sum()
            hourly = calls.groupby((calls.index // 60).astype(int)).sum()
            for hr, cc in hourly.items():
                mc_rows.append((regime, seed, int(hr), float(cc)))
            # dollar IM total per cycle
            im_tot = ch.groupby("t")["initial_margin"].sum()
            for tt, vv in im_tot.items():
                imfrac_rows.append((regime, seed, int(tt), float(vv)))
            cm_def = int(ch[ch.has_defaulted & (ch.kind == "BCM")].agent_id.nunique())
            nb_def = int(ch[ch.has_defaulted & (ch.kind == "NBCM")].agent_id.nunique())
            im_mean = float(im_tot.mean()); n_calls = int(ch["call_indicator"].sum())
        else:
            cm_def = nb_def = n_calls = 0; im_mean = 0.0
        scal_rows.append((regime, seed, cm_def, nb_def, len(cl), im_mean, float(ccp.total_df), n_calls))
        print(f"  {regime} seed {seed} done", flush=True)

# ── aggregate + write ────────────────────────────────────────────────────────
def _band(df, by, val):
    g = df.groupby(by)[val]
    return g.median().rename("p50").to_frame().join(
        g.quantile(.25).rename("p25")).join(g.quantile(.75).rename("p75")).reset_index()

pd.DataFrame(rep_rows, columns=["regime", "t", "buy_vol", "sell_vol", "mid", "V_t"]
             ).to_csv(f"{OUT}/representative_timeseries.csv", index=False)

cyc = pd.concat(cyc_frames, ignore_index=True) if cyc_frames else pd.DataFrame(columns=["regime","t","mtype","capital_ratio"])
_band(cyc, ["regime", "t", "mtype"], "capital_ratio").to_csv(f"{OUT}/capital_ratio_band.csv", index=False)

imt = pd.DataFrame(imt_rows, columns=["regime", "seed", "t", "type", "im"])
_band(imt, ["regime", "t", "type"], "im").to_csv(f"{OUT}/im_by_type.csv", index=False)

mc = pd.DataFrame(mc_rows, columns=["regime", "seed", "hour", "calls"])
mcb = _band(mc, ["regime", "hour"], "calls")
# hour-bin -> fractional trading DAY (handles uneven session lengths via day_start_steps)
_wc = {r: (_window(r)[2], list(day_start_steps(r))) for r in REGIMES}
def _to_day(r, h):
    s0, st = _wc[r]; step = s0 + int(h) * 60
    d = max(int(np.searchsorted(st, step, side="right") - 1), 0)
    frac = (step - st[d]) / max(1, st[d + 1] - st[d]) if d + 1 < len(st) else 0.0
    return d + frac
mcb["day"] = [_to_day(r, h) for r, h in zip(mcb.regime, mcb.hour)]
mcb.to_csv(f"{OUT}/margin_calls.csv", index=False)

imf = pd.DataFrame(imfrac_rows, columns=["regime", "seed", "t", "im_total"])
imband = _band(imf, ["regime", "t"], "im_total")
# CHARGED IM fraction path = im_fraction(daily EWMA sigma / sqrt(390)) — the DAILY close-to-close vol
# the model actually margins on (`simulation._sigma_daily_row`), NOT the per-step sigma_t (which spikes
# to an intraday artifact ~85%). Rebuilt once per regime from a throwaway sim so it matches the sim exactly.
import math as _math
def _charged_frac_row(regime):
    fv = pd.read_csv(G.FV_CSV[regime]); Vr = fv["V_smooth"].to_numpy(float); Sr = fv["sigma_t"].to_numpy(float)
    s0 = _window(regime)[2]
    pp = ModelParams(n_fundamental=30, n_momentum=20, n_zi=40, n_bcm=10, n_nbcm=5, n_bcm_with_clients=5,
                     v0=float(Vr[s0]), tick_size=0.25, dt_minutes=1.0, **CALIBRATED[regime],
                     stressed=(regime == "stressed"))
    trr = build_traders(pp, seed=SEED0); cc = build_clearing_tier(trr, pp, seed=SEED0, direct=False)
    sm = Simulation(pp, trr, seed=SEED0, ccp=cc, v_start=s0); sm.v_array, sm.sigma_t_array = Vr[s0:], Sr[s0:]
    return getattr(sm, "_sigma_daily_row", None)
imfr = []
for regime in REGIMES:
    row = _charged_frac_row(regime); SQ = _math.sqrt(390)
    for tt in (sorted(imf[imf.regime == regime]["t"].unique()) if len(imf) else []):
        sd = row[int(tt)] if (row is not None and int(tt) < len(row)) else float("nan")
        imfr.append((regime, int(tt), im_fraction(sd / SQ) if sd == sd else float("nan")))
imfrac = pd.DataFrame(imfr, columns=["regime", "t", "im_frac"])
imband.merge(imfrac, on=["regime", "t"], how="left").to_csv(f"{OUT}/im_path.csv", index=False)

pd.DataFrame(mem_rows, columns=["regime", "seed", "agent_id", "n_clients", "min_kappa", "breached"]
             ).to_csv(f"{OUT}/member_clientclearing.csv", index=False)

if rep_kappa_frames:   # rep seed: individual member kappa paths (fig B) + per-cycle call counts (fig F) + book depth (A2)
    pd.concat(rep_kappa_frames, ignore_index=True).to_csv(f"{OUT}/member_kappa_path.csv", index=False)
    pd.concat(rep_call_frames, ignore_index=True).to_csv(f"{OUT}/margincall_percycle.csv", index=False)
    pd.concat(book_depth_frames, ignore_index=True).to_csv(f"{OUT}/book_depth.csv", index=False)

scal = pd.DataFrame(scal_rows, columns=["regime", "seed", "cm_defaults", "nbcm_defaults",
                                        "client_defaults", "im_mean", "total_df", "n_margin_calls"])
# client IM share per seed (mean over snapshots): clients (FT+MT+ZI) / (clients + BCM house)
sh = imt.pivot_table(index=["regime", "seed", "t"], columns="type", values="im", aggfunc="sum").reset_index()
for c in ["FT", "MT", "ZI", "BCM"]:
    if c not in sh: sh[c] = 0.0
sh["client_im_share"] = (sh.FT + sh.MT + sh.ZI) / (sh.FT + sh.MT + sh.ZI + sh.BCM).replace(0, np.nan)
scal = scal.merge(sh.groupby(["regime", "seed"])["client_im_share"].mean().reset_index(),
                  on=["regime", "seed"], how="left")
scal.to_csv(f"{OUT}/summary_scalars.csv", index=False)

# per-client-default detail (client type FT/MT/ZI, carrying CM, position-at-default, loss, timing)
(pd.concat(cli_frames, ignore_index=True) if cli_frames else pd.DataFrame(
    columns=["t", "client_id", "kind", "cm_id", "cm_kind", "shortfall", "closeout_loss",
             "im_posted", "assumed_pos", "regime", "seed"])
 ).to_csv(f"{OUT}/client_defaults.csv", index=False)
# per-default waterfall decomposition (level reached + L1-L5 amounts + mutualised)
pd.DataFrame(wf_rows if wf_rows else [], columns=["member_id", "is_banking", "is_client",
    "deficit", "level", "L1_own_df", "L2_sitg", "L3_pooled_df", "L4_survivor_cash",
    "L5_ccp_cash", "mutualised", "regime", "seed"]).to_csv(f"{OUT}/waterfall_events.csv", index=False)
# client freeze onsets (own-distress vs CM-contagion, with kappa + timing) + porting on member default
pd.DataFrame(frz_rows, columns=["t", "client_id", "kind", "cm_id", "cm_kind", "reason",
    "kappa", "regime", "seed"]).to_csv(f"{OUT}/client_freezes.csv", index=False)
pd.DataFrame(prt_rows, columns=["t", "defaulted_cm", "cm_kind", "n_ported", "n_unported",
    "unported_position", "regime", "seed"]).to_csv(f"{OUT}/porting_events.csv", index=False)
# per-CM balance-sheet bands (cash / maintenance margin / DF contribution) by member type
if balance_frames:
    bal = pd.concat(balance_frames, ignore_index=True).melt(
        id_vars=["regime", "t", "mtype"], value_vars=["cash", "maintenance_margin", "df_contribution"],
        var_name="metric", value_name="val")
    _band(bal, ["regime", "t", "mtype", "metric"], "val").to_csv(f"{OUT}/member_balance_band.csv", index=False)

print(f"\nDONE -> {OUT}/  (N_SEEDS={N_SEEDS}, regimes={REGIMES})")
print(scal.groupby("regime")[["cm_defaults", "client_defaults", "im_mean", "total_df",
                              "n_margin_calls", "client_im_share"]].mean().round(3).to_string())
