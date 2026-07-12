"""
FastAPI wrapper around the model for the Lovable demo frontend.

    pip install fastapi uvicorn
    uvicorn api.server:app --port 8000        (from the repo root)

POST /simulate runs one cleared session-window live (~0.3 s per simulated day)
and returns JSON series + clearing outcomes. POST /simulate_stream streams the
same run day by day as NDJSON (one line per simulated session, with per-agent
topology state) so the frontend can animate charts and the network map while
the simulation runs. GET /meta returns panel defaults.

Empirical FV = the thesis baseline (model.md §2.3); synthetic FV = the joint
agent+FV two-factor log-vol optimum (thesis §4.5, output/overnight_joint_logvol/
calibration.json reverse_optimum; synthetic runs set IM_DAILY=False to match the
thesis stage-C convention). The annualised-vol slider overrides sigma_d on the
synthetic arm only; agent theta stays at the joint optimum, so moving it
demonstrates the theta/FV coupling that joint calibration resolves.
"""

import json
import math
import sys
import tempfile
import threading
import time
from pathlib import Path

import logging

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

log = logging.getLogger("uvicorn.error")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from model import globals as G
from model.globals import ModelParams, CALIBRATED, CCP_CASH
from model.run_simulation import build_traders, build_clearing_tier
from model.simulation import Simulation
from model.agents import (FundamentalTrader, MomentumTrader, ZeroIntelligenceTrader,
                          BankingClearingMember, NonBankingClearingMember)
from data import v_gbm

BARS_PER_DAY = 390
MIN_ANN = math.sqrt(390 * 252)
JOINT = json.loads((REPO / "output/overnight_joint_logvol/calibration.json").read_text())["reverse_optimum"]
WF_LABELS = ["none", "L1 defaulter DF", "L2 CCP skin-in-the-game",
             "L3 pooled default fund", "L4 survivor cash", "L5 CCP capital"]
KIND_SHORT = {"FundamentalTrader": "FT", "MomentumTrader": "MT",
              "ZeroIntelligenceTrader": "ZI", "BankingClearingMember": "BCM",
              "NonBankingClearingMember": "NBCM"}

_LOCK = threading.Lock()
_G_DEFAULTS = {k: getattr(G, k) for k in
               ("CLOSEOUT_RECOVERY", "IM_MODE", "IM_FLAT_FRAC", "CLIENT_MARGIN_NETTING",
                "IM_DAILY", "IM_FLOOR", "DF_ODD_FIXED")}
_FV_CACHE: dict = {}


def _fv_empirical(regime):
    if regime not in _FV_CACHE:
        df = pd.read_csv(REPO / G.FV_CSV[regime])
        dates = pd.to_datetime(df["ts"]).dt.date.to_numpy()
        starts = [0] + [i for i in range(1, len(dates)) if dates[i] != dates[i - 1]]
        _FV_CACHE[regime] = (df, starts)
    return _FV_CACHE[regime]


class SimRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, alias_generator=to_camel)

    regime: str = Field("stressed", pattern="^(calm|stressed)$")
    fv_source: str = Field("empirical", pattern="^(empirical|synthetic)$")
    ann_vol_pct: float | None = Field(None, ge=10, le=90)
    recovery: float = Field(0.80, ge=0.30, le=1.00)
    horizon_days: int = Field(20, ge=3, le=75)
    seed: int = Field(42, ge=0, le=10_000)
    im_mode: str = Field("reactive", pattern="^(reactive|flat)$")
    flat_im_pct: float = Field(8.0, ge=2, le=20)
    im_floor_pct: float = Field(4.0, ge=1, le=10)
    fixed_df: bool = False
    structure: str = Field("tiered", pattern="^(tiered|direct)$")
    netting: str = Field("gross", pattern="^(gross|net)$")


def _clean(x):
    f = float(x)
    return None if (f != f or math.isinf(f)) else round(f, 4)


def _json_safe(o):
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, float) and (o != o or math.isinf(o)):
        return None
    if isinstance(o, (np.floating, np.integer)):
        return _json_safe(float(o))
    return o


def _dumps(o) -> str:
    return json.dumps(_json_safe(o))


def _prepare(req: SimRequest) -> dict:
    if req.fv_source == "synthetic" and req.regime == "calm":
        req.regime = "stressed"
        log.info("synthetic FV requested with calm regime -> coerced to stressed")
    log.info("simulate: %s", req.model_dump())
    regime = req.regime
    if req.fv_source == "synthetic":
        sv = dict(JOINT["sv_override"])
        ann_vol = req.ann_vol_pct or round(sv["sigma_d"] * MIN_ANN * 100, 1)
        sv["sigma_d"] = ann_vol / 100.0 / MIN_ANN
        tmp = Path(tempfile.gettempdir()) / f"api_fv_{req.seed}_{ann_vol}_{req.horizon_days}.csv"
        V = v_gbm.generate(regime, seed=req.seed, n_days=req.horizon_days, out_path=tmp,
                           mu_override=JOINT["mu_override"], sv_override=sv,
                           jump_override=JOINT["jump_override"])
        n_steps = req.horizon_days * BARS_PER_DAY
        return dict(regime=regime, fv_csv=str(tmp), v0=float(V[0]), n_steps=n_steps,
                    theta={**CALIBRATED[regime], **JOINT["agent_overrides"]},
                    day_starts=list(range(0, n_steps, BARS_PER_DAY)),
                    ann_vol=ann_vol, synthetic=True)
    df, starts = _fv_empirical(regime)
    n_days = min(req.horizon_days, len(starts))
    n_steps = starts[n_days] if n_days < len(starts) else len(df)
    day_starts = starts[:n_days]
    v = df["V_smooth"].to_numpy(float)[:n_steps]
    r = np.diff(np.log(v))
    mask = np.ones(len(r), bool)
    for s in day_starts[1:]:
        mask[s - 1] = False
    return dict(regime=regime, fv_csv=str(REPO / G.FV_CSV[regime]),
                v0=float(df["V_smooth"].iloc[0]), n_steps=n_steps,
                theta=CALIBRATED[regime], day_starts=day_starts,
                ann_vol=round(float(np.std(r[mask])) * MIN_ANN * 100, 1), synthetic=False)


def _apply_globals(req: SimRequest, ctx: dict):
    G.CLOSEOUT_RECOVERY = float(req.recovery)
    G.IM_MODE = req.im_mode
    G.IM_FLAT_FRAC = req.flat_im_pct / 100.0
    G.IM_FLOOR = req.im_floor_pct / 100.0
    G.DF_ODD_FIXED = bool(req.fixed_df)
    G.CLIENT_MARGIN_NETTING = req.netting
    if ctx["synthetic"]:
        G.IM_DAILY = False


def _build(req: SimRequest, ctx: dict):
    params = ModelParams(
        n_fundamental=30, n_momentum=20, n_zi=40,
        n_bcm=10, n_nbcm=5, n_bcm_with_clients=5,
        v0=ctx["v0"], tick_size=0.25, dt_minutes=1.0,
        fv_csv=ctx["fv_csv"], **ctx["theta"], stressed=(ctx["regime"] == "stressed"))
    traders = build_traders(params, seed=req.seed)
    ccp = build_clearing_tier(traders, params, seed=req.seed,
                              direct=(req.structure == "direct"))
    sim = Simulation(params, traders, seed=req.seed, ccp=ccp)
    return sim, ccp, traders


def _roster(traders, ccp) -> list:
    agents = [{"id": int(ccp.ccp_id), "kind": "CCP", "cm_id": None,
               "cash0": round(float(CCP_CASH) / 1e9, 3)}]
    seen = {int(ccp.ccp_id)}
    for m in ccp.members.values():
        seen.add(int(m.agent_id))
        agents.append({"id": int(m.agent_id), "kind": KIND_SHORT.get(type(m).__name__, "CM"),
                       "cm_id": int(ccp.ccp_id), "cash0": round(float(m.cash) / 1e9, 3),
                       "n_clients": len(getattr(m, "client_ids", []) or [])})
    for t in traders:
        if int(t.agent_id) in seen or isinstance(t, BankingClearingMember):
            continue
        if isinstance(t, (FundamentalTrader, MomentumTrader, ZeroIntelligenceTrader)):
            agents.append({"id": int(t.agent_id), "kind": KIND_SHORT[type(t).__name__],
                           "cm_id": int(t.clearing_member_id) if t.clearing_member_id is not None else None,
                           "cash0": round(float(t.cash) / 1e9, 3)})
    return agents


def _agent_states(traders, ccp, cr_latest) -> list:
    states = [{"id": int(ccp.ccp_id), "cash": round(float(ccp.cash) / 1e9, 3),
               "pos": 0, "im": 0.0, "frozen": 0, "defaulted": 0, "cr": None}]
    seen = {int(ccp.ccp_id)}
    for m in ccp.members.values():
        seen.add(int(m.agent_id))
        bs = m.balance_sheet
        states.append({"id": int(m.agent_id), "cash": round(float(m.cash) / 1e9, 3),
                       "pos": int(getattr(m, "inventory", 0)),
                       "im": round(float(bs.initial_margin) / 1e9, 3),
                       "frozen": int(getattr(m, "_stopped", False)),
                       "defaulted": int(bs.has_defaulted or getattr(m, "has_defaulted", False)),
                       "cr": cr_latest.get(int(m.agent_id))})
    for t in traders:
        if int(t.agent_id) in seen or isinstance(t, BankingClearingMember) or not isinstance(
                t, (FundamentalTrader, MomentumTrader, ZeroIntelligenceTrader)):
            continue
        states.append({"id": int(t.agent_id), "cash": round(float(t.cash) / 1e9, 4),
                       "pos": int(t.inventory),
                       "im": round(float(t._posted_im) / 1e9, 4),
                       "frozen": int(t._stopped), "defaulted": int(t.has_defaulted),
                       "cr": None})
    return states


def _margin_rows(ch_rows, start_idx):
    out = {"t": [], "posted_im_usd_b": [], "margin_calls": [], "min_member_capital_ratio": []}
    ch = pd.DataFrame(ch_rows[start_idx:])
    if not len(ch):
        return out, {}
    members = ch[ch["kind"].isin(["BCM", "NBCM"])]
    by_t = ch.groupby("t")
    im_t = by_t["initial_margin"].sum()
    calls_t = by_t["call_indicator"].sum()
    kmin = members.groupby("t")["capital_ratio"].min() if len(members) else pd.Series(dtype=float)
    out["t"] = [int(t) for t in im_t.index]
    out["posted_im_usd_b"] = [round(float(x) / 1e9, 3) for x in im_t.to_numpy()]
    out["margin_calls"] = [int(x) for x in calls_t.to_numpy()]
    out["min_member_capital_ratio"] = [_clean(kmin.get(t, float("nan"))) for t in im_t.index]
    cr_latest = ({int(a): _clean(g["capital_ratio"].iloc[-1])
                  for a, g in members.groupby("agent_id")} if len(members) else {})
    return out, cr_latest


def _events(sim, cl_idx, mem_seen):
    cl_new = [{"t": int(r["t"]), "client_id": int(r["client_id"]),
               "kind": KIND_SHORT.get(str(r["kind"]), str(r["kind"])),
               "cm_id": int(r["cm_id"]),
               "shortfall_usd": round(float(r["shortfall"]), 0),
               "closeout_loss_usd": round(float(r["closeout_loss"]), 0)}
              for r in sim.client_history[cl_idx:]]
    mem_new = []
    for r in sim.clearing_history:
        if r["has_defaulted"] and r["kind"] in ("BCM", "NBCM") and r["agent_id"] not in mem_seen:
            mem_seen.add(r["agent_id"])
            mem_new.append({"t": int(r["t"]), "agent_id": int(r["agent_id"]),
                            "kind": str(r["kind"])})
    return cl_new, mem_new


def _summary(req, ctx, sim, ccp, n_client_def, member_defaults, t0):
    mid = pd.Series(sim.history["mid_price"]).ffill().bfill().to_numpy(float)
    ch = pd.DataFrame(sim.clearing_history)
    if len(ch):
        im_t = ch.groupby("t")["initial_margin"].sum()
        deepest = int(ch["waterfall_level"].max())
    else:
        im_t, deepest = pd.Series([0.0]), 0
    nbcm_def = sum(1 for m in member_defaults if m["kind"] == "NBCM")
    return {
        "regime": ctx["regime"], "fv_source": req.fv_source,
        "n_days": len(ctx["day_starts"]), "ann_vol_pct": ctx["ann_vol"],
        "recovery": req.recovery, "seed": req.seed,
        "drawdown_pct": round(100 * (float(mid.min()) / float(mid[0]) - 1), 2),
        "client_defaults": n_client_def,
        "member_defaults": len(member_defaults), "nbcm_defaults": nbcm_def,
        "deepest_waterfall": deepest, "waterfall_label": WF_LABELS[deepest],
        "mutualised": deepest >= 3,
        "im_peak_usd_b": round(float(im_t.max()) / 1e9, 2),
        "im_mean_usd_b": round(float(im_t.mean()) / 1e9, 2),
        "total_df_usd_b": round(float(ccp.total_df) / 1e9, 2),
        "ccp_cash_used_usd_b": round(float(CCP_CASH - ccp.cash) / 1e9, 3),
        "runtime_s": round(time.time() - t0, 1),
    }


def _day_lengths(ctx):
    ds, n = ctx["day_starts"], ctx["n_steps"]
    return [(ds[i + 1] if i + 1 < len(ds) else n) - ds[i] for i in range(len(ds))]


app = FastAPI(title="Client-Clearing ABM API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.exception_handler(RequestValidationError)
async def _log_422(request: Request, exc: RequestValidationError):
    body = await request.body()
    log.warning("422 %s errors=%s body=%s", request.url.path, exc.errors(), body[:500])
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


@app.get("/")
def root():
    return {"ok": True, "service": "client-clearing-abm",
            "endpoints": ["GET /meta", "POST /simulate", "POST /simulate_stream"]}


@app.get("/meta")
def meta():
    cal_vol = round(JOINT["sv_override"]["sigma_d"] * MIN_ANN * 100, 1)
    return {
        "defaults": {"regime": "stressed", "fv_source": "empirical",
                     "ann_vol_pct": cal_vol, "recovery": 0.80,
                     "horizon_days": 20, "seed": 42},
        "ranges": {"ann_vol_pct": [10, 90], "recovery": [0.30, 1.00],
                   "horizon_days": [3, 75], "seed": [0, 10_000]},
        "calibrated": {"synthetic_ann_vol_pct": cal_vol,
                       "joint_loss_D": round(JOINT["loss_D"], 2),
                       "empirical_days": {"calm": len(_fv_empirical("calm")[1]),
                                          "stressed": len(_fv_empirical("stressed")[1])}},
        "notes": {"synthetic": "stressed regime only (thesis §4.5 joint calibration)",
                  "ann_vol_pct": "applies to synthetic FV only; empirical reports realised vol"},
    }


@app.post("/simulate")
def simulate(req: SimRequest):
    ctx = _prepare(req)
    t0 = time.time()
    if not _LOCK.acquire(timeout=25):
        raise HTTPException(409, "another simulation is still running — retry in a few seconds")
    try:
        for k, val in _G_DEFAULTS.items():
            setattr(G, k, val)
        _apply_globals(req, ctx)
        sim, ccp, traders = _build(req, ctx)
        sim.run(n_steps=ctx["n_steps"])
    finally:
        for k, val in _G_DEFAULTS.items():
            setattr(G, k, val)
        _LOCK.release()

    mid = pd.Series(sim.history["mid_price"]).ffill().bfill().to_numpy(float)
    fund = np.asarray(sim.history["fundamental"], float)
    stride = max(1, len(mid) // 3000)
    idx = np.arange(0, len(mid), stride)
    margin, cr_latest = _margin_rows(sim.clearing_history, 0)
    mem_seen: set = set()
    cl_events, mem_events = _events(sim, 0, mem_seen)
    summary = _summary(req, ctx, sim, ccp, len(cl_events), mem_events, t0)
    topology = _agent_states(traders, ccp, cr_latest)
    return _json_safe({
        "summary": summary,
        "n_days": summary["n_days"],
        "regime": ctx["regime"],
        "ann_vol_pct": ctx["ann_vol"],
        "day_starts": [int(s) for s in ctx["day_starts"]],
        "series": {"t": [int(i) for i in idx],
                   "mid": [_clean(x) for x in mid[idx]],
                   "fundamental": [_clean(x) for x in fund[idx]],
                   "day_starts": [int(s) for s in ctx["day_starts"]]},
        "margin_cycles": margin,
        "client_defaults": cl_events,
        "member_defaults": mem_events,
        "events": {"client_defaults": cl_events, "member_defaults": mem_events},
        "agents": _roster(traders, ccp),
        "agent_states_by_day": [topology],
        "topology_final": topology,
    })


@app.post("/simulate_stream")
def simulate_stream(req: SimRequest):
    ctx = _prepare(req)
    if not _LOCK.acquire(timeout=25):
        raise HTTPException(409, "another simulation is still running — retry in a few seconds")

    def gen():
        t0 = time.time()
        try:
                for k, val in _G_DEFAULTS.items():
                    setattr(G, k, val)
                _apply_globals(req, ctx)
                sim, ccp, traders = _build(req, ctx)
                yield _dumps({
                    "type": "init", "n_days": len(ctx["day_starts"]),
                    "regime": ctx["regime"], "fv_source": req.fv_source,
                    "ann_vol_pct": ctx["ann_vol"], "recovery": req.recovery,
                    "seed": req.seed, "day_starts": [int(s) for s in ctx["day_starts"]],
                    "agents": _roster(traders, ccp)}) + "\n"

                step0 = 0
                mem_seen: set = set()
                for day, dlen in enumerate(_day_lengths(ctx)):
                    ch_idx, cl_idx = len(sim.clearing_history), len(sim.client_history)
                    sim.run(n_steps=dlen)
                    mid = pd.Series(sim.history["mid_price"][step0:step0 + dlen]).ffill().bfill().to_numpy(float)
                    fund = np.asarray(sim.history["fundamental"][step0:step0 + dlen], float)
                    stride = max(1, dlen // 65)
                    idx = np.arange(0, dlen, stride)
                    margin, cr_latest = _margin_rows(sim.clearing_history, ch_idx)
                    cl_new, mem_new = _events(sim, cl_idx, mem_seen)
                    yield _dumps({
                        "type": "day", "day": day,
                        "series": {"t": [int(step0 + i) for i in idx],
                                   "mid": [_clean(x) for x in mid[idx]],
                                   "fundamental": [_clean(x) for x in fund[idx]]},
                        "margin_cycles": margin,
                        "events": {"client_defaults": cl_new, "member_defaults": mem_new},
                        "agent_states": _agent_states(traders, ccp, cr_latest)}) + "\n"
                    step0 += dlen

                dedup: dict = {}
                for r in sim.clearing_history:
                    if r["has_defaulted"] and r["kind"] in ("BCM", "NBCM") and r["agent_id"] not in dedup:
                        dedup[r["agent_id"]] = {"t": int(r["t"]), "agent_id": int(r["agent_id"]),
                                                "kind": str(r["kind"])}
                yield _dumps({
                    "type": "done",
                    "summary": _summary(req, ctx, sim, ccp, len(sim.client_history),
                                        list(dedup.values()), t0)}) + "\n"
        except GeneratorExit:
            raise
        except Exception as e:
            log.exception("stream failed")
            yield _dumps({"type": "error", "detail": f"{type(e).__name__}: {e}"}) + "\n"
        finally:
                for k, val in _G_DEFAULTS.items():
                    setattr(G, k, val)
                _LOCK.release()

    return StreamingResponse(gen(), media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
