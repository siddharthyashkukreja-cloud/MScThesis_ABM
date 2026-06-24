import os, sys
# self-bootstrap: agent_context/ is one level under the repo root
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT); os.chdir(_ROOT)
import numpy as np, pandas as pd
from model import globals as G

"""Re-derive every H1/H2/NET table + QA checks in agent_context/RESULTS_METHODOLOGY.md from the
committed run output/thesis_final/experiments/rows.csv and the live model.globals. Read-only."""

R = pd.read_csv("output/thesis_final/experiments/rows.csv")
B = 1e9
def fmtB(x): return f"${x/B:.1f}B"
def stats(s):
    s = pd.Series(s, dtype=float); return s.mean(), s.std(ddof=1)
def wf_freq(a): return {k: float((a.deepest_waterfall >= k).mean()) for k in (1, 2, 3, 4, 5)}

print("############ A. QA / SANITY ############")
print("rows:", len(R), "| exp counts:", dict(R.exp.value_counts()), "| seeds", R.seed.min(), "-", R.seed.max())
for reg in ("calm", "stressed"):
    d = R[R.regime == reg].drawdown
    print(f"  drawdown[{reg}]: mean={d.mean():.4f} range {d.min():.4f}..{d.max():.4f} sd={d.std(ddof=1):.3f}")
for tag, exps, grp, summ in [("h1", ["H1"], ["regime", "direct"], "summary_h1.csv"),
                             ("h2", ["H2", "H2_decoupleDF"], ["regime", "margin", "decouple"], "summary_h2.csv"),
                             ("net", ["NET"], ["regime", "netting"], "summary_net.csv")]:
    obs = ["drawdown", "client_defaults", "cm_defaults", "deepest_waterfall", "im_mean", "im_peak",
           "total_df", "ccp_cash_used", "client_loss_absorbed"]
    mine = R[R.exp.isin(exps)].groupby(grp)[obs].mean().reset_index()
    comm = pd.read_csv(f"output/thesis_final/experiments/{summ}")
    common = [c for c in obs if c in comm.columns]
    m = mine.merge(comm, on=grp, suffixes=("_a", "_b"))
    md = max(float((m[f"{c}_a"] - m[f"{c}_b"]).abs().max()) for c in common)
    print(f"  rows-vs-{summ}: max abs diff = {md:.2e}  ({'reconciles to rounding' if md < 1e-5 else 'CHECK'})")

print("\n############ B. H1 ############")
for reg in ("calm", "stressed"):
    print(f"--- {reg} ---")
    for d, lbl in [(False, "TIERED"), (True, "DIRECT")]:
        a = R[(R.exp == "H1") & (R.regime == reg) & (R.direct == d)]
        wf = wf_freq(a); dm, ds = stats(a.deepest_waterfall)
        cd, cds = stats(a.client_defaults); cmd, cmds = stats(a.cm_defaults)
        print(f"  {lbl:6s} depth {dm:.2f}±{ds:.2f}(max {int(a.deepest_waterfall.max())}) L>=1 {wf[1]:.0%} L>=3 {wf[3]:.0%} L>=4 {wf[4]:.0%} "
              f"| cl {cd:.2f}±{cds:.2f} cm {cmd:.2f}±{cmds:.2f} | IM {fmtB(a.im_mean.mean())}/{fmtB(a.im_peak.mean())} "
              f"DF {fmtB(a.total_df.mean())} CCP {a.ccp_cash_used.mean()/1e6:.1f}M mut {fmtB(a.client_loss_absorbed.mean())}")

print("\n############ C. H2 ############")
for reg in ("stressed", "calm"):
    print(f"--- {reg} (tiered, gross) ---")
    for m in ["flat05", "reactive", "flat12", "static"]:
        a = R[(R.exp == "H2") & (R.regime == reg) & (R.margin == m)]
        cd, cds = stats(a.client_defaults)
        print(f"  {m:9s} cl {cd:.2f}±{cds:.2f} IM {fmtB(a.im_mean.mean())}/{fmtB(a.im_peak.mean())} DF {fmtB(a.total_df.mean())} depth {a.deepest_waterfall.mean():.2f}")
    a = R[(R.exp == "H2_decoupleDF") & (R.regime == reg)]
    cd, cds = stats(a.client_defaults)
    print(f"  {'rx+decpl':9s} cl {cd:.2f}±{cds:.2f} IM {fmtB(a.im_mean.mean())} DF {fmtB(a.total_df.mean())}")

print("\n############ D. NET ############")
for reg in ("calm", "stressed"):
    g = R[(R.exp == "NET") & (R.regime == reg) & (R.netting == "gross")]
    n = R[(R.exp == "NET") & (R.regime == reg) & (R.netting == "net")]
    print(f"  {reg}: gross {fmtB(g.im_mean.mean())} net {fmtB(n.im_mean.mean())} ratio {n.im_mean.mean()/g.im_mean.mean():.3f}x "
          f"peak {n.im_peak.mean()/g.im_peak.mean():.3f}x cl {g.client_defaults.mean():.2f}->{n.client_defaults.mean():.2f}")

print("\n############ E. H2 IM-FRACTION PATH ############")
FV = {r: pd.read_csv(f"data/fv_{r}.csv") for r in ("calm", "stressed")}
ds = G.daily_sigma_series(); ds.index = pd.DatetimeIndex(ds.index)
if ds.index.tz is not None: ds.index = ds.index.tz_localize(None)
ds = pd.Series(ds.to_numpy(), index=ds.index.normalize())
coef = G.IM_CONF_Z * np.sqrt(G.IM_MPOR_DAYS)
_CAP = np.inf if G.IM_CAP is None else G.IM_CAP   # IM_CAP=None (D77) -> uncapped
def band(reg, s, e, label):
    ts = pd.DatetimeIndex(pd.to_datetime(FV[reg]["ts"]))
    if ts.tz is not None: ts = ts.tz_localize(None)
    dates = ts.normalize()[s:e]
    sd = ds.reindex(dates, method="ffill").to_numpy(float)
    frac = np.minimum(_CAP, np.maximum(G.IM_FLOOR, coef * sd))
    per = pd.Series(frac, index=dates.to_numpy()).groupby(level=0).first()
    cap = float((per >= _CAP - 1e-9).mean()); flo = float((per <= G.IM_FLOOR + 1e-9).mean())
    print(f"  {reg:8s} {label:20s} sess={len(per):2d} min {per.min()*100:5.2f}% med {per.median()*100:5.2f}% "
          f"max {per.max()*100:5.2f}% pk/fl {per.max()/per.min():.2f}x cap {cap:.0%} floor {flo:.0%}")
for reg in ("calm", "stressed"):
    st = list(G.day_start_steps(reg))
    band(reg, 0, len(FV[reg]), "FULL series")
    if reg == "stressed":
        band(reg, st[10], st[30], "expt win 10-30")
    else:
        band(reg, 0, st[20], "expt win 0-20")
