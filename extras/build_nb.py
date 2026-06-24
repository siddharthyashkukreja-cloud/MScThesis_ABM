"""Generate model_design.ipynb — the presentation notebook for the two-tier model
(calibrated LOB market layer + regulation-grounded central-clearing tier).
Markdown narrative + figure cells; rebuild with `python3 build_nb.py`, then
"Run All" in Jupyter to regenerate every figure from the current model."""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

OUT = "model_design.ipynb"
cells = []
def md(s): cells.append(new_markdown_cell(s))
def code(s): cells.append(new_code_cell(s))

# ── Title, question, roadmap ─────────────────────────────────────────────────
md("""# An Agent-Based Model of Central Clearing
**MSc thesis · client clearing, contagion and CCP systemic risk · ES E-mini S&P 500 futures**

### The question
A central counterparty (CCP) stands between buyers and sellers so that one trader's default
doesn't directly hit its counterparties. But most end-users don't face the CCP directly —
they **clear through a clearing member (CM)**, a bank or broker that guarantees their trades.
This tiering is the focus: *does clearing clients through CMs change how — and how badly — a
market stress propagates?* When a client defaults, does the loss stay with its CM, or cascade
to the CCP and the surviving members (mutualisation)? And what governs which?

### The approach — a two-tier model
1. **Market-microstructure layer** (Part A): a limit order book where **fundamental** (value),
   **momentum** (trend) and **zero-intelligence** (noise) traders trade ES futures around an
   exogenous, real-data fundamental. It is *calibrated* so the simulated price reproduces the
   empirical ES stylised facts — the environment the clearing layer acts in is earned, not assumed.
2. **Central-clearing layer** (Part B), on top: a CCP, **banking CMs** (which also trade their
   own book) and **non-banking CMs** (pure intermediaries), each clearing a book of clients,
   with real margin machinery — procyclical VaR initial margin, USD variation margin settled
   against filled prices, a **cash-prefunded** cover-2 default fund, a five-level loss
   waterfall, client **porting** on a member default, and Almgren–Chriss fire-sales. Losses
   flow **client → CM → CCP → surviving members**.

### Headline experiment (Part B5–B6, run on the *real* COVID crash)
At true COVID severity the cover-2 framework **contains** client defaults — absorbed by the
defaulter's own margin and its CM. A **reverse-stress** sweep (Euronext A9 §5) then amplifies
the real path to locate the **mutualisation onset** — the multiplier at which losses first
reach the pooled default fund (L3) and surviving-member cash (L4) — with the CCP solvent
throughout.

### Contents
**Part 0** — [model at a glance](#glance) · [fundamental toggle](#fvmode) · [setup](#setup)
**Part A — the market layer** — [agents](#agents) · [price formation](#price) ·
[inventory concentration](#inventory) · [stylised facts](#facts) · [calibration](#calibration)
**Part B — the clearing layer** — [mechanics](#clearing) · [topology](#topology) ·
[stressed dynamics](#stressdyn) · [defaults & waterfall](#defaults) ·
[client contagion](#clients) · [reverse stress](#reverse)
**[Summary](#summary)**

Run all cells to regenerate every figure. Figures use the **calm** 2019 regime and the
**stressed** Feb–Apr 2020 COVID window (overnight gaps retained, so it carries the true
≈−33% drawdown).""")

md("""<a id="glance"></a>
## 0.1 · Model at a glance

| | |
|---|---|
| **Asset** | E-mini S&P 500 (ES front-month), 0.25 tick, 1-min call auction; real RTH sessions ≈405 bars and variable, boundaries read from the data (D57) |
| **Regimes** | calm (2019) · stressed (COVID crash, Feb–Apr 2020) |
| **Fundamental** | exogenous Kalman-filtered real efficient price of the ES mid, overnight gaps retained (D56); synthetic SV-jump path kept as a robustness alternative (§A5) |
| **Market agents** | 30 fundamental + 10 banking-CM (FT-cast) + 20 momentum + 40 zero-intelligence = 100 on the book |
| **Clearing tier** | 1 CCP + 10 BCM + 5 NBCM; procyclical VaR margin, cash-prefunded cover-2 SLOIM default fund, deficit-based 5-level waterfall, client porting on CM default (D61) |
| **Client clearing** | 90 clients (FT/MT/ZI) clear through 10 client-carrying CMs, 5–15 per CM (skewed); a client default cascades to its CM — the contagion channel |
| **Volume / cash** | per-regime `VOLUME_LOT` matches empirical ES volume; CM/client cash from CFTC/LCH disclosures |

**Calibration is LOCKED (D65).** The behavioural θ is the grid-search optimum of the
recomposed loop (zi_mu calibrated in both regimes; evidence-tightened boxes), validated
by fresh-common-seed re-ranking, a high-resolution surrogate cross-check, and
beyond-floor probes:

| regime | `ft_sigma_c` | `zi_alpha` | `zi_delta` | `zi_mu` | `p_zi` | D (grid, 3-seed) |
|---|---|---|---|---|---|---|
| calm | 0.95 | 0.26 | 0.02 | 0.0125 | L2-measured 0.543 | 43.9 |
| stressed | 0.25 | 0.32 | 0.02 | 0.0583 | 0.18 (calibrated) | 5.8 |

(`ft_alpha = mt_alpha = 1`; `mt_lambda = 0.05` pinned per Majewski. Stressed Hill 3.03 vs
target 3.18. D is comparable only at a fixed seed count — the KS component's bootstrap
scale is sim-sized.)""")

md("""<a id="fvmode"></a>
## 0.2 · Fundamental toggle — data-derived vs simulated $V_t$

The simulator is driven by an exogenous fundamental $V_t$. The **`FV_MODE`** switch in the
setup cell selects which one — everything downstream (θ, the clearing tier, every figure)
is identical; only $V_t$ changes:

- **`"kalman"`** (default — the headline model): the **data-derived** Kalman-smoothed real ES
  mid (`data/v_kalman.py`; real ≈405-bar sessions, overnight gaps retained). $V_t$ is the
  real price with microstructure noise filtered out, so the simulated mid tracks the *actual*
  ES path.
- **`"simulated"`**: the **synthetic** Stein–Stein/Heston stochastic-volatility + Merton-jump
  path (`data/v_gbm.py`, §A5). Same *volatility regime* as the data but a **random path**, not
  the real episode — the price overlay will not track real ES. It answers a robustness
  question: *does the agent layer still reproduce the stylised facts on an independent
  fundamental?*

Flip `FV_MODE` and re-run the notebook to switch.""")

md("""<a id="setup"></a>
## 0.3 · Setup — run both regimes once

One simulation per regime (clearing tier active), shared by every figure below.""")

code("""%matplotlib inline
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import warnings; warnings.filterwarnings("ignore")

from model.globals import (ModelParams, V0, CALIBRATED, VOLUME_LOT, CONTRACT_USD,
                           CCP_CALIBRATION, SIGMA_V, im_fraction, FV_CSV)
from model.agents import (FundamentalTrader, MomentumTrader,
                          ZeroIntelligenceTrader, BankingClearingMember,
                          NonBankingClearingMember)
from model.simulation import Simulation
from run_simulation import build_traders, build_clearing_tier, BARS_PER_DAY

plt.rcParams.update({"figure.dpi": 110, "axes.grid": True,
                     "grid.alpha": 0.25, "font.size": 10})
PROC = Path("data/processed")
N_DAYS = {"calm": 30, "stressed": 29}     # stressed ~= full COVID window; calm comparable

# ── Fundamental toggle (§0.2) ────────────────────────────────────────────────
FV_MODE = "kalman"          # <-- flip to "simulated" for the synthetic SV-jump V_t
FV_FILE = {
    "kalman":    {r: FV_CSV[r]              for r in ("calm", "stressed")},
    "simulated": {r: f"data/fv_gbm_{r}.csv" for r in ("calm", "stressed")},
}
def fv_path(regime): return FV_FILE[FV_MODE][regime]

def run_regime(regime, seed=42):
    p = ModelParams(n_fundamental=30, n_momentum=20, n_zi=40,
                    n_bcm=10, n_nbcm=5, n_bcm_with_clients=5,
                    v0=V0[regime], tick_size=0.25, dt_minutes=1.0,
                    fv_csv=fv_path(regime),
                    **CALIBRATED[regime], stressed=(regime == "stressed"))
    traders = build_traders(p, seed=seed)
    ccp = build_clearing_tier(traders, p, seed=seed)
    cm0 = {t.agent_id: t.clearing_member_id for t in traders
           if t.clearing_member_id is not None}      # initial client->CM map (porting stats)
    sim = Simulation(p, traders, seed=seed, ccp=ccp)
    hist = pd.DataFrame(sim.run(BARS_PER_DAY * N_DAYS[regime]))
    clearing = pd.DataFrame(sim.clearing_history)
    return dict(params=p, traders=traders, ccp=ccp, sim=sim, hist=hist,
                clearing=clearing, cm0=cm0)

def emp_returns(regime):
    df = pd.read_csv(PROC / f"ES_front_{regime}_1m.csv", parse_dates=["ts"])
    df["date"] = df["ts"].dt.date
    parts = [np.diff(np.log(s[s > 0].values))
             for _, s in df.groupby("date")["mid"] if len(s) > 1]
    return np.concatenate(parts)

# Session opens of the ACTIVE fundamental, read from its own `ts` (D57: data-driven).
def fv_opens(regime):
    ts = pd.read_csv(fv_path(regime), usecols=["ts"], parse_dates=["ts"])["ts"]
    d = ts.dt.normalize().to_numpy()
    return [int(i) for i in np.flatnonzero(np.r_[True, d[1:] != d[:-1]])]

def real_opens(regime, n=None):
    return [d for d in fv_opens(regime) if d > 0 and (n is None or d < n)]

def day_of(regime, t):                # 0-based session index of each step t
    s = np.asarray(fv_opens(regime))
    return np.searchsorted(s, np.asarray(t), side="right") - 1

def sim_returns(hist, regime):
    m = pd.Series(hist["mid_price"]).ffill().to_numpy(); m = m[m > 0]
    r = np.diff(np.log(m))            # drop overnight (day-boundary) returns (D56/D57)
    return np.delete(r, [d - 1 for d in real_opens(regime, len(m))])

def real_mid_spliced(regime):
    # The REAL ES mid at its REAL per-day levels — overnight gaps PRESERVED (D56),
    # row-aligned with the gapped Kalman fundamental and the simulated mid; the only
    # difference from the fundamental is the RTS smoothing (XGB-Chiarella §2.5.2).
    df = pd.read_csv(PROC / f"ES_front_{regime}_1m.csv", parse_dates=["ts"])
    df["date"] = df["ts"].dt.date
    parts = [np.log(s["mid"][s["mid"] > 0].values) for _, s in df.groupby("date")]
    return np.exp(np.concatenate([p for p in parts if len(p)]))

REGIMES = ["calm", "stressed"]
SIM = {r: run_regime(r) for r in REGIMES}
EMP = {r: emp_returns(r) for r in REGIMES}
RET = {r: sim_returns(SIM[r]["hist"], r) for r in REGIMES}
REAL = {r: real_mid_spliced(r) for r in REGIMES}
for r in REGIMES:
    h = SIM[r]["hist"]; cl = SIM[r]["clearing"]
    nd = int(cl[cl.has_defaulted].agent_id.nunique()) if len(cl) else 0
    print(f"{r:9s} {len(h):6d} steps | contracts/min {h['volume'].mean()*VOLUME_LOT[r]:6.0f} "
          f"| ret std emp {EMP[r].std():.1e} vs sim {RET[r].std():.1e} | defaulted CMs {nd}")""")

# ── PART A — market layer ────────────────────────────────────────────────────
md("""---
# Part A · The market layer

<a id="agents"></a>
## A1 · Agents

All rates are per-minute Bernoulli probabilities (ODD-native; no `dt` rescaling).

- **Fundamental traders (FT, 30 + 10 BCM own-account).** Each holds a fixed belief draw
  `z ~ N(0,1)` and quotes a reservation `R = V_t + z · ft_sigma_c · σ_t · v0` — the belief
  cloud widens when local volatility `σ_t` is high. One limit order *at* `R` per step,
  replace-on-new, side `sign(R − mid)`. The persistent `z` is load-bearing: high-`z` FTs
  accumulate longs, low-`z` shorts, building the **concentrated inventories** the clearing
  layer margins against. `ft_sigma_c` is calibrated — the ablation showed it is the single
  dominant lever on the return tail.
- **Momentum traders (MT, 20).** EWMA trend `M_t = (1−λ)M_{t−1} + λ r_t` (λ = 0.05 pinned,
  as Majewski et al. fix the trend horizon externally); limit-only, geometric placement
  depth.
- **Zero-intelligence traders (ZI, 40).** Cont–Stoikov–Talreja noise floor: limit order
  w.p. `zi_alpha`, market order w.p. `zi_mu` (pinned 0.025), cancel each resting order
  w.p. `zi_delta` — the sole control on ZI order lifetime (no blanket TTL, D58).
- **Placement** (ZI + MT): `k ~ Geometric(p_zi)` ticks from the mid, `p_zi` fit from real
  MBP-10 book depth.
- **Removed by ablation:** the market maker (damped volatility and clustering) and the
  volatility trader (helps stressed, harms calm — held as a regime-specific option).""")

md("""<a id="price"></a>
## A2 · Price formation, volume and the order book

Each price panel overlays three series (after XGB-Chiarella §2.5.2): the **real ES mid**
(grey), the **fundamental $V_t$** (red — the *smoothed* real mid, so it is *less* noisy,
not more), and the **simulated mid** (blue). The simulated mid tracks $V_t$ while agent
flow adds its own microstructure noise. The intraday zooms (one real RTH session each;
the stressed panel a crash-window day) show the tracking and the simulated bid-ask bounce.
The final 2×2 set tracks **book depth**, the **bid/ask depth ratio** and **margin calls
per day** — in the stressed regime the book thins and margin pressure spikes.""")

code("""fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharex="col")
for j, r in enumerate(REGIMES):
    h = SIM[r]["hist"]; t = h["t"].to_numpy()
    rm = REAL[r][:len(t)]
    _fvlbl = ("fundamental $V_t$ (smoothed real)" if FV_MODE == "kalman"
              else "fundamental $V_t$ (synthetic SV-jump)")
    axes[0, j].plot(t[:len(rm)], rm, color="#9e9e9e", lw=2.6, alpha=0.7, zorder=1, label="real ES mid")
    axes[0, j].plot(t, h["fundamental"], color="#b00020", lw=1.0, zorder=2, label=_fvlbl)
    axes[0, j].plot(t, pd.Series(h["mid_price"]).ffill(), color="#1565c0", lw=0.8, alpha=0.85, zorder=3, label="simulated mid")
    axes[0, j].set_title(f"{r}: simulated mid vs fundamental vs real  [FV={FV_MODE}]"); axes[0, j].legend(fontsize=8)
    axes[0, j].set_ylabel("price")
    axes[1, j].plot(t, h["volume"]*VOLUME_LOT[r], color="#37474f", lw=0.5)
    axes[1, j].set_title(f"{r}: volume"); axes[1, j].set_xlabel("minute"); axes[1, j].set_ylabel("contracts/min")
plt.tight_layout(); plt.show()

# Intraday zoom — ONE real RTH session per regime (sessions are ~405 bars, D57).
ZOOM = {"calm": 2, "stressed": 11}
_fvz = ("fundamental $V_t$ (smoothed real)" if FV_MODE == "kalman"
        else "fundamental $V_t$ (synthetic SV-jump)")
fig, axes = plt.subplots(1, 2, figsize=(14, 4.2))
for ax, r in zip(axes, REGIMES):
    h = SIM[r]["hist"]; op = real_opens(r); k = min(ZOOM[r], len(op) - 2)
    day = slice(op[k], op[k + 1])
    ax.plot(h["t"][day], REAL[r][day], color="#9e9e9e", lw=1.1, alpha=0.8, zorder=1, label="real ES mid")
    ax.plot(h["t"][day], h["fundamental"][day], color="#b00020", lw=1.7, zorder=3, label=_fvz)
    ax.plot(h["t"][day], pd.Series(h["mid_price"]).ffill()[day], color="#1565c0", lw=1.0, zorder=2, label="simulated mid")
    ax.set_title(f"{r}: intraday zoom — one RTH session"); ax.set_xlabel("minute"); ax.legend(fontsize=8)
axes[0].set_ylabel("price")
plt.tight_layout(); plt.show()

# Book depth, bid/ask ratio, margin-call pressure.
fig, axes = plt.subplots(2, 2, figsize=(13, 8))
for r, c in zip(REGIMES, ["#1565c0", "#b00020"]):
    h = SIM[r]["hist"]
    tot = ((h["bid_depth"] + h["ask_depth"]) * VOLUME_LOT[r]).rolling(60, min_periods=1).mean()
    axes[0, 0].plot(h["t"], tot, color=c, lw=0.9, label=r)
axes[0, 0].set_title("total book depth (contracts, 60-min mean)")
axes[0, 0].set_ylabel("resting contracts"); axes[0, 0].legend(fontsize=8)
hs = SIM["stressed"]["hist"]; vl = VOLUME_LOT["stressed"]
axes[0, 1].plot(hs["t"], (hs["bid_depth"]*vl).rolling(60, min_periods=1).mean(), color="#2e7d32", lw=0.9, label="bid (buy)")
axes[0, 1].plot(hs["t"], (hs["ask_depth"]*vl).rolling(60, min_periods=1).mean(), color="#ef6c00", lw=0.9, label="ask (sell)")
axes[0, 1].set_title("stressed: bid vs ask depth (book thinning)")
axes[0, 1].set_ylabel("resting contracts"); axes[0, 1].legend(fontsize=8)
for r, c in zip(REGIMES, ["#1565c0", "#b00020"]):
    h = SIM[r]["hist"]
    ratio = ((h["bid_depth"] + 1) / (h["ask_depth"] + 1)).rolling(60, min_periods=1).mean()
    axes[1, 0].plot(h["t"], ratio, color=c, lw=0.9, label=r)
axes[1, 0].axhline(1.0, color="k", lw=0.7, ls="--")
axes[1, 0].set_title("bid/ask depth ratio (60-min mean)")
axes[1, 0].set_xlabel("minute"); axes[1, 0].set_ylabel("bid / ask"); axes[1, 0].legend(fontsize=8)
for r, c in zip(REGIMES, ["#1565c0", "#b00020"]):
    cl = SIM[r]["clearing"]
    if len(cl):
        calls = cl.groupby(day_of(r, cl.t.to_numpy())).call_indicator.sum()
        axes[1, 1].plot(calls.index, calls.values, color=c, marker="o", ms=3, label=r)
axes[1, 1].set_title("margin calls per day"); axes[1, 1].set_xlabel("day"); axes[1, 1].set_ylabel("# margin calls"); axes[1, 1].legend(fontsize=8)
plt.tight_layout(); plt.show()""")

md("""<a id="inventory"></a>
## A3 · Inventory concentration

Persistent heterogeneous beliefs make high-`z` fundamental traders accumulate longs and
low-`z` ones shorts — demand resolves into a spread of **concentrated inventories**, which
is what the clearing layer's margin and default mechanics act on.""")

code("""def classify(t):
    if isinstance(t, BankingClearingMember): return "BCM"
    if isinstance(t, FundamentalTrader): return "FT"
    if isinstance(t, MomentumTrader): return "MT"
    if isinstance(t, ZeroIntelligenceTrader): return "ZI"
    return "other"
fig, axes = plt.subplots(1, 2, figsize=(13, 4.3))
tr = SIM["calm"]["traders"]; inv = {}
for t in tr: inv.setdefault(classify(t), []).append(t.inventory)
types = ["FT", "BCM", "MT", "ZI"]; x = np.arange(len(types))
axes[0].bar(x-0.2, [np.sum(np.abs(inv.get(k,[0]))) for k in types], 0.4, label="gross |inv|", color="#1565c0")
axes[0].bar(x+0.2, [abs(np.sum(inv.get(k,[0]))) for k in types], 0.4, label="|net inv|", color="#ef6c00")
axes[0].set_xticks(x); axes[0].set_xticklabels(types); axes[0].legend(fontsize=8)
axes[0].set_title("calm: inventory by agent type"); axes[0].set_ylabel("contracts (model lots)")
ftbcm = inv.get("FT",[]) + inv.get("BCM",[])
axes[1].hist(ftbcm, bins=15, color="#1565c0", alpha=0.85); axes[1].axvline(0, color="k", lw=0.8)
axes[1].set_title("calm: FT/BCM inventory cross-section (concentration)")
axes[1].set_xlabel("end-of-run inventory"); axes[1].set_ylabel("count")
plt.tight_layout(); plt.show()""")

md("""<a id="facts"></a>
## A4 · Stylised facts — the calibration targets

Simulated vs empirical ES 1-minute returns, presented as in HFABM / XGB-Chiarella: the
fat-tailed return distribution and the Hill tail index, then the autocorrelations of
**returns**, **absolute returns** and **squared returns** as offset stems with the
±1.96/√N band. Returns show near-zero linear autocorrelation (close to a martingale);
absolute/squared returns show positive, slowly-decaying autocorrelation — **volatility
clustering**. (Long-horizon clustering, lags ≳30 min, is a documented structural limit of
a single momentum timescale.) Kurtosis is a *diagnostic*, not a target — Hill is the tail
measure the loss optimises, and it matches (calm ≈3.0 vs empirical ≈2.96).""")

code("""def acf(x, nlags):
    x = x - x.mean(); v = x.dot(x)
    return np.array([1.0] + [x[:-k].dot(x[k:])/v for k in range(1, nlags+1)])
def hill_curve(x, fracs=np.linspace(0.02, 0.12, 25)):
    a = np.sort(np.abs(x))[::-1]; n = len(a); out = []
    for f in fracs:
        k = max(5, int(f*n)); xi = np.mean(np.log(a[:k])) - np.log(a[k])
        out.append(1.0/xi if xi > 0 else np.nan)
    return fracs, np.array(out)
def acf_stem(ax, x, nlags, color, label, off=0.0):
    a = acf(x, nlags); lags = np.arange(nlags+1) + off
    ax.vlines(lags, 0, a, color=color, alpha=0.85, lw=1.3)
    ax.plot(lags, a, "o", color=color, ms=3, label=label)

r = "calm"; s, e = RET[r], EMP[r]; ss, es = s/s.std(), e/e.std()
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
bins = np.linspace(-8, 8, 80)
axes[0].hist(es, bins=bins, density=True, alpha=0.5, color="#b00020", label="empirical")
axes[0].hist(ss, bins=bins, density=True, alpha=0.5, color="#1565c0", label="simulated")
axes[0].plot(bins, np.exp(-bins**2/2)/np.sqrt(2*np.pi), "k--", lw=0.8, label="N(0,1)")
axes[0].set_yscale("log"); axes[0].set_title(f"({r}) return distribution (std., log-y)")
axes[0].set_xlabel("standardised return"); axes[0].legend(fontsize=8)
fe, he = hill_curve(e); fs, hs = hill_curve(s)
axes[1].plot(fe, he, color="#b00020", label="empirical"); axes[1].plot(fs, hs, color="#1565c0", label="simulated")
axes[1].set_title("Hill tail index vs tail fraction"); axes[1].set_xlabel("tail fraction k/n")
axes[1].set_ylabel("Hill index"); axes[1].legend(fontsize=8)
plt.tight_layout(); plt.show()

fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2), sharex=True)
L = 30; ci = 1.96/np.sqrt(len(s))   # 95% band at the (shorter) simulated length
series = [(e, s, "return ACF", (-0.2, 0.35)),
          (np.abs(e), np.abs(s), "|return| ACF (clustering)", (-0.05, 0.4)),
          (e**2, s**2, "squared-return ACF (clustering)", (-0.05, 0.4))]
for ax, (ev, sv, title, yl) in zip(axes, series):
    acf_stem(ax, ev, L, "#b00020", "empirical", off=-0.16)
    acf_stem(ax, sv, L, "#1565c0", "simulated", off=+0.16)
    ax.axhspan(-ci, ci, color="grey", alpha=0.15); ax.axhline(0, color="k", lw=0.6)
    ax.set_title(title); ax.set_ylim(*yl); ax.set_xlabel("lag (min)")
axes[0].set_ylabel("autocorrelation"); axes[0].legend(fontsize=8)
plt.tight_layout(); plt.show()
print(f"{r}: sim excess kurtosis {((s/s.std())**4).mean()-3:.1f} (diagnostic) "
      f"| empirical {((e/e.std())**4).mean()-3:.1f}")""")

md("""<a id="calibration"></a>
## A5 · Calibration methodology, locked baseline and alternatives

**Loss.** Behavioural parameters are fit by simulated method of moments: the loss is a
**five-component standardised-moment distance**, each component divided by its empirical
block-bootstrap sampling SD (Franke & Westerhoff 2012; Künsch 1989 moving blocks), so
heterogeneous moments are commensurable:

| component | matches | source |
|---|---|---|
| KS | whole return distribution (2-sample Kolmogorov–Smirnov) | XGB-Chiarella §3.2.4 |
| V | return standard deviation | Gao et al. 2022 |
| ACF1 | autocorrelation of returns, lags {1, 5, 10, 20} | Cont 2001 |
| ACF2 | autocorrelation of \\|returns\\| (clustering), lags {1, 5, 10, 20} | Cont 2001 / 2005 |
| Hill | tail index (banded) | HFABM §4.1.1 |

KS and Hill are deliberately paired — KS pins the distribution body, Hill the tail; the
ablation showed each controls a failure mode the other cannot.

**Two solvers, one answer (D60).** An exhaustive **grid search** (Gao et al. 2023; calm
7³ = 343 nodes, stressed 5⁴ = 625, 3 seeds) is the defensible headline; a high-resolution
**surrogate-assisted SMM** (XGBoost θ→D, Sobol pool argmin, active learning, stage-2
refinement on the true simulator — Gao et al. 2022) lands on the *same* optimum. The locked
θ is in §0.1 and `globals.CALIBRATED`.

**Robustness campaign (parsimony verdict).** Eight extensions were screened per regime;
none earned its parameters. The clearest case: an apparent −34% stressed win from FT/MT
activation gates did **not replicate** at higher resolution (poorly-identified extra
parameters; high surrogate R² ≠ reproducible optimum). Only calibrating `mt_lambda` gave a
small (~1 D) reliable stressed gain — not adopted, parsimony retained.

**Fundamental alternative (`FV_MODE = "simulated"`).** A Stein–Stein/Heston SV +
Merton-jump path:

```
σ_{t+1}             = θ + e^(−α)(σ_t − θ) + s·Z_σ                  (Vasicek–OU vol)
log V_{t+1}−log V_t = (μ − ½σ_t²) + σ_t·Z + J,  J ~ N(0, δ²) w.p. λ/390   (Merton jumps)
```

The ablation showed the Kalman efficient price fits better in both regimes once the tail
is controlled — and is more defensible (the real path; no synthetic clustering injected).

**Agents tried and removed (ablation):** market maker (damps volatility/clustering),
volatility trader (stressed-only gain), Cont threshold trader (no durable clustering),
Pareto order sizes (degraded the tail under linear impact).""")

# ── PART B — clearing layer ──────────────────────────────────────────────────
md("""---
# Part B · The clearing layer

<a id="clearing"></a>
## B1 · Mechanics — margins, default fund, waterfall, porting

All constants are regulatory / ODD pins (none SMM-calibrated).

**Margin cycle (hourly).** Variation margin marks each cleared book to market in USD
(`× VOLUME_LOT × CONTRACT_USD`) and settles into cash — **against the average filled
price** (ODD §Margin Call: P/L = N·(P_market − P_filled), D61), so execution slippage in
fire-sale book walks is realised, not dropped. **Initial margin** is a procyclical
VaR/SPAN scan (99% / 2-day VaR, anti-procyclicality floor ≈ CME ES margin: ≈6% calm,
≈12% stressed). **Capital ratio** = cash / IM, floored at 8% (CFTC Reg 1.17 — net capital
vs *risk margin*, not gross notional): a breaching BCM deleverages via Almgren–Chriss, a
breaching NBCM **stops out** — state-based, its clients frozen while the breach lasts
(D61). A BCM's house book runs under a VaR risk limit so it cannot accumulate an unbounded
prop position.

**Default fund (cash-prefunded, D61).** Cover-2 **Stress-Loss-Over-Initial-Margin**
(Euronext A9): the two most-exposed members' loss above posted margin under an
extreme-but-plausible move, ×1.10. Members **pay their contributions in cash** into a
CCP-held pool (ODD §Step Sequence 8); a post-draw daily recompute is a **replenishment
cash call** — the margin-liquidity channel the ESRB documented in March 2020.

**Member default.** On cash exhaustion the CCP runs the five-level waterfall on the
member's realised **cash deficit**: L1 own DF → L2 exchange skin-in-the-game → L3 pooled
DF → L4 surviving-member cash pro-rata → L5 CCP. The defaulted member is **removed**
(ODD §Step Sequence 13); its surviving clients are **ported** to other client-carrying CMs
with spare capacity — receivers must stay above the 8% floor with the ported book
(EMIR Art. 48(5)-(6); porting in stress is *not* guaranteed — OFR 2026: the modal client
has a single clearing agent) — and unported clients are closed out. The CCP assumes the
own book plus unported positions and **fire-sells it on the LOB** (Almgren–Chriss; a
flagged deviation from the ODD's position auction — the open-market book walk is the
thesis's price-impact contagion channel). Disposal-period P&L **resumes the waterfall**
(D61), so the close-out is deficit-consistent end to end.

**Client default.** Each client posts its own VM from its own cash, is position-capped by
house margin (20%, 5× leverage), freezes at the 8% floor, defaults at zero — its CM covers
the uncollected margin, assumes the position and liquidates it via Almgren–Chriss. This
client → CM → CCP cascade is the thesis's central contagion mechanism.""")

md("""<a id="topology"></a>
## B2 · Agent topology (the clearing star)

The hub is the CCP + exchange; each spoke a clearing member; orange dots and node labels
are the **actual** skewed client books (5–15 per CM). Non-banking CMs hold the largest
books (all their capacity is client clearing); five of the ten BCMs are own-account only.
A bigger book = more concentrated client-default risk.""")

code("""ccp = SIM["calm"]["ccp"]
mem = sorted(ccp.members.values(),
             key=lambda m: (isinstance(m, NonBankingClearingMember), len(m.client_ids)), reverse=True)
fig, ax = plt.subplots(figsize=(9.2, 9.2)); ax.set_aspect("equal"); ax.axis("off")
ax.set_title("Clearing-tier star topology — actual skewed client books (node label = book size)", fontsize=11)
ax.scatter([0], [0], s=3600, c="#b00020", zorder=5)
ax.text(0, 0, "CCP +\\nExchange", ha="center", va="center", color="white", fontsize=8.5, weight="bold", zorder=6)
for k, cm in enumerate(mem):
    ang = 2*np.pi*k/len(mem) + np.pi/2
    x, y = np.cos(ang), np.sin(ang)
    ax.plot([0, x], [0, y], color="#90a4ae", lw=1.0, zorder=1)
    is_n = isinstance(cm, NonBankingClearingMember)
    col = "#2e7d32" if is_n else "#1565c0"; kind = "NBCM" if is_n else "BCM"
    n = len(cm.client_ids)
    ax.scatter([x], [y], s=540, c=col, zorder=4, edgecolors="white", linewidths=1.2)
    ax.text(x, y, f"{kind}\\n{n}", ha="center", va="center", color="white", fontsize=6.0, zorder=6)
    for c in range(n):                                   # one dot per ACTUAL client
        cang = ang + (c - (n - 1) / 2) * (0.30 / max(n, 1))
        cx, cy = 1.55*np.cos(cang), 1.55*np.sin(cang)
        ax.plot([x, cx], [y, cy], color="#cfd8dc", lw=0.4, zorder=1)
        ax.scatter([cx], [cy], s=16, c="#ef6c00", zorder=3)
from matplotlib.lines import Line2D
leg = [Line2D([0],[0], marker="o", color="w", markerfacecolor=c, markersize=11, label=l)
       for c, l in [("#b00020","CCP + Exchange (LOB)"), ("#1565c0","Banking CM (own + client)"),
                    ("#2e7d32","Non-banking CM (clients only)"), ("#ef6c00","Clients — node # = book size (5–15)")]]
ax.legend(handles=leg, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=2, frameon=False)
ax.set_xlim(-1.95, 1.95); ax.set_ylim(-1.95, 1.95)
plt.tight_layout(); plt.show()""")

md("""<a id="stressdyn"></a>
## B3 · Clearing members under stress — capital, cash, variation margin

Each line is one clearing member over the COVID window: capital ratio vs the 8% floor,
the cash trajectory (now including DF funding outflows and any L3/L4 draws), and
cumulative variation margin. Crosses mark defaults.""")

code("""r = "stressed"; cl = SIM[r]["clearing"]; clm = cl[cl.kind != "CCP"]
deftimes = clm[clm.has_defaulted].groupby("agent_id").t.min().to_dict()
fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))
for aid, g in clm.groupby("agent_id"):
    col = "#1565c0" if g["kind"].iloc[0] == "BCM" else "#2e7d32"
    axes[0].plot(g.t, g.capital_ratio.clip(-0.1, 1.0), color=col, lw=0.8, alpha=0.7)
    axes[1].plot(g.t, g.cash/1e9, color=col, lw=0.8, alpha=0.7)
    axes[2].plot(g.t, g.vm_cumulative/1e9, color=col, lw=0.8, alpha=0.7)
    if aid in deftimes:
        td = deftimes[aid]; row = g[g.t == td]
        axes[0].plot(td, float(row.capital_ratio.clip(-0.1,1.0).iloc[0]), "x", color="#b00020", ms=8, mew=2)
axes[0].axhline(CCP_CALIBRATION["cap_ratio_floor"], color="#b00020", ls="--", lw=1)
axes[0].set_title("capital ratio / CM (cash / IM, CFTC Reg 1.17)"); axes[0].set_ylabel("cash / initial margin")
axes[1].axhline(0, color="k", lw=0.7); axes[1].set_title("cash / CM"); axes[1].set_ylabel("$bn")
axes[2].set_title("cumulative variation margin / CM"); axes[2].set_ylabel("$bn")
for a in axes: a.set_xlabel("minute")
from matplotlib.lines import Line2D
axes[0].legend(handles=[Line2D([0],[0],color="#1565c0",label="BCM"),
                        Line2D([0],[0],color="#2e7d32",label="NBCM"),
                        Line2D([0],[0],color="#b00020",ls="--",label="8% floor"),
                        Line2D([0],[0],marker="x",color="#b00020",ls="",label="default")], fontsize=7)
plt.tight_layout(); plt.show()""")

md("""<a id="defaults"></a>
## B4 · Defaults and the waterfall

Per defaulted member: when it failed, its state at default, its book size, and the deepest
waterfall level its **cash deficit** reached. The waterfall histogram counts every
waterfall event — including **CCP disposal-account rows** (D61: losses while the CCP
fire-sells an assumed defaulted book resume the originating waterfall).""")

code("""for r in REGIMES:
    cl = SIM[r]["clearing"]; ccp = SIM[r]["ccp"]
    clm = cl[cl.kind != "CCP"] if len(cl) else cl
    ev = clm[clm.has_defaulted].sort_values("t").groupby("agent_id").first().reset_index()
    print(f"================  {r.upper()}  ================")
    nbcm = int((ev.kind=='NBCM').sum()); nbk = int((ev.kind=='BCM').sum())
    print(f"defaults: {len(ev)}  ({nbk} BCM, {nbcm} NBCM)   "
          f"| cover-2 total_DF ${ccp.total_df/1e9:.2f}B  prefunded DF cash held ${ccp.df_cash/1e9:.2f}B")
    print(f"SITG end ${ccp.own_df/1e9:.2f}B | CCP cash end ${ccp.cash/1e9:.2f}B (start $7.5B) "
          f"| CCP disposal rows: {int((cl.kind=='CCP').sum()) if len(cl) else 0}")
    if len(ev):
        im = im_fraction(SIGMA_V[r])      # procyclical IM fraction → exposure = IM / im
        tbl = pd.DataFrame({
            "CM": ev.agent_id, "kind": ev.kind,
            "default_min": ev.t.astype(int),
            "default_day": day_of(r, ev.t.to_numpy()).astype(int),
            "cap_ratio@def": ev.capital_ratio.round(3),
            "own_pos_lots": ev.own_position.astype(int),
            "book_exposure_$B": (ev.initial_margin/im/1e9).round(2),
            "waterfall_lvl": ev.waterfall_level.astype(int),
        }).reset_index(drop=True)
        print(tbl.to_string(index=False))
    print()

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
for r, c in zip(REGIMES, ["#1565c0", "#b00020"]):
    cl = SIM[r]["clearing"]
    lv = cl[cl.waterfall_level>0].waterfall_level.value_counts().reindex([1,2,3,4,5], fill_value=0)
    axes[0].bar(lv.index + (0.18 if r=="stressed" else -0.18), lv.values, 0.36, color=c, label=r)
axes[0].set_xticks([1,2,3,4,5])
axes[0].set_xticklabels(["L1 own\\nDF","L2\\nSITG","L3 pooled\\nDF","L4 survivor\\ncash","L5\\nexchange"], fontsize=7)
axes[0].set_title("deepest waterfall level per event (incl. CCP disposal)"); axes[0].set_ylabel("# events"); axes[0].legend(fontsize=8)
for r, c in zip(REGIMES, ["#1565c0", "#b00020"]):
    cl = SIM[r]["clearing"]
    calls = cl.groupby(day_of(r, cl.t.to_numpy())).call_indicator.sum()
    axes[1].plot(calls.index, calls.values, color=c, marker="o", ms=3, label=r)
axes[1].set_title("margin calls per day"); axes[1].set_xlabel("day"); axes[1].set_ylabel("# calls"); axes[1].legend(fontsize=8)
plt.tight_layout(); plt.show()""")

md("""<a id="clients"></a>
## B5 · Client clearing — the client → CM contagion channel, and porting

Each client posts its own variation margin, freezes at the 8% floor, defaults on cash
exhaustion; its CM covers the uncollected margin, assumes the position and liquidates it
(the close-out slippage settles against filled prices, D61). **Left:** client defaults by
type — the thinner-capitalised accounts fail first. **Right:** the client-default loss each
CM absorbs — the cash drain that can topple the CM itself. If a CM defaults, its surviving
clients are **ported** to other client CMs with capacity (EMIR Art. 48); the printout
reports how many clients moved or were frozen.""")

code("""r = "stressed"; ch = pd.DataFrame(SIM[r]["sim"].client_history); cl = SIM[r]["clearing"]
_order = ["FundamentalTrader", "MomentumTrader", "ZeroIntelligenceTrader"]
_short = {"FundamentalTrader": "FT (asset mgr)", "MomentumTrader": "MT (CTA)", "ZeroIntelligenceTrader": "ZI (noise)"}
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
counts = [int((ch.kind == k).sum()) if len(ch) else 0 for k in _order]
axes[0].bar(range(3), counts, color=["#1565c0", "#2e7d32", "#ef6c00"])
axes[0].set_xticks(range(3)); axes[0].set_xticklabels([_short[k] for k in _order])
axes[0].set_title("stressed: client defaults by type")
axes[0].set_ylabel("# client defaults")
ab = cl[cl.kind != "CCP"].groupby("agent_id").agg(absorbed=("client_loss_absorbed", "sum"),
                                                  kind=("kind", "first")).reset_index()
ab = ab[ab.absorbed > 0].sort_values("absorbed")
axes[1].bar(range(len(ab)), ab.absorbed/1e6,
            color=["#1565c0" if k == "BCM" else "#2e7d32" for k in ab.kind])
axes[1].set_title("stressed: client-default loss absorbed per CM")
axes[1].set_ylabel("$M absorbed"); axes[1].set_xlabel("clearing member (sorted)")
plt.tight_layout(); plt.show()

nd = ch.client_id.nunique() if len(ch) else 0
tot = ch.shortfall.sum()/1e6 if len(ch) else 0.0
ported = [t for t in SIM[r]["traders"] if t.clearing_member_id is not None
          and SIM[r]["cm0"].get(t.agent_id) != t.clearing_member_id]
frozen = [t for t in SIM[r]["traders"] if t._stopped and not t.has_defaulted]
print(f"stressed: {nd} client defaults | ${tot:.0f}M uncollected-VM shortfall absorbed by CMs "
      f"(close-out slippage settles via the assumed-book VM, D61)")
print(f"porting (D61): {len(ported)} clients re-homed to surviving CMs | "
      f"{len(frozen)} clients frozen (stop-out / unported / floor)")""")

md("""<a id="reverse"></a>
## B6 · Reverse stress — locating the mutualisation onset (Euronext A9 §5)

At actual COVID severity the cover-2 framework *contains* the cleared book. The
reverse-stress test amplifies the real COVID path by a multiplier `c`
(`covid_contagion.py`) and tracks the deepest waterfall level and default counts as `c`
rises. Two mutualisation definitions are reported (both cash-real after D61):
**L3** — the prefunded pooled default fund is drawn (survivors lose prepaid cash and face
a replenishment call); **L4** — surviving-member cash is assessed directly. The onset
multiplier quantifies the system's resilience as a single number.""")

code('''from covid_contagion import run as covid_run
cs = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
rs = [covid_run(c) for c in cs]
fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
axes[0].step(cs, [r["deepest_waterfall"] for r in rs], where="mid", color="#b00020", lw=1.6, marker="o")
axes[0].axhline(3, color="#ef6c00", ls="--", lw=1, label="L3 = pooled-DF mutualisation")
axes[0].axhline(4, color="#2e7d32", ls="--", lw=1, label="L4 = survivor-cash assessment")
axes[0].set_title("deepest waterfall level vs crash amplification")
axes[0].set_xlabel("amplification c (× real COVID path)"); axes[0].set_ylabel("waterfall level")
axes[0].set_yticks([0, 1, 2, 3, 4, 5]); axes[0].legend(fontsize=8)
axes[1].plot(cs, [r["client_defaults"] for r in rs], color="#1565c0", marker="o", label="client defaults")
axes[1].plot(cs, [r["cm_defaults"] for r in rs], color="#b00020", marker="s", label="CM (member) defaults")
axes[1].set_title("defaults vs crash amplification")
axes[1].set_xlabel("amplification c (× real COVID path)"); axes[1].set_ylabel("# defaults"); axes[1].legend(fontsize=8)
plt.tight_layout(); plt.show()
b3 = next((c for c, r in zip(cs, rs) if r["deepest_waterfall"] >= 3), None)
b4 = next((c for c, r in zip(cs, rs) if r["deepest_waterfall"] >= 4), None)
print("drawdowns: " + ", ".join(f"c={c:g}:{r['drawdown']:.0f}%" for c, r in zip(cs, rs)))
print(f"mutualisation onset: L3 (pooled DF) {'c >= %g' % b3 if b3 else 'not reached'} | "
      f"L4 (survivor cash) {'c >= %g' % b4 if b4 else 'not reached'} | "
      f"CCP cash end ${rs[-1]['ccp_cash']/1e9:.2f}B (start $7.5B)")
print("NB single-seed traces — the Monte-Carlo ensemble over seeds is the planned next step.")''')

md("""<a id="summary"></a>
---
## Summary

**Part A.** The calibrated market layer reproduces the ES stylised facts under both
regimes — fat tails (calm Hill ≈3.0 vs empirical ≈2.96), near-zero return autocorrelation,
short-horizon volatility clustering — with the behavioural θ **locked (D60)** by two
cross-validating methods (grid headline, surrogate confirmation). Long-horizon clustering
remains a documented structural limit; the robustness campaign's parsimony verdict (richer
agent dynamics do not earn their parameters) is reported as a result, not a failure.

**Part B.** The clearing layer runs real, regulation-grounded machinery: procyclical
VaR/SPAN initial margin, variation margin settled against filled prices (D61), a cash/IM
capital floor (CFTC Reg 1.17), a **cash-prefunded** cover-2 SLOIM default fund with daily
replenishment calls (Euronext A9; D61), a deficit-consistent five-level waterfall that the
CCP's own disposal losses resume (D61), client **porting** on member default (EMIR
Art. 48; D61), and Almgren–Chriss deleveraging and fire-sales whose book-walk is the
contagion channel. On the real COVID window the framework **contains** the book; the
reverse-stress sweep locates the L3/L4 mutualisation onset with the CCP solvent throughout.

**Next:** Monte-Carlo ensembles over seeds (distributions of the onset multiplier, default
counts, waterfall depth, mutualised loss) and designed sweeps over CM count, client-book
concentration, porting on/off, and margin methodology.""")

nb = new_notebook(cells=cells, metadata={
    "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
    "language_info": {"name": "python"}})
with open(OUT, "w") as f:
    nbf.write(nb, f)
print("wrote", OUT, "with", len(cells), "cells")
