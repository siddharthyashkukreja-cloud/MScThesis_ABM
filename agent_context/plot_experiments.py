"""
Figures for the overnight contagion campaign — reads the ensemble rows written by
run_experiments.py and renders the H1 / H2 / E6 / open-disorder panels.

Usage:
  python3 plot_experiments.py [rows_csv] [figdir]
Defaults: rows_csv=output/experiments/rows.csv, figdir=output/experiments/figures
(env FIGDIR overrides figdir — e.g. point it at the thesis Figures/ folder).

Each panel is independent and wrapped so one empty slice can't kill the rest. Lines
are ensemble means with a p10-p90 band over seeds.
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROWS = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("output/experiments/rows.csv")
FIGDIR = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(os.environ.get(
    "FIGDIR", "output/experiments/figures"))


def _load() -> pd.DataFrame:
    df = pd.read_csv(ROWS)
    if "error" in df.columns:
        df = df[(df["error"].isna()) | (df["error"].astype(str) == "")].copy()
    return df


def _mean_band(g: pd.DataFrame, x: str, y: str):
    """(x, mean, p10, p90) over the grouping variable x."""
    a = g.groupby(x)[y].agg(["mean", lambda s: s.quantile(.1), lambda s: s.quantile(.9)])
    a.columns = ["mean", "p10", "p90"]
    a = a.sort_index()
    return a.index.to_numpy(), a["mean"].to_numpy(), a["p10"].to_numpy(), a["p90"].to_numpy()


def _line(ax, g, x, y, label, color=None):
    xs, m, lo, hi = _mean_band(g, x, y)
    ln, = ax.plot(xs, m, marker="o", ms=4, label=label, color=color)
    ax.fill_between(xs, lo, hi, alpha=0.15, color=ln.get_color())
    return ln


def _save(fig, name):
    FIGDIR.mkdir(parents=True, exist_ok=True)
    p = FIGDIR / name
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p}")


def h1_defaults(df):
    st = df[(df.regime == "stressed") & (df.im_regime == "reactive")
            & (df.reanchor == 1) & (df.scenario == "gapped")]
    funds = [f for f in ("kalman", "gbm") if f in st.fund.unique()]
    if not funds:
        return
    fig, axes = plt.subplots(1, len(funds), figsize=(6 * len(funds), 4.2), squeeze=False)
    for ax, fund in zip(axes[0], funds):
        sub = st[st.fund == fund]
        for mode, col in (("tiered", "tab:blue"), ("direct", "tab:red")):
            g = sub[sub["mode"] == mode]
            if len(g):
                _line(ax, g, "c", "client_defaults", f"{mode}", col)
        ax.set_title(f"H1 — client defaults vs amplifier ({fund})")
        ax.set_xlabel("reverse-stress multiplier c  (1 = actual COVID)")
        ax.set_ylabel("client defaults (mean, p10-p90)")
        ax.grid(alpha=.3); ax.legend()
    _save(fig, "h1_client_defaults_vs_c.png")


def h1_waterfall(df):
    st = df[(df.regime == "stressed") & (df.im_regime == "reactive")
            & (df.reanchor == 1) & (df.scenario == "gapped")]
    funds = [f for f in ("kalman", "gbm") if f in st.fund.unique()]
    if not funds:
        return
    fig, axes = plt.subplots(1, len(funds), figsize=(6 * len(funds), 4.2), squeeze=False)
    for ax, fund in zip(axes[0], funds):
        sub = st[st.fund == fund]
        for mode, col in (("tiered", "tab:blue"), ("direct", "tab:red")):
            g = sub[sub["mode"] == mode]
            if len(g):
                _line(ax, g, "c", "deepest_waterfall", mode, col)
        ax.axhline(3, ls="--", c="grey", lw=1)
        ax.text(ax.get_xlim()[0], 3.04, " L3 = pooled-DF mutualisation", fontsize=8, c="grey")
        ax.set_title(f"H1 — deepest waterfall level vs c ({fund})")
        ax.set_xlabel("multiplier c"); ax.set_ylabel("deepest waterfall level")
        ax.grid(alpha=.3); ax.legend()
    _save(fig, "h1_waterfall_vs_c.png")


def _breach_c(df, by_cols):
    recs = []
    for keys, g in df.groupby(by_cols + ["seed"]):
        keys = keys if isinstance(keys, tuple) else (keys,)
        hit = g[g["mutualised_l3"] == 1]["c"]
        recs.append({**dict(zip(by_cols, keys[:len(by_cols)])),
                     "breach_c": hit.min() if len(hit) else np.nan})
    return pd.DataFrame(recs)


def h1_breach_box(df):
    st = df[(df.regime == "stressed") & (df.im_regime == "reactive")
            & (df.reanchor == 1) & (df.scenario == "gapped")]
    if not len(st):
        return
    bt = _breach_c(st, ["fund", "mode"])
    groups, data = [], []
    for fund in [f for f in ("kalman", "gbm") if f in bt.fund.unique()]:
        for mode in ("direct", "tiered"):
            bc = bt[(bt.fund == fund) & (bt["mode"] == mode)]["breach_c"].dropna()
            groups.append(f"{fund}\n{mode}"); data.append(bc.to_numpy() if len(bc) else np.array([np.nan]))
    fig, ax = plt.subplots(figsize=(1.6 * len(groups) + 2, 4.2))
    ax.boxplot([d[~np.isnan(d)] if np.isfinite(d).any() else [np.nan] for d in data],
               showmeans=True)
    ax.set_xticks(range(1, len(groups) + 1)); ax.set_xticklabels(groups)
    ax.set_title("H1 — pooled-DF (L3) breach multiplier by mode")
    ax.set_ylabel("smallest c reaching L3 (per seed)")
    ax.grid(alpha=.3, axis="y")
    _save(fig, "h1_breach_multiplier_box.png")


def h2_defaults(df):
    st = df[(df.regime == "stressed") & (df["mode"] == "tiered")
            & (df.reanchor == 1) & (df.scenario == "gapped") & (df.fund == "kalman")]
    if not len(st):
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
    for reg in ("reactive", "static", "flat12", "flat05"):
        g = st[st.im_regime == reg]
        if len(g):
            _line(ax1, g, "c", "client_defaults", reg)
            _line(ax2, g, "c", "deepest_waterfall", reg)
    ax1.set_title("H2 — client defaults vs c by margin regime (Kalman, tiered)")
    ax1.set_xlabel("multiplier c"); ax1.set_ylabel("client defaults"); ax1.grid(alpha=.3); ax1.legend()
    ax2.set_title("H2 — deepest waterfall vs c by margin regime")
    ax2.axhline(3, ls="--", c="grey", lw=1)
    ax2.set_xlabel("multiplier c"); ax2.set_ylabel("deepest waterfall"); ax2.grid(alpha=.3); ax2.legend()
    _save(fig, "h2_margin_regimes_vs_c.png")


def h2_calmcost(df):
    calm = df[df.regime == "calm"]
    if not len(calm):
        return
    a = calm.groupby("im_regime")["im_mean"].mean().reindex(
        ["flat05", "reactive", "static", "flat12"]).dropna()
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.bar(a.index, a.to_numpy() / 1e9, color=["tab:green", "tab:blue", "tab:cyan", "tab:red"][:len(a)])
    ax.set_title("H2 — calm-period collateral cost (mean posted IM)")
    ax.set_ylabel("mean IM in calm ($B)"); ax.set_xlabel("margin regime")
    ax.grid(alpha=.3, axis="y")
    _save(fig, "h2_calm_collateral_cost.png")


def e6(df):
    st = df[(df.regime == "stressed") & (df.fund == "kalman") & (df["mode"] == "tiered")
            & (df.im_regime == "reactive") & (df.reanchor == 1)]
    st = st[st.scenario.isin(["gapped", "shock"])]
    if not st.scenario.nunique() == 2:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for kind, col in (("gapped", "tab:blue"), ("shock", "tab:orange")):
        g = st[st.scenario == kind]
        if len(g):
            _line(ax, g, "c", "client_defaults", kind, col)
    ax.set_title("E6 — overnight gaps vs single shock (matched drawdown)")
    ax.set_xlabel("multiplier c"); ax.set_ylabel("client defaults"); ax.grid(alpha=.3); ax.legend()
    _save(fig, "e6_gaps_vs_shock.png")


def opendisorder(df):
    st = df[(df.regime == "stressed") & (df.fund == "kalman") & (df["mode"] == "tiered")
            & (df.im_regime == "reactive") & (df.scenario == "gapped")]
    if st.reanchor.nunique() < 2:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for ra, lab, col in ((1, "clean reprice (default)", "tab:blue"),
                         (0, "disorderly open (no re-anchor)", "tab:purple")):
        g = st[st.reanchor == ra]
        if len(g):
            _line(ax, g, "c", "client_defaults", lab, col)
    ax.set_title("Open-disorder sensitivity — client defaults vs c")
    ax.set_xlabel("multiplier c"); ax.set_ylabel("client defaults"); ax.grid(alpha=.3); ax.legend()
    _save(fig, "opendisorder_vs_c.png")


def main():
    if not ROWS.exists():
        print(f"no rows at {ROWS} — run the experiments first"); return
    df = _load()
    print(f"[plot] {len(df)} rows from {ROWS} -> {FIGDIR}/")
    for fn in (h1_defaults, h1_waterfall, h1_breach_box, h2_defaults,
               h2_calmcost, e6, opendisorder):
        try:
            fn(df)
        except Exception as e:
            print(f"  SKIP {fn.__name__}: {type(e).__name__}: {e}")
    print("[plot] done")


if __name__ == "__main__":
    main()
