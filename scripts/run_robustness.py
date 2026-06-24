#!/usr/bin/env python3
"""
scripts/run_robustness.py — clearing-side robustness experiments that COMPLETE the thesis
results, complementing the core descriptive / H1 / netting runs (run_thesis_experiments.py)
and the two-factor synthetic ensemble (overnight_2f.py). Three stages, each checkpointed to
output/robustness/ after every run so an interruption never loses work:

  SEVERITY   Scale the stressed fundamental's log-returns by k in {1.0 .. 2.0} (and sigma_t
             likewise) and run the realistic REACTIVE margin. Traces where the member tier is
             exhausted — client/member defaults, mutualisation, deepest waterfall vs the
             realised drawdown. This maps the severity the deep dive flagged as UNMAPPED
             (member failures are rare at the single COVID drawdown).

  CLOSEOUT   Sweep CLOSEOUT_RECOVERY in {0.70, 0.80, 0.90, 0.95} for the flat-8 / flat-12 arms
             (the arms where members fail) to BOUND how much "more margin -> more member
             failure" depends on this uncalibrated close-out knob (deep-dive recommendation).

  MECHANISM  Re-run flat-4 / flat-8 / flat-12 logging every client default's assumed position
             size; pool the distribution per arm (median / p90 / p99 / max / tail-fraction) to
             QUANTIFY the tail-concentration (freeze-trap) mechanism across all seeds, not just
             the two traced in the deep dive.

Reuses run_thesis_experiments.run_scenario unchanged (same clearing model, ODD fixed fund),
so the numbers are directly comparable to the H1 table. Pure numpy/pandas.

Overnight:
    SEV_SEEDS=20 CO_SEEDS=20 MECH_SEEDS=40 nohup python scripts/run_robustness.py \
        > output/robustness.out 2>&1 &
Smoke (seconds):
    python scripts/run_robustness.py --smoke
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
os.chdir(REPO)
sys.path.insert(0, str(REPO))

import model.globals as G                              # noqa: E402
from model.simulation import Simulation                # noqa: E402
import scripts.run_thesis_experiments as RTE           # noqa: E402

OUT = REPO / "output" / "robustness"
OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / "run.log"


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def scale_path(df, k):
    """Amplify the stressed path's severity: scale intraday log-returns by k (anchored at the
    first level) and sigma_t by k. k=1.0 is the identity (the realised COVID path)."""
    V = df["V_smooth"].to_numpy(float)
    logv = np.log(V)
    r = np.diff(logv)
    logk = np.concatenate([[logv[0]], logv[0] + np.cumsum(k * r)])
    out = df.copy()
    out["V_smooth"] = np.exp(logk)
    out["sigma_t"] = df["sigma_t"].to_numpy(float) * k
    return out


# ---------------------------------------------------------------------------
# STAGE 1 : severity sweep (reactive margin, where is the member tier exhausted?)
# ---------------------------------------------------------------------------
def severity(seeds, levels, deadline):
    base = RTE._FV["stressed"].copy()
    rows = []
    log(f"SEVERITY: reactive margin, k={levels}, {len(seeds)} seeds/level")
    try:
        for k in levels:
            RTE._FV["stressed"] = scale_path(base, k)
            for sd in seeds:
                r = RTE.run_scenario("stressed", sd, margin="reactive")
                r["severity"] = k
                rows.append(r)
                pd.DataFrame(rows).to_csv(OUT / "severity.csv", index=False)
                if time.time() > deadline:
                    log("  severity: hit deadline, stopping"); raise TimeoutError
            s = pd.DataFrame([x for x in rows if x["severity"] == k])
            log(f"  k={k:.2f}: drawdown {s.drawdown.mean():6.1f}%  client_def {s.client_defaults.mean():5.1f}"
                f"  member_def {s.cm_defaults.mean():.2f}  reach-fund {100*(s.deepest_waterfall>=3).mean():4.0f}%"
                f"  IMpeak {s.im_peak.mean()/1e9:.0f}B")
    except TimeoutError:
        pass
    finally:
        RTE._FV["stressed"] = base
    df = pd.DataFrame(rows)
    if len(df):
        g = df.groupby("severity").agg(
            drawdown=("drawdown", "mean"), client_def=("client_defaults", "mean"),
            member_def=("cm_defaults", "mean"),
            reach_fund=("deepest_waterfall", lambda x: float((x >= 3).mean())),
            IM_peak_B=("im_peak", lambda x: x.mean() / 1e9)).round(2).reset_index()
        g.to_csv(OUT / "severity_summary.csv", index=False)
        log("SEVERITY done:\n" + g.to_string(index=False))
        return g
    return None


# ---------------------------------------------------------------------------
# STAGE 2 : close-out recovery sweep (does the member-default result survive?)
# ---------------------------------------------------------------------------
def closeout(seeds, levels, arms, deadline):
    saved = G.CLOSEOUT_RECOVERY
    rows = []
    log(f"CLOSEOUT: recovery={levels}, arms={arms}, {len(seeds)} seeds/cell")
    try:
        for rec in levels:
            G.CLOSEOUT_RECOVERY = rec
            for arm in arms:
                for sd in seeds:
                    r = RTE.run_scenario("stressed", sd, margin=arm)
                    r["closeout_recovery"] = rec
                    rows.append(r)
                    pd.DataFrame(rows).to_csv(OUT / "closeout.csv", index=False)
                    if time.time() > deadline:
                        log("  closeout: hit deadline, stopping"); raise TimeoutError
                s = pd.DataFrame([x for x in rows if x["closeout_recovery"] == rec and x["margin"] == arm])
                log(f"  rec={rec:.2f} {arm}: member_def {s.cm_defaults.mean():.2f}"
                    f"  reach-fund {100*(s.deepest_waterfall>=3).mean():4.0f}%  client_def {s.client_defaults.mean():.1f}")
    except TimeoutError:
        pass
    finally:
        G.CLOSEOUT_RECOVERY = saved
    df = pd.DataFrame(rows)
    if len(df):
        g = df.groupby(["closeout_recovery", "margin"]).agg(
            member_def=("cm_defaults", "mean"),
            reach_fund=("deepest_waterfall", lambda x: float((x >= 3).mean())),
            client_def=("client_defaults", "mean")).round(2).reset_index()
        g.to_csv(OUT / "closeout_summary.csv", index=False)
        log("CLOSEOUT done:\n" + g.to_string(index=False))
        return g
    return None


# ---------------------------------------------------------------------------
# STAGE 3 : tail-concentration mechanism (position-at-default distribution)
# ---------------------------------------------------------------------------
def mechanism(seeds, arms, deadline):
    stash = {}
    orig = Simulation.run

    def patched(self, n, *a, **k):
        r = orig(self, n, *a, **k)
        stash["sim"] = self
        return r
    Simulation.run = patched
    rows, raw = [], []
    log(f"MECHANISM: position-at-default, arms={arms}, {len(seeds)} seeds/arm")
    try:
        for arm in arms:
            pool = []
            for sd in seeds:
                RTE.run_scenario("stressed", sd, margin=arm)
                cl = pd.DataFrame(stash["sim"].client_history)
                if len(cl) and "assumed_pos" in cl:
                    ap = cl["assumed_pos"].abs().tolist()
                    pool += ap
                    raw += [{"margin": arm, "seed": sd, "assumed_pos": v} for v in ap]
                if time.time() > deadline:
                    log("  mechanism: hit deadline, stopping"); raise TimeoutError
            a = np.array(pool, float)
            if len(a):
                rows.append(dict(margin=arm, n_defaults=len(a),
                                 median=round(float(np.median(a)), 0),
                                 p90=round(float(np.quantile(a, .90)), 0),
                                 p99=round(float(np.quantile(a, .99)), 0),
                                 max=round(float(a.max()), 0),
                                 tail_gt1800=round(float((a > 1800).mean()), 3)))
                pd.DataFrame(rows).to_csv(OUT / "mechanism_summary.csv", index=False)
                pd.DataFrame(raw).to_csv(OUT / "mechanism_raw.csv", index=False)
    except TimeoutError:
        pass
    finally:
        Simulation.run = orig
    g = pd.DataFrame(rows)
    if len(g):
        log("MECHANISM done (median falls, tail grows as margin rises):\n" + g.to_string(index=False))
        return g
    return None


def write_summary(sev, clo, mech):
    L = ["# Robustness experiments — severity, close-out, mechanism\n",
         f"Generated {time.strftime('%Y-%m-%d %H:%M')}\n"]
    L.append("## Severity sweep (reactive margin)")
    L.append(sev.to_string(index=False) if sev is not None else "(not completed)")
    L.append("\n## Close-out recovery sweep")
    L.append(clo.to_string(index=False) if clo is not None else "(not completed)")
    L.append("\n## Tail-concentration mechanism (|position| at client default)")
    L.append(mech.to_string(index=False) if mech is not None else "(not completed)")
    (OUT / "SUMMARY.md").write_text("\n".join(L) + "\n")
    log("wrote SUMMARY.md")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-hours", type=float, default=4.0)
    ap.add_argument("--sev-seeds", type=int, default=int(os.environ.get("SEV_SEEDS", 20)))
    ap.add_argument("--co-seeds", type=int, default=int(os.environ.get("CO_SEEDS", 20)))
    ap.add_argument("--mech-seeds", type=int, default=int(os.environ.get("MECH_SEEDS", 40)))
    ap.add_argument("--window", default=os.environ.get("WINDOW_DAYS", ""),
                    help='restrict window, e.g. "10:30" (default full window)')
    ap.add_argument("--skip", default="", help="comma list of stages to skip: severity,closeout,mechanism")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    if a.window:
        os.environ["WINDOW_DAYS"] = a.window
    sev_levels = [1.0, 1.25, 1.5, 1.75, 2.0]
    co_levels = [0.70, 0.80, 0.90, 0.95]
    co_arms = ["flat08", "flat12"]
    mech_arms = ["flat04", "flat08", "flat12"]
    if a.smoke:
        os.environ["WINDOW_DAYS"] = "10:30"
        a.sev_seeds = a.co_seeds = 2; a.mech_seeds = 2
        sev_levels = [1.0, 1.5]; co_levels = [0.80, 0.95]; co_arms = ["flat12"]; mech_arms = ["flat04", "flat12"]

    base_seed = 42
    sev_seeds = list(range(base_seed, base_seed + a.sev_seeds))
    co_seeds = list(range(base_seed, base_seed + a.co_seeds))
    mech_seeds = list(range(base_seed, base_seed + a.mech_seeds))
    skip = {s.strip() for s in a.skip.split(",") if s.strip()}

    t0 = time.time()
    deadline = t0 + a.max_hours * 3600.0
    try:
        LOG.write_text("")
    except OSError:
        pass
    log("=" * 70)
    log(f"ROBUSTNESS RUN  window={os.environ.get('WINDOW_DAYS','full')}  "
        f"seeds sev/co/mech={a.sev_seeds}/{a.co_seeds}/{a.mech_seeds}  budget={a.max_hours}h")
    sev = clo = mech = None
    if "severity" not in skip:
        try:
            sev = severity(sev_seeds, sev_levels, deadline)
        except Exception as e:
            log(f"SEVERITY CRASHED (continuing): {e}")
    if "closeout" not in skip:
        try:
            clo = closeout(co_seeds, co_levels, co_arms, deadline)
        except Exception as e:
            log(f"CLOSEOUT CRASHED (continuing): {e}")
    if "mechanism" not in skip:
        try:
            mech = mechanism(mech_seeds, mech_arms, deadline)
        except Exception as e:
            log(f"MECHANISM CRASHED (continuing): {e}")
    write_summary(sev, clo, mech)
    log(f"ALL DONE in {(time.time()-t0)/60:.1f} min. Outputs in {OUT}")


if __name__ == "__main__":
    main()
