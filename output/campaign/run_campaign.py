"""Overnight calibration-campaign driver (autonomous).

Runs E0, E2, E3, E4a, E4b, E6, E5, E1 in the brief's cheap->expensive order. Each
calibration experiment is isolated in its own subprocess with experiment-specific env
flags (so calibrate.py's import-time bound gating is honoured), gets a fresh LHS cache
(several experiments share param keys but differ STRUCTURALLY, which the cache guard does
not catch), copies its per-regime calibrated_params.json, appends a SUMMARY.md row, and
runs a book-depth spot check. Errors are logged to SUMMARY.md and the campaign continues.

Run:  python3 output/campaign/run_campaign.py   (intended for the background)
"""
import os, sys, json, time, shutil, subprocess, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output"
CAMP = OUT / "campaign"
LOGS = CAMP / "logs"
SUMMARY = CAMP / "SUMMARY.md"
RESULTS = CAMP / "campaign_results.json"
PY = sys.executable

STD = ["48", "20", "2", "1", "12", "8"]          # standard (<=5 calibrated params)
HI = ["96", "20", "2", "1", "16", "12"]          # high-dim (E5, >=6 params)

# (eid, env-flags, args, one-line description) — campaign order E0,E2,E3,E4a,E4b,(E6),E5,E1.
CALIB = [
    ("E0",  {},                                                                STD, "baseline control (no changes)"),
    ("E2",  {"MT_LAMBDA_IN_LOOP": "1"},                                         STD, "mt_lambda in the loop (0.004-0.20)"),
    ("E3",  {"MT_LAMBDA_FIXED": "0.00385"},                                     STD, "mt_lambda pinned ~3h half-life"),
    ("E4a", {"N_MOMENTUM": "30", "MT_LAMBDA_FIXED": "0.00385"},                 STD, "30 short-lambda MT, long lambda"),
    ("E4b", {"N_MOMENTUM": "15", "N_MOMENTUM_LONG": "15", "MT_LAMBDA_LONG": "0.00385"}, STD, "two-cohort MT 15 long + 15 short"),
    ("E5",  {"FTMT_GATES": "1"},                                                HI,  "FT/MT Bernoulli gate + cancellation (high-dim)"),
    ("E1",  {"CLEARING_IN_LOOP": "1"},                                          STD, "clearing/CCP tier active in the loop"),
]
REGIMES = ("calm", "stressed")


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(CAMP / "driver.log", "a") as f:
        f.write(line + "\n")


def init_summary():
    if SUMMARY.exists() and SUMMARY.stat().st_size > 0:
        return
    with open(SUMMARY, "w") as f:
        f.write("# Calibration campaign — results summary\n\n")
        f.write("Interim screening resolution (NOT thesis-final). Columns: D = validated true "
                "loss; KS/V/ACF1/ACF2/Hill = Franke-standardised component deltas (sampling-SDs "
                "off); kurtosis = diagnostic only. key theta = stage-2 optimum. R2 = held-out "
                "D-surrogate accuracy (>~0.5 trustworthy). Each experiment's behaviour is gated "
                "behind an env flag, default OFF, so E0 is the recoverable baseline.\n\n")
        f.write("| EID | regime | D | KS | V | ACF1 | ACF2 | Hill | kurtosis | key theta | notes |\n")
        f.write("|-----|--------|---|----|---|------|------|------|----------|-----------|-------|\n")


def append_row(cells):
    with open(SUMMARY, "a") as f:
        f.write("| " + " | ".join(cells) + " |\n")


def fmt(x, p=3):
    if x is None:
        return "—"
    try:
        if x != x:
            return "nan"
        return f"{x:.{p}f}"
    except Exception:
        return str(x)


def clean_caches():
    for pat in ("calibration_lhs_*.csv", "calibration_stage2_*.csv"):
        for fp in OUT.glob(pat):
            fp.unlink()
    cp = OUT / "calibrated_params.json"
    if cp.exists():
        cp.unlink()


def depth_probe(regime, env):
    try:
        r = subprocess.run([PY, str(CAMP / "_depth_probe.py"), regime,
                            str(OUT / "calibrated_params.json")],
                           env={**os.environ, **env}, cwd=str(ROOT),
                           capture_output=True, text=True, timeout=600)
        out = r.stdout.strip().splitlines()
        return json.loads(out[-1]) if out else {}
    except Exception as e:
        return {"error": str(e)}


def run_regime(eid, regime, env, args, desc, store):
    """Run one regime of one experiment; copy JSON, log SUMMARY row, depth-probe."""
    # Resume-skip: if this (eid, regime) already completed without error in a
    # prior driver invocation (store loaded from campaign_results.json), don't
    # re-run or re-append a SUMMARY row — restarts after a crash are idempotent.
    prev = store.get(eid, {}).get(regime)
    if prev and "error" not in prev and prev.get("true_loss_validated") is not None:
        log(f"{eid} {regime}: already done (D={fmt(prev.get('true_loss_validated'),2)}) — skip")
        return
    logp = LOGS / f"{eid}_{regime}.log"
    cmd = [PY, "calibrate.py", "run", regime, *args]
    log(f"{eid} {regime}: {' '.join(cmd)}  flags={env or '{}'}")
    t0 = time.time()
    try:
        with open(logp, "w") as lf:
            r = subprocess.run(cmd, env={**os.environ, **env}, cwd=str(ROOT),
                               stdout=lf, stderr=subprocess.STDOUT, timeout=6 * 3600)
        dt = time.time() - t0
        if r.returncode != 0:
            tail = logp.read_text().splitlines()[-3:]
            log(f"{eid} {regime}: FAILED rc={r.returncode} ({dt:.0f}s)")
            append_row([eid, regime, "ERROR", "", "", "", "", "", "",
                        f"`{env}`", f"rc={r.returncode}: {' '.join(tail)[-140:]}"])
            store.setdefault(eid, {})[regime] = {"error": f"rc={r.returncode}", "flags": env}
            return
        # Copy + augment the per-regime calibrated params.
        src = OUT / "calibrated_params.json"
        dst = CAMP / f"{eid}_{regime}.json"
        shutil.copy(src, dst)
        payload = json.load(open(dst))
        payload["campaign"] = {"eid": eid, "regime": regime, "flags": env,
                               "args": args, "desc": desc}
        json.dump(payload, open(dst, "w"), indent=2)
        res = payload["results"][regime]
        cd = res.get("component_deltas", {})
        val = res.get("validated", {})
        sd = res.get("surrogate_d_accuracy", {})
        theta = res.get("theta_stage2", {})
        D = res.get("true_loss_validated")
        r2 = sd.get("r2")
        depth = depth_probe(regime, env)
        dmean = depth.get("depth_mean")
        theta_str = ", ".join(f"{k}={v:.3g}" for k, v in theta.items())
        note = f"R2={fmt(r2,2)}"
        if r2 is not None and r2 == r2 and r2 < 0.5:
            note += " (near-blind, distrust theta)"
        note += f"; depth~{fmt(dmean,0)}"
        if dmean is not None and dmean == dmean and dmean > 5000:
            note += " (HIGH)"
        if desc:
            note += f"; {desc}"
        append_row([eid, regime, fmt(D, 2), fmt(cd.get("KS")), fmt(cd.get("V")),
                    fmt(cd.get("ACF1")), fmt(cd.get("ACF2")), fmt(cd.get("Hill")),
                    fmt(val.get("ret_kurtosis"), 1), f"`{theta_str}`", note])
        store.setdefault(eid, {})[regime] = {
            "true_loss_validated": D, "component_deltas": cd,
            "ret_kurtosis": val.get("ret_kurtosis"), "hill": val.get("hill_tail_index"),
            "theta_stage2": theta, "surrogate_r2": r2, "depth": depth,
            "flags": env, "args": args, "seconds": dt}
        log(f"{eid} {regime}: D={fmt(D,2)} R2={fmt(r2,2)} depth~{fmt(dmean,0)} ({dt:.0f}s)")
    except Exception:
        tb = traceback.format_exc()
        log(f"{eid} {regime}: EXCEPTION\n{tb}")
        append_row([eid, regime, "ERROR", "", "", "", "", "", "", f"`{env}`",
                    f"exception: {tb.splitlines()[-1][:120]}"])
        store.setdefault(eid, {})[regime] = {"error": tb.splitlines()[-1], "flags": env}


def run_e6(store):
    """E6 — gaps vs single shock (contagion scenario, not a calibration)."""
    prev = store.get("E6")
    if prev and "error" not in prev and prev.get("csv"):
        log("E6: already done — skip")
        return
    log("E6: gaps vs single shock scenario sweep")
    logp = LOGS / "E6.log"
    try:
        with open(logp, "w") as lf:
            r = subprocess.run([PY, "covid_contagion.py", "scenario"], cwd=str(ROOT),
                               env=os.environ.copy(), stdout=lf, stderr=subprocess.STDOUT,
                               timeout=3 * 3600)
        if r.returncode != 0:
            tail = logp.read_text().splitlines()[-3:]
            append_row(["E6", "stressed", "—", "", "", "", "", "", "", "`gaps vs shock`",
                        f"ERROR rc={r.returncode}: {' '.join(tail)[-120:]}"])
            store["E6"] = {"error": f"rc={r.returncode}"}
            return
        import pandas as pd
        df = pd.read_csv(CAMP / "E6_gaps_vs_shock.csv")
        g = df.groupby("kind")[["client_defaults", "cm_defaults", "waterfall_level_reached"]].mean()
        note = "; ".join(f"{k}: cl_def~{g.loc[k,'client_defaults']:.1f} "
                         f"cm_def~{g.loc[k,'cm_defaults']:.1f} maxWF~{g.loc[k,'waterfall_level_reached']:.1f}"
                         for k in g.index)
        append_row(["E6", "stressed", "—", "", "", "", "", "", "", "`gaps vs shock c-sweep`",
                    f"see E6_gaps_vs_shock.csv. {note}"])
        store["E6"] = {"csv": "E6_gaps_vs_shock.csv",
                       "by_kind": json.loads(g.to_json(orient="index"))}
        log(f"E6: done. {note}")
    except Exception:
        tb = traceback.format_exc()
        log(f"E6: EXCEPTION\n{tb}")
        append_row(["E6", "stressed", "ERROR", "", "", "", "", "", "", "`gaps vs shock`",
                    f"exception: {tb.splitlines()[-1][:120]}"])
        store["E6"] = {"error": tb.splitlines()[-1]}


def main():
    LOGS.mkdir(parents=True, exist_ok=True)
    init_summary()
    store = {}
    if RESULTS.exists():
        try:
            store = json.load(open(RESULTS))
        except Exception:
            store = {}
    t_start = time.time()
    log(f"=== campaign start (budget ~8h) python={PY} ===")

    order = ["E0", "E2", "E3", "E4a", "E4b", "E6", "E5", "E1"]
    by_id = {e[0]: e for e in CALIB}
    for eid in order:
        if eid == "E6":
            run_e6(store)
            json.dump(store, open(RESULTS, "w"), indent=2)
            continue
        _, env, args, desc = by_id[eid]
        log(f"--- {eid} ({desc}) — fresh caches ---")
        clean_caches()
        for regime in REGIMES:
            run_regime(eid, regime, env, args, desc, store)
            json.dump(store, open(RESULTS, "w"), indent=2)
        elapsed_h = (time.time() - t_start) / 3600.0
        log(f"{eid} complete. elapsed {elapsed_h:.2f}h")

    log(f"=== campaign done in {(time.time()-t_start)/3600.0:.2f}h ===")
    with open(SUMMARY, "a") as f:
        f.write(f"\n_Campaign finished in {(time.time()-t_start)/3600.0:.2f}h "
                f"at {time.strftime('%Y-%m-%d %H:%M')}._\n")


if __name__ == "__main__":
    main()
