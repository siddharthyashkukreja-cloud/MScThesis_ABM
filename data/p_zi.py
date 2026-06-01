"""
data/p_zi.py — order-book calibration from DataBento MBP-10 data.

Three things, measured in one streaming pass per file, aggregated across all
files in a regime:

1. PLACEMENT-DEPTH distribution — how far new limit orders ('A' add events)
   rest from the prevailing mid, in ticks. Fits BOTH:
     - Geometric(p_zi)             — 1 parameter, the current ZI/MT model
     - Log-normal(mu_l, sigma_l)   — 2 parameters, the HFABM placement model
   and reports which fits the empirical histogram better (log-likelihood),
   so the 1-vs-2-parameter choice is evidence-based.

2. CANCELLATION RATE (zi_delta) — per-resting-order per-minute cancellation
   probability. Counts top-10 'C' events (events with depth column < 10) and
   divides by the top-10 resting-order count (sum of bid_ct_00..09 +
   ask_ct_00..09) sampled at minute boundaries. The MBP-10 'depth' column
   makes the top-10 filter clean. Per-step Bernoulli δ = 1 − exp(−rate).

3. BOOK STATE at 1-min boundaries — spread and resting depth, sampled once
   per 1-min bucket (the sim cadence). Empirical spread/depth the agent-
   calibration loop should reproduce as a validation target.

The MBP-10 files are large (250 MB – 1 GB compressed each); this script
STREAMS them — bounded memory (per-symbol histogram + a handful of 1-min
samples + cancellation tallies).

Input  : four MBP-10 .zst files per regime — see REGIME_FILES below.
Output : output/p_zi_params.json

Usage:
    python data/p_zi.py calibrate [max_add]                            # all regimes, all files
    python data/p_zi.py calibrate-regime <regime> [max_add_per_file]   # one regime, all files
    python data/p_zi.py calibrate-file <path.csv.zst> [regime] [max_add]
"""

from __future__ import annotations
import contextlib
import csv
import io
import json
import math
import statistics
import subprocess
import sys
from collections import Counter
from pathlib import Path

try:
    import zstandard as zstd
    _HAVE_ZSTD_MODULE = True
except ImportError:                     # fall back to the `zstd` CLI
    _HAVE_ZSTD_MODULE = False


@contextlib.contextmanager
def _zst_text(path: Path):
    """Yield a decompressed UTF-8 text stream for a .zst file — streaming,
    never loading the whole file. Uses the `zstandard` module if installed,
    else pipes through the `zstd` CLI."""
    if _HAVE_ZSTD_MODULE:
        dctx = zstd.ZstdDecompressor()
        with open(path, "rb") as fh, dctx.stream_reader(fh) as reader:
            yield io.TextIOWrapper(reader, encoding="utf-8", newline="")
    else:
        proc = subprocess.Popen(["zstdcat", str(path)], stdout=subprocess.PIPE)
        try:
            yield io.TextIOWrapper(proc.stdout, encoding="utf-8", newline="")
        finally:
            if proc.stdout:
                proc.stdout.close()
            proc.wait()

REPO_DIR = Path(__file__).parent.parent
OUT_DIR = REPO_DIR / "output"

# Four MBP-10 files per regime (calm = 2019 / stressed = 2020 COVID week).
L2_DIR = Path("/Users/siddharth/Downloads/L2_data_thesis")
REGIME_FILES = {
    "calm": [
        L2_DIR / "glbx-mdp3-20190603.mbp-10.csv.zst",
        L2_DIR / "glbx-mdp3-20190604.mbp-10.csv.zst",
        L2_DIR / "glbx-mdp3-20190605.mbp-10.csv.zst",
        L2_DIR / "glbx-mdp3-20190606.mbp-10.csv.zst",
    ],
    "stressed": [
        L2_DIR / "glbx-mdp3-20200316.mbp-10.csv.zst",
        L2_DIR / "glbx-mdp3-20200317.mbp-10.csv.zst",
        L2_DIR / "glbx-mdp3-20200318.mbp-10.csv.zst",
        L2_DIR / "glbx-mdp3-20200319.mbp-10.csv.zst",
    ],
}

TICK_SIZE = 0.25
RTH_START_S = 13 * 3600 + 30 * 60      # 13:30 UTC
RTH_END_S = 20 * 3600 + 15 * 60        # 20:15 UTC
MAX_DEPTH_TICKS = 200                   # histogram tail truncation
BUCKET_S = 60                           # 1-min book-state sampling cadence


def _parse_price(raw: str) -> float:
    """DataBento prices are either decimal or 1e-9 fixed-point integers.
    ES decimal is ~2000-3500; the nano form is ~2.5e12 — detect by magnitude."""
    v = float(raw)
    if abs(v) > 1e7:
        v *= 1e-9
    return v


def _seconds_of_day(ts_raw: str):
    """ts_event as ns-since-epoch integer, or ISO-8601 string. Returns
    seconds-of-day in UTC, or None if unparseable."""
    s = ts_raw.strip()
    if not s:
        return None
    try:
        ns = int(s)
        return (ns // 1_000_000_000) % 86400
    except ValueError:
        try:
            t = s.split("T")[1] if "T" in s else s.split(" ")[1]
            hh, mm, ss = t[:8].split(":")
            return int(hh) * 3600 + int(mm) * 60 + int(float(ss))
        except Exception:
            return None


def _new_cd():
    return dict(n_cancels_top10=0, sum_cancel_size_top10=0, n_min_samples=0,
                sum_resting_orders_top10=0, sum_resting_volume_top10=0,
                sum_passive_add_size=0, n_passive_adds=0)


def _process_file(path: Path, rth_only: bool = True, max_add: int | None = None):
    """Stream one MBP-10 file. Returns (depth_hist, book_samples, cancel_data, stats).
      depth_hist   : {symbol: Counter{depth_ticks: count}}    — placement depths
      book_samples : {symbol: [(spread_ticks, inside_depth, total_depth), ...]}
                     book state sampled once per 1-min bucket
      cancel_data  : {symbol: {... cancel + resting-order tallies ...}}
                     for the δ calibration: top-10 cancel events vs resting orders
    'depth' column < 10 = event within top-10 levels. Resting orders sampled
    from bid_ct_00..09 + ask_ct_00..09 (per-level order counts) at minute
    boundaries.
    If `max_add` is set, stop once that many passive add events are collected."""
    depth_hist: dict = {}
    book_samples: dict = {}
    cancel_data: dict = {}
    last_bucket: dict = {}
    n_rows = n_add = n_passive = n_aggressive = n_rth_skip = n_cancel_top10 = 0
    with _zst_text(path) as text:
        rd = csv.DictReader(text)
        need = {"action", "side", "price", "depth", "size",
                "bid_px_00", "ask_px_00", "symbol",
                "bid_ct_00", "ask_ct_00"}
        missing = need - set(rd.fieldnames or [])
        if missing:
            raise KeyError(f"{path.name}: missing columns {missing}; "
                           f"header was {rd.fieldnames}")
        ts_col = "ts_event" if "ts_event" in rd.fieldnames else "ts_recv"
        for row in rd:
            n_rows += 1
            action = row["action"]
            if action not in ("A", "C"):
                continue
            sod = _seconds_of_day(row.get(ts_col, ""))
            in_rth = sod is not None and (RTH_START_S <= sod < RTH_END_S)
            if rth_only and not in_rth:
                if action == "A":
                    n_rth_skip += 1
                continue
            sym = row["symbol"]
            try:
                depth_lvl = int(row["depth"])
            except (ValueError, TypeError, KeyError):
                depth_lvl = -1

            # ── per-minute book-state + resting-order sample ──
            if sod is not None:
                bucket = int(sod // BUCKET_S)
                if last_bucket.get(sym) != bucket:
                    last_bucket[sym] = bucket
                    try:
                        bid = _parse_price(row["bid_px_00"])
                        ask = _parse_price(row["ask_px_00"])
                    except (ValueError, KeyError):
                        bid = ask = -1.0
                    if bid > 0 and ask > 0 and ask >= bid:
                        try:
                            total_vol = 0.0
                            n_orders_top10 = 0
                            for i in range(10):
                                total_vol += float(row.get(f"bid_sz_{i:02d}") or 0)
                                total_vol += float(row.get(f"ask_sz_{i:02d}") or 0)
                                n_orders_top10 += int(float(row.get(f"bid_ct_{i:02d}") or 0))
                                n_orders_top10 += int(float(row.get(f"ask_ct_{i:02d}") or 0))
                            inside = (float(row.get("bid_sz_00") or 0) +
                                      float(row.get("ask_sz_00") or 0))
                        except ValueError:
                            total_vol = inside = 0.0
                            n_orders_top10 = 0
                        book_samples.setdefault(sym, []).append(
                            ((ask - bid) / TICK_SIZE, inside, total_vol))
                        cd = cancel_data.setdefault(sym, _new_cd())
                        cd["n_min_samples"] += 1
                        cd["sum_resting_orders_top10"] += n_orders_top10
                        cd["sum_resting_volume_top10"] += total_vol

            if action == "C":
                # top-10 cancel — depth column gives the level directly
                if 0 <= depth_lvl < 10:
                    n_cancel_top10 += 1
                    cd = cancel_data.setdefault(sym, _new_cd())
                    cd["n_cancels_top10"] += 1
                    try:
                        cd["sum_cancel_size_top10"] += int(float(row.get("size") or 0))
                    except ValueError:
                        pass
                continue

            # action == "A" — placement depth (passive adds, ticks from mid)
            n_add += 1
            try:
                bid = _parse_price(row["bid_px_00"])
                ask = _parse_price(row["ask_px_00"])
                px = _parse_price(row["price"])
            except (ValueError, KeyError):
                continue
            if bid <= 0 or ask <= 0 or ask < bid:
                continue
            mid = 0.5 * (bid + ask)
            side = row["side"]
            if side == "B":
                depth = (mid - px) / TICK_SIZE       # passive bid rests below mid
            elif side == "A":
                depth = (px - mid) / TICK_SIZE       # passive ask rests above mid
            else:
                continue
            # round half UP — plain round() is banker's rounding, which bins
            # X.5-tick distances off a half-tick mid onto even integers (the
            # even-tick sawtooth artifact). round-half-up is monotone.
            k = int(depth + 0.5)
            if k < 1:
                n_aggressive += 1                    # at/through mid — marketable-ish
                continue
            n_passive += 1
            k = min(k, MAX_DEPTH_TICKS)
            depth_hist.setdefault(sym, Counter())[k] += 1
            try:
                cd = cancel_data.setdefault(sym, _new_cd())
                cd["sum_passive_add_size"] += int(float(row.get("size") or 0))
                cd["n_passive_adds"] += 1
            except ValueError:
                pass
            if n_rows % 5_000_000 == 0:
                print(f"    {path.name}: {n_rows/1e6:.0f}M rows, "
                      f"{n_passive/1e6:.2f}M passive adds, "
                      f"{n_cancel_top10/1e6:.2f}M top-10 cancels", flush=True)
            if max_add is not None and n_passive >= max_add:
                break
    stats = dict(n_rows=n_rows, n_add=n_add, n_passive=n_passive,
                 n_aggressive=n_aggressive, n_rth_skip=n_rth_skip,
                 n_cancel_top10=n_cancel_top10)
    return depth_hist, book_samples, cancel_data, stats


def _fit_geometric(counter: Counter) -> dict:
    """MLE for Geometric on support {1, 2, ...}: p = 1 / mean(k). Reports the
    log-likelihood of the empirical histogram under the fitted distribution."""
    total = sum(counter.values())
    if total == 0:
        return {"p_zi": None, "mean_depth_ticks": None, "loglik": None, "n": 0}
    mean = sum(k * c for k, c in counter.items()) / total
    p = 1.0 / mean
    loglik = None
    if 0.0 < p < 1.0:
        lp, l1p = math.log(p), math.log(1.0 - p)
        loglik = sum(c * (lp + (k - 1) * l1p) for k, c in counter.items())
    return {"p_zi": p, "mean_depth_ticks": mean, "loglik": loglik, "n": total}


def _fit_lognormal(counter: Counter) -> dict:
    """MLE for a log-normal on the depth ticks (fit on log k), with a
    discretised log-likelihood: P(k) = Φ((ln(k+½)−μ)/σ) − Φ((ln(k−½)−μ)/σ)."""
    total = sum(counter.values())
    if total == 0:
        return {"mu": None, "sigma": None, "loglik": None, "n": 0}
    mu = sum(math.log(k) * c for k, c in counter.items()) / total
    var = sum((math.log(k) - mu) ** 2 * c for k, c in counter.items()) / total
    sigma = math.sqrt(var) if var > 0 else 1e-9

    def Phi(x):
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    loglik = 0.0
    for k, c in counter.items():
        hi = Phi((math.log(k + 0.5) - mu) / sigma)
        lo = Phi((math.log(max(k - 0.5, 1e-9)) - mu) / sigma)
        loglik += c * math.log(max(hi - lo, 1e-300))
    return {"mu": mu, "sigma": sigma, "loglik": loglik, "n": total}


def _fit_cancel(cd: dict) -> dict:
    """Per-resting-order per-minute cancellation rate from aggregated counts.
    Per-step Bernoulli δ = 1 − exp(−rate) at 1-min cadence (≈ rate for small δ).
    Numerator: top-10 cancel events. Denominator: top-10 resting orders summed
    across minute snapshots (sum_resting_orders / n_min_samples = mean)."""
    nC = cd.get("n_cancels_top10", 0)
    nM = cd.get("n_min_samples", 0)
    sumR = cd.get("sum_resting_orders_top10", 0)
    sumV = cd.get("sum_resting_volume_top10", 0)
    sumCv = cd.get("sum_cancel_size_top10", 0)
    sumA = cd.get("sum_passive_add_size", 0)
    nA = cd.get("n_passive_adds", 0)
    if nM == 0 or sumR == 0:
        return {"zi_delta": None, "rate_per_min": None,
                "mean_cancels_per_min": None, "mean_resting_orders_top10": None,
                "n_minutes": nM, "n_cancels_total": nC}
    mean_cancels = nC / nM
    mean_resting = sumR / nM
    rate = mean_cancels / mean_resting             # per-order per-min Poisson rate
    delta = 1.0 - math.exp(-rate)                  # 1-min Bernoulli probability
    mean_resting_vol = sumV / nM
    mean_cancel_size = sumCv / max(nC, 1)
    mean_add_size = sumA / max(nA, 1)
    # volume-based sanity check: per-contract per-min cancel fraction
    rate_vol = (sumCv / nM) / max(mean_resting_vol, 1.0)
    return {
        "zi_delta": delta,
        "rate_per_min": rate,
        "mean_cancels_per_min": mean_cancels,
        "mean_resting_orders_top10": mean_resting,
        "mean_resting_volume_top10": mean_resting_vol,
        "mean_cancel_size": mean_cancel_size,
        "mean_passive_add_size": mean_add_size,
        "rate_per_min_volume_based": rate_vol,
        "n_minutes": nM,
        "n_cancels_total": nC,
        "n_passive_adds_total": nA,
    }


def _book_stats(samples: list) -> dict:
    """samples: list of (spread_ticks, inside_depth, total_depth)."""
    if not samples:
        return {}

    def q(xs, p):
        xs = sorted(xs)
        return xs[min(int(p * len(xs)), len(xs) - 1)]

    spr = [s[0] for s in samples]
    ins = [s[1] for s in samples]
    tot = [s[2] for s in samples]
    return {
        "n_samples": len(samples),
        "spread_ticks": {"mean": statistics.fmean(spr), "median": q(spr, 0.5),
                         "p10": q(spr, 0.1), "p90": q(spr, 0.9)},
        "inside_depth": {"mean": statistics.fmean(ins), "median": q(ins, 0.5)},
        "total_depth_10lvl": {"mean": statistics.fmean(tot),
                              "median": q(tot, 0.5)},
    }


def _summarise(counter: Counter, book: list, cd: dict, stats: dict,
               front, source) -> dict:
    """Build the per-regime result dict: placement fits + cancellation + book state.
    `front` is either a single symbol or {filename: symbol} when aggregated.
    `source` mirrors that — single filename or list of filenames."""
    geom = _fit_geometric(counter)
    logn = _fit_lognormal(counter)
    total = sum(counter.values())
    hist_frac = {str(k): counter.get(k, 0) / total for k in range(1, 16)}
    mode_k = max(counter, key=counter.get)
    tail = {"P(k>10)": sum(c for k, c in counter.items() if k > 10) / total,
            "P(k>20)": sum(c for k, c in counter.items() if k > 20) / total}
    g_ll = geom["loglik"] if geom["loglik"] is not None else -1e300
    l_ll = logn["loglik"] if logn["loglik"] is not None else -1e300
    return {
        "source": source,
        "front_month": front,
        "n_rows": stats["n_rows"],
        "n_add": stats["n_add"],
        "n_passive": stats["n_passive"],
        "n_cancel_top10": stats.get("n_cancel_top10", 0),
        "frac_aggressive": stats["n_aggressive"] /
        max(stats["n_aggressive"] + stats["n_passive"], 1),
        "placement": {
            "geometric": geom,
            "lognormal": logn,
            "better_fit": "lognormal" if l_ll > g_ll else "geometric",
            "loglik_gain_lognormal": l_ll - g_ll,
            "mode_ticks": mode_k,
            "hist_frac_k1_15": hist_frac,
            "tail": tail,
        },
        "cancellation": _fit_cancel(cd),
        "book_state_1min": _book_stats(book),
    }


def _print_summary(regime: str, res: dict):
    pl = res["placement"]
    g, l = pl["geometric"], pl["lognormal"]
    front = res["front_month"]
    if isinstance(front, dict):
        front_str = ", ".join(f"{f}:{s}" for f, s in front.items())
    else:
        front_str = str(front)
    print(f"\n[{regime}] front={front_str}  "
          f"n_passive={g['n']:,}  frac_aggressive={res['frac_aggressive']:.3f}")
    print(f"  PLACEMENT DEPTH:")
    print(f"    Geometric : p_zi={g['p_zi']:.4f}  mean={g['mean_depth_ticks']:.2f} ticks"
          f"  loglik={g['loglik']:,.0f}")
    print(f"    Log-normal: mu={l['mu']:.3f}  sigma={l['sigma']:.3f}"
          f"  loglik={l['loglik']:,.0f}")
    print(f"    better fit: {pl['better_fit'].upper()}  "
          f"(Δloglik={pl['loglik_gain_lognormal']:,.0f})  "
          f"empirical mode at k={pl['mode_ticks']} ticks")
    hf = pl["hist_frac_k1_15"]
    bars = "  ".join(f"{k}:{hf[str(k)]*100:.0f}%" for k in range(1, 9))
    print(f"    hist  {bars}")
    print(f"    tail  P(k>10)={pl['tail']['P(k>10)']:.3f}  "
          f"P(k>20)={pl['tail']['P(k>20)']:.3f}")
    cn = res.get("cancellation") or {}
    if cn and cn.get("zi_delta") is not None:
        print(f"  CANCELLATION (top-10 only):")
        print(f"    mean cancels/min = {cn['mean_cancels_per_min']:.1f}   "
              f"mean resting orders (top-10) = {cn['mean_resting_orders_top10']:.0f}")
        print(f"    rate = {cn['rate_per_min']:.4f}/order/min   →   "
              f"zi_delta (Bernoulli 1-min) = {cn['zi_delta']:.4f}")
        print(f"    volume-based check: rate = {cn['rate_per_min_volume_based']:.4f}/min   "
              f"(mean add size {cn['mean_passive_add_size']:.1f}, "
              f"mean cancel size {cn['mean_cancel_size']:.1f})")
        print(f"    ({cn['n_minutes']} min samples, "
              f"{cn['n_cancels_total']:,} cancels)")
    bk = res.get("book_state_1min") or {}
    if bk:
        s, d, t = bk["spread_ticks"], bk["inside_depth"], bk["total_depth_10lvl"]
        print(f"  BOOK STATE (1-min, n={bk['n_samples']}):")
        print(f"    spread  mean={s['mean']:.2f} median={s['median']:.0f} "
              f"p10={s['p10']:.0f} p90={s['p90']:.0f} ticks")
        print(f"    inside depth  mean={d['mean']:.0f} median={d['median']:.0f}"
              f"   |  10-level depth  mean={t['mean']:.0f} median={t['median']:.0f}")


def calibrate(max_add_per_file: int | None = None, verbose: bool = True) -> dict:
    """Calibrate both regimes — all files per regime, from REGIME_FILES.
    `max_add_per_file` caps passive adds per file (for fast testing); None = full."""
    results = {}
    for regime, paths in REGIME_FILES.items():
        results[regime] = calibrate_regime(regime, paths,
                                           max_add_per_file=max_add_per_file,
                                           verbose=verbose)
    return results


def _agg_cd(target: dict, src: dict):
    for k, v in src.items():
        target[k] = target.get(k, 0) + v


def calibrate_regime(regime: str, paths: list,
                     max_add_per_file: int | None = None,
                     verbose: bool = True) -> dict:
    """Stream all files for a regime; aggregate depth histograms, cancel
    tallies, and book samples across files. Front-month picked per file
    (volume-dominant symbol)."""
    agg_hist: Counter = Counter()
    agg_book: list = []
    agg_cd: dict = _new_cd()
    agg_stats = dict(n_rows=0, n_add=0, n_passive=0, n_aggressive=0,
                     n_rth_skip=0, n_cancel_top10=0)
    front_by_file: dict = {}
    sources: list = []
    for path in paths:
        path = Path(path)
        if not path.exists():
            if verbose:
                print(f"[{regime}] file missing: {path}")
            continue
        if verbose:
            cap = f"(cap {max_add_per_file} adds) " if max_add_per_file else ""
            print(f"[{regime}] streaming {path.name} {cap}...", flush=True)
        depth_hist, book_samples, cancel_data, stats = _process_file(
            path, max_add=max_add_per_file)
        if not depth_hist:
            if verbose:
                print(f"[{regime}]   no passive adds in {path.name}")
            continue
        front = max(depth_hist, key=lambda s: sum(depth_hist[s].values()))
        front_by_file[path.name] = front
        sources.append(path.name)
        agg_hist.update(depth_hist[front])
        agg_book.extend(book_samples.get(front, []))
        if front in cancel_data:
            _agg_cd(agg_cd, cancel_data[front])
        for k in agg_stats:
            agg_stats[k] += stats.get(k, 0)
        if verbose:
            cd_front = cancel_data.get(front, {})
            print(f"  ✓ {path.name}: front={front}, "
                  f"{stats['n_passive']:,} passive adds, "
                  f"{cd_front.get('n_cancels_top10', 0):,} top-10 cancels, "
                  f"{cd_front.get('n_min_samples', 0)} min samples")
    if not agg_hist:
        print(f"[{regime}] aggregation empty — no usable files")
        return {}
    res = _summarise(agg_hist, agg_book, agg_cd, agg_stats,
                     front=front_by_file, source=sources)
    if verbose:
        _print_summary(regime, res)
    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / "p_zi_params.json"
    existing = json.loads(out.read_text()) if out.exists() else {}
    existing[regime] = res
    out.write_text(json.dumps(existing, indent=2))
    if verbose:
        print(f"\nSaved {out}  (regime '{regime}')")
        print("→ copy placement.geometric.p_zi into model/globals.py :: P_ZI")
        print("→ copy cancellation.zi_delta into model/globals.py :: ZI_DELTA")
    return res


def calibrate_file(path: Path, regime: str = "calm",
                   max_add: int | None = None, verbose: bool = True) -> dict:
    """Single-file calibration — useful for quick smoke tests."""
    return calibrate_regime(regime, [Path(path)],
                            max_add_per_file=max_add, verbose=verbose)


def _main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    cmd = sys.argv[1]
    if cmd == "calibrate":
        max_add = int(sys.argv[2]) if len(sys.argv) >= 3 else None
        calibrate(max_add_per_file=max_add)
    elif cmd == "calibrate-regime":
        if len(sys.argv) < 3:
            print("Usage: calibrate-regime <regime> [max_add_per_file]")
            sys.exit(1)
        regime = sys.argv[2]
        if regime not in REGIME_FILES:
            print(f"Unknown regime '{regime}'. Known: {list(REGIME_FILES)}")
            sys.exit(1)
        max_add = int(sys.argv[3]) if len(sys.argv) >= 4 else None
        calibrate_regime(regime, REGIME_FILES[regime],
                         max_add_per_file=max_add)
    elif cmd == "calibrate-file":
        if len(sys.argv) < 3:
            print("Usage: calibrate-file <path> [regime] [max_add]")
            sys.exit(1)
        regime = sys.argv[3] if len(sys.argv) >= 4 else "calm"
        max_add = int(sys.argv[4]) if len(sys.argv) >= 5 else None
        calibrate_file(Path(sys.argv[2]), regime, max_add=max_add)
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    _main()
