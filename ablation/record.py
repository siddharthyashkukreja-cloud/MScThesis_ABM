#!/usr/bin/env python3
"""Append one results.csv row per regime from output/calibrated_params.json.

Lives in ablation/ so it survives the per-cell `git restore --source=BASE`.
Usage:
  python3 ablation/record.py --cell C0 --fundamental Kalman --mm 0 --vt 0 \
      --free ft_sigma_c,zi_alpha,zi_mu,zi_delta --fixed p_zi \
      [--json output/calibrated_params.json] [--out ablation/results.csv]

Columns (PLAN.md):
  cell, regime, fundamental, mm, vt, free_params, fixed_params, D, dKS, dV,
  dACF1, dACF2, dHill, ret_std, ret_kurtosis, hill, acf_r_1, acf_absr_1,
  acf_absr_5, acf_absr_10, acf_absr_20, surrogate_r2, n_samples, timestamp
"""
import argparse
import csv
import datetime
import json
import os
import sys

COLUMNS = [
    "cell", "regime", "fundamental", "mm", "vt", "free_params", "fixed_params",
    "D", "dKS", "dV", "dACF1", "dACF2", "dHill", "ret_std", "ret_kurtosis",
    "hill", "acf_r_1", "acf_absr_1", "acf_absr_5", "acf_absr_10", "acf_absr_20",
    "surrogate_r2", "n_samples", "timestamp",
]


def g(d, *keys, default=""):
    """Nested-get with default; tolerate None and missing keys."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur or cur[k] is None:
            return default
        cur = cur[k]
    return cur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--fundamental", required=True)
    ap.add_argument("--mm", required=True)
    ap.add_argument("--vt", required=True)
    ap.add_argument("--free", required=True, help="comma-separated free param keys")
    ap.add_argument("--fixed", default="", help="comma-separated fixed param keys")
    ap.add_argument("--json", default="output/calibrated_params.json")
    ap.add_argument("--out", default="ablation/results.csv")
    ap.add_argument("--timestamp", default=None,
                    help="ISO timestamp; defaults to now (UTC)")
    args = ap.parse_args()

    ts = args.timestamp or (datetime.datetime.now(datetime.timezone.utc)
                            .isoformat(timespec="seconds").replace("+00:00", "Z"))

    with open(args.json) as f:
        payload = json.load(f)

    free = "|".join(p for p in args.free.split(",") if p)
    fixed = "|".join(p for p in args.fixed.split(",") if p) or "none"

    results = payload.get("results", {})
    if not results:
        print(f"WARNING: no 'results' in {args.json}", file=sys.stderr)

    rows = []
    for regime, r in results.items():
        val = r.get("validated", {}) or {}
        cd = r.get("component_deltas", {}) or {}
        rows.append({
            "cell": args.cell,
            "regime": regime,
            "fundamental": args.fundamental,
            "mm": args.mm,
            "vt": args.vt,
            "free_params": free,
            "fixed_params": fixed,
            "D": g(r, "true_loss_validated"),
            "dKS": g(cd, "KS"),
            "dV": g(cd, "V"),
            "dACF1": g(cd, "ACF1"),
            "dACF2": g(cd, "ACF2"),
            "dHill": g(cd, "Hill"),
            "ret_std": g(val, "ret_std"),
            "ret_kurtosis": g(val, "ret_kurtosis"),
            "hill": g(val, "hill_tail_index"),
            "acf_r_1": g(val, "acf_r_1"),
            "acf_absr_1": g(val, "acf_absr_1"),
            "acf_absr_5": g(val, "acf_absr_5"),
            "acf_absr_10": g(val, "acf_absr_10"),
            "acf_absr_20": g(val, "acf_absr_20"),
            "surrogate_r2": g(r, "surrogate_d_accuracy", "r2"),
            "n_samples": g(r, "n_total_samples"),
            "timestamp": ts,
        })

    write_header = not os.path.exists(args.out) or os.path.getsize(args.out) == 0
    with open(args.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if write_header:
            w.writeheader()
        for row in rows:
            w.writerow(row)

    print(f"Appended {len(rows)} row(s) for cell {args.cell} to {args.out}")
    for row in rows:
        print(f"  {row['regime']}: D={row['D']} dKS={row['dKS']} dHill={row['dHill']} "
              f"hill={row['hill']} ret_kurtosis={row['ret_kurtosis']}")


if __name__ == "__main__":
    main()
