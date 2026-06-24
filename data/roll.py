"""
data/roll.py
Produce a single continuous front-month ES series per period,
rolled on the CME schedule, at 1-min bars (the simulation cadence).
"""

from pathlib import Path
import pandas as pd

DATA_DIR  = Path(__file__).parent
PROC_DIR  = DATA_DIR / "processed"

# CME ES quarterly expiry: 3rd Friday of Mar/Jun/Sep/Dec
# Roll to next contract 8 calendar days before expiry (Thursday prior)
ROLL_DATES = {
    # expiry date        : roll-away date (switch to next contract on this date)
    "2019-03-15": "2019-03-07",
    "2019-06-21": "2019-06-13",
    "2019-09-20": "2019-09-12",
    "2019-12-20": "2019-12-12",
    "2020-03-20": "2020-03-12",
    "2020-06-19": "2020-06-11",
}

# Contract order for front-month resolution
CONTRACT_ORDER = ["ESH9","ESM9","ESU9","ESZ9","ESH0","ESM0","ESU0","ESZ0",
                  "ESH1","ESM1","ESU1","ESZ1"]

def front_month_at(dt: pd.Timestamp) -> str:
    for expiry_str, roll_str in sorted(ROLL_DATES.items()):
        roll_dt = pd.Timestamp(roll_str, tz="UTC")
        if dt < roll_dt:
            # front month is the one expiring at expiry_str
            expiry_dt = pd.Timestamp(expiry_str, tz="UTC")
            # derive symbol from expiry month
            month_code = {3:"H", 6:"M", 9:"U", 12:"Z"}[expiry_dt.month]
            year_code  = str(expiry_dt.year)[-1]
            return f"ES{month_code}{year_code}"
    return "ESZ0"  # fallback for late 2020


def _front_month(df: pd.DataFrame) -> pd.DataFrame:
    """Tag each row with the front-month contract and keep only those rows."""
    df = df.copy()
    df.index = pd.to_datetime(df.index, utc=True)
    df["front"] = df.index.map(front_month_at)
    return df[df["symbol"] == df["front"]].drop(columns="front")


def build_continuous(ohlcv_path: Path, bbo_path: Path, name: str) -> pd.DataFrame:
    """Rolled front-month 1-min series. `close` is the last trade price
    (OHLCV feed); `mid` is the end-of-minute (best_bid+best_ask)/2 from the
    BBO feed. Calibration matches against the mid, which avoids the bid-ask
    bounce carried by the trade price."""
    ohlcv = _front_month(pd.read_csv(ohlcv_path, index_col=0, parse_dates=True))
    df1 = ohlcv[["open","high","low","close","volume"]].resample("1min").agg({
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    }).dropna(subset=["close"])

    bbo = _front_month(pd.read_csv(bbo_path, index_col=0, parse_dates=True))
    mid = ((bbo["bid_px_00"] + bbo["ask_px_00"]) / 2.0).resample("1min").last()
    df1["mid"] = mid.reindex(df1.index).ffill().bfill()

    out = PROC_DIR / f"{name}_1m.csv"
    df1.to_csv(out)
    print(f"  -> {out}  ({len(df1):,} bars)")
    return df1


print("Building continuous front-month series...")
calm     = build_continuous(PROC_DIR / "ohlcv_calm.csv",
                            PROC_DIR / "bbo_calm.csv",     "ES_front_calm")
stressed = build_continuous(PROC_DIR / "ohlcv_stressed.csv",
                            PROC_DIR / "bbo_stressed.csv", "ES_front_stressed")
print("Done.")