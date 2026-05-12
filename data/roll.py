"""
data/roll.py
Produce a single continuous front-month ES series per period,
rolled on the CME schedule, resampled to 5-min bars.
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


def build_continuous(ohlcv_path: Path, name: str) -> pd.DataFrame:
    df = pd.read_csv(ohlcv_path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)

    # Tag each row with the front-month contract at that timestamp
    df["front"] = df.index.map(front_month_at)

    # Keep only rows where symbol matches the front-month
    df = df[df["symbol"] == df["front"]].drop(columns="front")

    # Resample to 5-min OHLCV
    df5 = df[["open","high","low","close","volume"]].resample("5min").agg({
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    }).dropna(subset=["close"])

    out = PROC_DIR / f"{name}_5m.csv"
    df5.to_csv(out)
    print(f"  -> {out}  ({len(df5):,} bars)")
    return df5


print("Building continuous front-month series...")
calm     = build_continuous(PROC_DIR / "ohlcv_calm.csv",     "ES_front_calm")
stressed = build_continuous(PROC_DIR / "ohlcv_stressed.csv", "ES_front_stressed")
print("Done.")