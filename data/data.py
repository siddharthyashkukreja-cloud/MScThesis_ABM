from pathlib import Path
import databento as db
import pandas as pd
import zstandard as zstd
import io

DATA_DIR = Path(__file__).parent
OUT_DIR  = DATA_DIR / "processed"
OUT_DIR.mkdir(exist_ok=True)

FILES = {
    "glbx-mdp3-20190101-20191230.ohlcv-1m.csv.zst":  ("ohlcv_calm",     "ohlcv"),
    "glbx-mdp3-20200224-20200402.ohlcv-1m.dbn.zst":  ("ohlcv_stressed", "ohlcv"),
    "glbx-mdp3-20190101-20191230.bbo-1m.csv.zst":    ("bbo_calm",       "bbo"),
    "glbx-mdp3-20200224-20200402.bbo-1m.csv.zst":    ("bbo_stressed",   "bbo"),
}

RTH_UTC = ("13:30", "20:15")  # 08:30–15:15 CT


def load_csv_zst(path: Path) -> pd.DataFrame:
    dctx = zstd.ZstdDecompressor()
    with open(path, "rb") as fh:
        with dctx.stream_reader(fh) as reader:
            return pd.read_csv(io.TextIOWrapper(reader, encoding="utf-8"))


def load_dbn_zst(path: Path) -> pd.DataFrame:
    return db.DBNStore.from_file(str(path)).to_df()


def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if "ts_event" in df.columns:
        df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True)
        df = df.set_index("ts_event")
    df.index.name = "ts"
    df = df[df["symbol"].str.match(r"^ES[A-Z]\d$")]  # front-month outrights only
    df = df.sort_index().between_time(*RTH_UTC)
    return df[["symbol", "open", "high", "low", "close", "volume"]]


def clean_bbo(df: pd.DataFrame) -> pd.DataFrame:
    # Use ts_recv as index (ts_event has NaNs in BBO)
    df["ts_recv"] = pd.to_datetime(df["ts_recv"], utc=True)
    df = df.set_index("ts_recv")
    df.index.name = "ts"
    df = df[df["symbol"].str.match(r"^ES[A-Z]\d$")]  # outrights only, no spreads
    df = df.sort_index().between_time(*RTH_UTC)
    return df[["symbol", "bid_px_00", "ask_px_00", "bid_sz_00", "ask_sz_00"]]


for fname, (stem, schema) in FILES.items():
    path = DATA_DIR / fname
    if not path.exists():
        print(f"SKIP {fname}")
        continue

    print(f"Loading {fname} ...")
    df = load_dbn_zst(path) if fname.endswith(".dbn.zst") else load_csv_zst(path)
    df = clean_ohlcv(df) if schema == "ohlcv" else clean_bbo(df)

    out = OUT_DIR / f"{stem}.csv"
    df.to_csv(out)
    print(f"  -> {out}  ({len(df):,} rows, {df['symbol'].nunique()} symbols)")

print("Done.")

OUT_DIR = Path("data/processed")

for fname in OUT_DIR.glob("*.csv"):
    df = pd.read_csv(fname, index_col=0, parse_dates=True)
    if "symbol" not in df.columns:
        print(f"\n{fname.name}  — no symbol column, skipping")
        continue
    print(f"\n{fname.name}")
    print(df["symbol"].value_counts().to_string())