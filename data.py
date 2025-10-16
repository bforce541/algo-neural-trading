import yfinance as yf
import pandas as pd
from pathlib import Path
from config import Config
from utils import ensure_dirs

def fetch_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    df = df.rename(columns=str.capitalize)  # Open, High, Low, Close, Volume
    df.index.name = "Date"
    return df

def load_or_fetch(tickers: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    ensure_dirs(Config.data_dir)
    out: dict[str, pd.DataFrame] = {}
    for t in tickers:
        fp = Path(Config.data_dir) / f"{t}_{start}_{end}.parquet"
        if fp.exists():
            out[t] = pd.read_parquet(fp)
        else:
            df = fetch_ohlcv(t, start, end)
            df.to_parquet(fp)
            out[t] = df
    return out
