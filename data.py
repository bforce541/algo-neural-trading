from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

from config import Config
from utils import ensure_dirs


def _normalize_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    required = ["Open", "High", "Low", "Close", "Volume"]
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(c[0]).capitalize() for c in df.columns]
    else:
        df = df.rename(columns={c: str(c).capitalize() for c in df.columns})
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required OHLCV columns: {missing}")
    return df[required].copy()


def fetch_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data fetched for {ticker} between {start} and {end}")
    df = _normalize_ohlcv_columns(df)
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    return df.sort_index()


def load_or_fetch(tickers: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    ensure_dirs(Config.data_dir)
    out: dict[str, pd.DataFrame] = {}

    for ticker in tickers:
        fp = Path(Config.data_dir) / f"{ticker}_{start}_{end}.parquet"
        if fp.exists():
            df = pd.read_parquet(fp)
            df.index = pd.to_datetime(df.index)
        else:
            df = fetch_ohlcv(ticker=ticker, start=start, end=end)
            df.to_parquet(fp, index=True)
        out[ticker] = df

    return out
