from __future__ import annotations

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands

from config import Config

BASE_COLS = ["Open", "High", "Low", "Close", "Volume"]
LABEL_COLS = ["future_ret", "label_up"]


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["ret_1"] = out["Close"].pct_change(1)
    out["ret_5"] = out["Close"].pct_change(5)
    out["ret_10"] = out["Close"].pct_change(10)

    out["sma_10"] = out["Close"].rolling(10).mean()
    out["sma_20"] = out["Close"].rolling(20).mean()
    out["sma_ratio"] = out["sma_10"] / out["sma_20"] - 1

    out["range_pct"] = (out["High"] - out["Low"]) / (out["Close"] + 1e-9)

    rsi = RSIIndicator(out["Close"], window=14)
    out["rsi_14"] = rsi.rsi()

    macd = MACD(out["Close"])
    out["macd"] = macd.macd()
    out["macd_signal"] = macd.macd_signal()
    out["macd_hist"] = out["macd"] - out["macd_signal"]

    bb = BollingerBands(out["Close"], window=20, window_dev=2)
    out["bb_high"] = bb.bollinger_hband()
    out["bb_low"] = bb.bollinger_lband()
    out["bb_pos"] = (out["Close"] - out["bb_low"]) / ((out["bb_high"] - out["bb_low"]) + 1e-9)

    rolling_vol = out["Volume"].rolling(20)
    out["vol_z"] = (out["Volume"] - rolling_vol.mean()) / (rolling_vol.std() + 1e-9)

    look = Config.lookahead_days
    out["future_ret"] = out["Close"].pct_change(look).shift(-look)
    out["label_up"] = (out["future_ret"] > 0).astype(int)

    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    return out


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in BASE_COLS + LABEL_COLS]


def train_test_split_time(df: pd.DataFrame, split_date: str):
    split_dt = pd.Timestamp(split_date)
    train = df[df.index < split_dt]
    test = df[df.index >= split_dt]

    if train.empty or test.empty:
        raise ValueError(
            f"Invalid split_date={split_date}. train_rows={len(train)}, test_rows={len(test)}"
        )

    cols = feature_columns(df)
    return train[cols], train["label_up"], test[cols], test["label_up"], cols
