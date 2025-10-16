import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands
from config import Config

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["ret_1"] = out["Close"].pct_change()
    out["ret_5"] = out["Close"].pct_change(5)
    out["ret_10"] = out["Close"].pct_change(10)

    out["sma_10"] = out["Close"].rolling(10).mean()
    out["sma_20"] = out["Close"].rolling(20).mean()
    out["sma_ratio"] = out["sma_10"] / out["sma_20"] - 1

    rsi = RSIIndicator(out["Close"], window=14)
    out["rsi_14"] = rsi.rsi()

    macd = MACD(out["Close"])
    out["macd"] = macd.macd()
    out["macd_signal"] = macd.macd_signal()

    bb = BollingerBands(out["Close"], window=20, window_dev=2)
    out["bb_high"] = bb.bollinger_hband()
    out["bb_low"]  = bb.bollinger_lband()
    out["bb_pos"]  = (out["Close"] - out["bb_low"]) / (out["bb_high"] - out["bb_low"])

    out["vol_z"] = (out["Volume"] - out["Volume"].rolling(20).mean()) / (out["Volume"].rolling(20).std() + 1e-9)

    # Label: next-day positive return
    look = Config.lookahead_days
    out["future_ret"] = out["Close"].pct_change(look).shift(-look)
    out["label_up"] = (out["future_ret"] > 0).astype(int)

    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    return out

def train_test_split_time(df: pd.DataFrame, split_date: str):
    train = df.loc[:split_date]
    test  = df.loc[split_date:]
    X_cols = [c for c in df.columns if c not in ["Open","High","Low","Close","Volume","future_ret","label_up"]]
    return train[X_cols], train["label_up"], test[X_cols], test["label_up"], X_cols
