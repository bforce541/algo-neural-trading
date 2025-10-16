import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from config import Config
from utils import ensure_dirs

def long_flat_backtest(prices: pd.Series, preds: pd.Series, threshold: float = 0.5) -> dict:
    """preds = probability of up-move; position = 1 if prob > threshold else 0"""
    preds = preds.reindex(prices.index).fillna(0.0)
    pos = (preds > threshold).astype(int)
    ret = prices.pct_change().fillna(0.0)
    strat_ret = ret * pos.shift(1).fillna(0)  # enter next day open/close effect simplified
    equity = (1 + strat_ret).cumprod()
    buyhold = (1 + ret).cumprod()

    stats = {
        "Total Return (strategy)": equity.iloc[-1] - 1,
        "Total Return (buy&hold)": buyhold.iloc[-1] - 1,
        "Sharpe (daily)": np.sqrt(252) * strat_ret.mean() / (strat_ret.std() + 1e-9),
        "Win Rate": (strat_ret > 0).mean(),
    }
    return stats, equity, buyhold

def save_plot(equity: pd.Series, buyhold: pd.Series, title: str, outpath: str):
    ensure_dirs(Config.artifacts_dir)
    plt.figure(figsize=(9,5))
    equity.plot(label="Strategy")
    buyhold.plot(label="Buy & Hold")
    plt.legend(); plt.title(title); plt.xlabel("Date"); plt.ylabel("Equity (normalized)")
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(outpath); plt.close()
