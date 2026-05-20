from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import Config
from utils import ensure_dirs


def long_flat_backtest(
    prices: pd.Series,
    preds: pd.Series,
    threshold: float = 0.5,
    fee_bps: float = Config.fee_bps,
    slippage_bps: float = Config.slippage_bps,
) -> tuple[dict, pd.Series, pd.Series, pd.Series]:
    preds = preds.reindex(prices.index).fillna(0.0)
    pos = (preds > threshold).astype(int)

    ret = prices.pct_change().fillna(0.0)
    shifted_pos = pos.shift(1).fillna(0)
    gross = ret * shifted_pos

    trades = shifted_pos.diff().abs().fillna(shifted_pos.abs())
    tc_per_turnover = (fee_bps + slippage_bps) / 10_000.0
    costs = trades * tc_per_turnover

    strat_ret = gross - costs

    equity = (1 + strat_ret).cumprod()
    buyhold = (1 + ret).cumprod()

    daily_vol = strat_ret.std() + 1e-9
    sharpe = np.sqrt(Config.trading_days_per_year) * strat_ret.mean() / daily_vol
    cum_max = equity.cummax()
    drawdown = equity / cum_max - 1

    stats = {
        "Total Return (strategy)": float(equity.iloc[-1] - 1),
        "Total Return (buy&hold)": float(buyhold.iloc[-1] - 1),
        "Sharpe (daily)": float(sharpe),
        "Max Drawdown": float(drawdown.min()),
        "Win Rate": float((strat_ret > 0).mean()),
        "Turnover (avg daily)": float(trades.mean()),
    }
    return stats, equity, buyhold, drawdown


def save_plot(
    equity: pd.Series,
    buyhold: pd.Series,
    drawdown: pd.Series,
    title: str,
    outpath: str,
) -> None:
    ensure_dirs(Config.artifacts_dir)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(11, 7),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )

    fig.patch.set_facecolor("#f4f4f1")
    ax1.set_facecolor("#f9f8f4")
    ax2.set_facecolor("#f9f8f4")

    ax1.plot(equity.index, equity.values, label="Strategy", color="#0a9396", linewidth=2.2)
    ax1.plot(buyhold.index, buyhold.values, label="Buy & Hold", color="#ca6702", linewidth=1.8, alpha=0.9)
    ax1.set_title(title, fontsize=13, fontweight="bold")
    ax1.set_ylabel("Equity (normalized)")
    ax1.legend(frameon=False)

    ax2.fill_between(drawdown.index, drawdown.values, 0, color="#bb3e03", alpha=0.35)
    ax2.set_ylabel("Drawdown")
    ax2.set_xlabel("Date")

    for ax in (ax1, ax2):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=140)
    plt.close(fig)
