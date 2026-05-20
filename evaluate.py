from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score

from backtest import long_flat_backtest, save_plot
from config import Config
from data import load_or_fetch
from features import make_features, train_test_split_time
from utils import ensure_dirs


def evaluate_single_ticker(
    ticker: str,
    model_name: str,
    start: str,
    end: str,
    threshold: float,
) -> dict:
    model_fp = Path(Config.artifacts_dir) / f"model_{model_name}_{ticker}.joblib"
    if not model_fp.exists():
        raise FileNotFoundError(
            f"Missing model artifact: {model_fp}. Run train.py first for {ticker}/{model_name}."
        )

    model = joblib.load(model_fp)

    raw = load_or_fetch([ticker], start, end)[ticker]
    feat_df = make_features(raw)
    _, _, X_te, y_te, _ = train_test_split_time(feat_df, Config.test_split_date)

    proba = pd.Series(model.predict_proba(X_te)[:, 1], index=X_te.index, name="proba_up")

    auc = float(roc_auc_score(y_te, proba))
    y_pred = (proba > threshold).astype(int)
    report = classification_report(y_te, y_pred, output_dict=True)

    prices = feat_df.loc[X_te.index, "Close"]
    stats, equity, buyhold, drawdown = long_flat_backtest(
        prices=prices,
        preds=proba,
        threshold=threshold,
    )

    ensure_dirs(Config.artifacts_dir)
    Path(Config.artifacts_dir, f"cls_report_{model_name}_{ticker}.json").write_text(
        json.dumps(report, indent=2)
    )
    Path(Config.artifacts_dir, f"bt_stats_{model_name}_{ticker}.json").write_text(
        json.dumps(stats, indent=2)
    )
    save_plot(
        equity=equity,
        buyhold=buyhold,
        drawdown=drawdown,
        title=f"{ticker} | {model_name.upper()} Strategy vs Buy & Hold",
        outpath=str(Path(Config.artifacts_dir, f"equity_{model_name}_{ticker}.png")),
    )

    return {
        "ticker": ticker,
        "model": model_name,
        "auc": auc,
        "backtest": stats,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--model", choices=["logreg", "rf", "nn"], default="rf")
    ap.add_argument("--threshold", type=float, default=0.52)
    args = ap.parse_args()

    results = []
    for ticker in args.tickers:
        results.append(
            evaluate_single_ticker(
                ticker=ticker,
                model_name=args.model,
                start=args.start,
                end=args.end,
                threshold=args.threshold,
            )
        )

    Path(Config.artifacts_dir, f"eval_summary_{args.model}.json").write_text(
        json.dumps(results, indent=2)
    )

    print(f"Evaluation complete. Artifacts saved in {Config.artifacts_dir}")
    for row in results:
        bt = row["backtest"]
        print(
            f"{row['ticker']}: AUC={row['auc']:.4f} | "
            f"Return={bt['Total Return (strategy)']:.2%} | "
            f"Sharpe={bt['Sharpe (daily)']:.2f} | "
            f"MaxDD={bt['Max Drawdown']:.2%}"
        )


if __name__ == "__main__":
    main()
