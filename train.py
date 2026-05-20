from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score

from config import Config
from data import load_or_fetch
from features import make_features, train_test_split_time
from models import BaselineModels, NeuralSignalModel
from utils import ensure_dirs, set_global_seed


def build_model(model_name: str):
    if model_name == "logreg":
        return BaselineModels.logreg(seed=Config.seed)
    if model_name == "rf":
        return BaselineModels.rf(
            n_estimators=Config.rf_n_estimators,
            max_depth=Config.rf_max_depth,
            random_state=Config.seed,
        )
    if model_name == "nn":
        return NeuralSignalModel(
            hidden_size=Config.nn_hidden_size,
            dropout=Config.nn_dropout,
            lr=Config.nn_lr,
            weight_decay=Config.nn_weight_decay,
            epochs=Config.nn_epochs,
            batch_size=Config.nn_batch_size,
            seed=Config.seed,
        )
    raise ValueError("model_name must be one of {logreg, rf, nn}")


def train_single_ticker(model_name: str, ticker: str, df: pd.DataFrame) -> dict:
    feat_df = make_features(df)
    X_tr, y_tr, X_te, y_te, cols = train_test_split_time(feat_df, Config.test_split_date)

    model = build_model(model_name)
    model.fit(X_tr, y_tr)

    p_te = pd.Series(model.predict_proba(X_te)[:, 1], index=X_te.index)
    auc = float(roc_auc_score(y_te, p_te))

    model_fp = Path(Config.artifacts_dir) / f"model_{model_name}_{ticker}.joblib"
    joblib.dump(model, model_fp)

    meta = {
        "ticker": ticker,
        "model": model_name,
        "rows_train": int(len(X_tr)),
        "rows_test": int(len(X_te)),
        "features": cols,
        "auc_test": auc,
        "split_date": Config.test_split_date,
    }
    Path(Config.artifacts_dir, f"train_meta_{model_name}_{ticker}.json").write_text(
        json.dumps(meta, indent=2)
    )
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--model", choices=["logreg", "rf", "nn"], default="rf")
    args = ap.parse_args()

    set_global_seed(Config.seed)
    ensure_dirs(Config.artifacts_dir)

    raw = load_or_fetch(args.tickers, args.start, args.end)

    summary = []
    for ticker in args.tickers:
        meta = train_single_ticker(args.model, ticker, raw[ticker])
        summary.append(meta)

    out = Path(Config.artifacts_dir, f"train_summary_{args.model}.json")
    out.write_text(json.dumps(summary, indent=2))

    print(f"Saved training artifacts in {Config.artifacts_dir}")
    for row in summary:
        print(f"{row['ticker']}: test AUC={row['auc_test']:.4f} rows(train/test)=({row['rows_train']}/{row['rows_test']})")


if __name__ == "__main__":
    main()
