import argparse, json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score
from config import Config
from data import load_or_fetch
from features import make_features, train_test_split_time
from backtest import long_flat_backtest, save_plot
from utils import ensure_dirs
import joblib

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--model", choices=["logreg","rf"], default="rf")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    t = args.tickers[0]
    model_fp = Path(Config.artifacts_dir) / f"model_{args.model}_{t}.joblib"
    model = joblib.load(model_fp)

    data = load_or_fetch([t], args.start, args.end)[t]
    f = make_features(data)
    X_tr, y_tr, X_te, y_te, cols = train_test_split_time(f, Config.test_split_date)

    # Prob of "up"
    proba = pd.Series(model.predict_proba(X_te)[:,1], index=X_te.index, name="proba_up")

    # Classification metrics
    auc = roc_auc_score(y_te, proba)
    y_pred = (proba > args.threshold).astype(int)
    report = classification_report(y_te, y_pred, output_dict=True)

    # Backtest
    stats, equity, buyhold = long_flat_backtest(f.loc[X_te.index, "Close"], proba, args.threshold)

    ensure_dirs(Config.artifacts_dir)
    Path(Config.artifacts_dir, f"cls_report_{t}.json").write_text(json.dumps(report, indent=2))
    Path(Config.artifacts_dir, f"bt_stats_{t}.json").write_text(json.dumps(stats, indent=2))
    save_plot(equity, buyhold, f"{t} Strategy vs Buy&Hold", str(Path(Config.artifacts_dir, f"equity_{t}.png")))

    print("AUC:", round(auc, 4))
    print("Backtest:", stats)
    print("Artifacts saved in:", Config.artifacts_dir)

if __name__ == "__main__":
    main()
