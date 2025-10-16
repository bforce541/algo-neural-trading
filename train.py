import argparse, json
from pathlib import Path
import pandas as pd
from config import Config
from data import load_or_fetch
from features import make_features, train_test_split_time
from models.baseline import BaselineModels
from utils import ensure_dirs

def fit_and_save(model_name: str, X_tr, y_tr, out_fp: Path):
    if model_name == "logreg":
        model = BaselineModels.logreg()
    elif model_name == "rf":
        from config import Config
        model = BaselineModels.rf(Config.rf_n_estimators, Config.rf_max_depth, Config.random_state)
    else:
        raise ValueError("model_name must be one of {logreg, rf}")

    model.fit(X_tr, y_tr)
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    import joblib; joblib.dump(model, out_fp)
    return model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--model", choices=["logreg","rf"], default="rf")
    args = ap.parse_args()

    ensure_dirs(Config.artifacts_dir)
    data = load_or_fetch(args.tickers, args.start, args.end)

    # For now, single ticker training (extend to multi later)
    t = args.tickers[0]
    f = make_features(data[t])
    X_tr, y_tr, X_te, y_te, cols = train_test_split_time(f, Config.test_split_date)

    model_fp = Path(Config.artifacts_dir) / f"model_{args.model}_{t}.joblib"
    model = fit_and_save(args.model, X_tr, y_tr, model_fp)

    meta = {
        "ticker": t,
        "start": args.start,
        "end": args.end,
        "features": cols,
        "model": args.model,
        "rows_train": len(X_tr),
        "rows_test": len(X_te),
    }
    Path(Config.artifacts_dir, f"train_meta_{t}.json").write_text(json.dumps(meta, indent=2))
    print("Saved:", model_fp, "\nMeta:", meta)

if __name__ == "__main__":
    main()
