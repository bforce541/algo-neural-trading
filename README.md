# Neural Trading Engine
A compact, research-grade neural + classical signal engine for daily equities.

## What this project now includes
- Data ingestion + parquet caching with `yfinance`
- Feature engineering (returns, trend, momentum, volatility, volume)
- Model families:
  - `logreg` (baseline)
  - `rf` (tree ensemble baseline)
  - `nn` (PyTorch MLP neural classifier)
- Time-safe train/test split (no overlap leakage)
- Multi-ticker train/eval workflows
- Realistic long/flat backtest with transaction costs + slippage
- Styled equity + drawdown chart output
- Styled HTML dashboard report for evaluation summaries

## Install
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train
```bash
python train.py --tickers AAPL MSFT NVDA --start 2017-01-01 --end 2024-12-31 --model nn
```

## Evaluate + backtest
```bash
python evaluate.py --tickers AAPL MSFT NVDA --start 2019-01-01 --end 2024-12-31 --model nn --threshold 0.52
```

## Build dashboard
```bash
python report.py --model nn
```

Generated artifacts are written to `artifacts/`:
- `model_<model>_<ticker>.joblib`
- `train_meta_<model>_<ticker>.json`
- `cls_report_<model>_<ticker>.json`
- `bt_stats_<model>_<ticker>.json`
- `equity_<model>_<ticker>.png`
- `eval_summary_<model>.json`
- `dashboard_<model>.html`

## Notes
- This is for research/education and not financial advice.
- Backtest assumes daily bars and a simplified execution model.
