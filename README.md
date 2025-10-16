# Neural Trading Engine (baseline)
Backend-only prototype for a quant engine. 
- Data: yfinance OHLCV
- Features: TA indicators
- Models: LogisticRegression / RandomForest (next: RL)
- Backtest: long/flat, no leverage

> For research/education only. Not financial advice.

## Quickstart
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Train on AAPL, 2017–2024
python train.py --tickers AAPL --start 2017-01-01 --end 2024-12-31

# Evaluate + backtest (plots saved to artifacts/)
python evaluate.py --tickers AAPL --start 2019-01-01 --end 2024-12-31 --model rf
