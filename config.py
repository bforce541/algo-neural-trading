from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    data_dir: str = "data"
    artifacts_dir: str = "artifacts"

    lookahead_days: int = 1
    test_split_date: str = "2023-01-01"

    rf_n_estimators: int = 400
    rf_max_depth: int | None = None

    seed: int = 42

    # Neural model defaults
    nn_hidden_size: int = 64
    nn_dropout: float = 0.15
    nn_lr: float = 1e-3
    nn_weight_decay: float = 1e-4
    nn_epochs: int = 35
    nn_batch_size: int = 128

    # Backtest defaults
    trading_days_per_year: int = 252
    fee_bps: float = 5.0
    slippage_bps: float = 2.0
