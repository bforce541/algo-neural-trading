from dataclasses import dataclass

@dataclass
class Config:
    data_dir: str = "data"
    artifacts_dir: str = "artifacts"
    lookahead_days: int = 1          # label = next-day return > 0
    test_split_date: str = "2023-01-01"
    rf_n_estimators: int = 400
    rf_max_depth: int | None = None
    random_state: int = 42
