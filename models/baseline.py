from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


class BaselineModels:
    @staticmethod
    def logreg(seed: int = 42) -> LogisticRegression:
        return LogisticRegression(max_iter=3000, solver="lbfgs", random_state=seed)

    @staticmethod
    def rf(
        n_estimators: int = 400,
        max_depth: int | None = None,
        random_state: int = 42,
    ) -> RandomForestClassifier:
        return RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
        )
