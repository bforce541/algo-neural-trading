from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

@dataclass
class BaselineModels:
    @staticmethod
    def logreg():
        return LogisticRegression(max_iter=2000, n_jobs=None)

    @staticmethod
    def rf(n_estimators=400, max_depth=None, random_state=42):
        return RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
        )
