from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class MLPClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass
class NeuralSignalModel:
    hidden_size: int = 64
    dropout: float = 0.15
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 35
    batch_size: int = 128
    seed: int = 42
    device: str = "cpu"

    def __post_init__(self):
        self.scaler = StandardScaler()
        self.model: MLPClassifier | None = None

    def fit(self, X, y):
        X_np = np.asarray(X, dtype=np.float32)
        y_np = np.asarray(y, dtype=np.float32)

        X_scaled = self.scaler.fit_transform(X_np)
        X_t = torch.tensor(X_scaled, dtype=torch.float32)
        y_t = torch.tensor(y_np, dtype=torch.float32)

        torch.manual_seed(self.seed)

        self.model = MLPClassifier(
            in_dim=X_t.shape[1], hidden=self.hidden_size, dropout=self.dropout
        ).to(self.device)
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        criterion = nn.BCEWithLogitsLoss()

        ds = TensorDataset(X_t, y_t)
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True, drop_last=False)

        self.model.train()
        for _ in range(self.epochs):
            for xb, yb in dl:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                optimizer.zero_grad()
                logits = self.model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

        return self

    def predict_proba(self, X):
        if self.model is None:
            raise ValueError("Model has not been fit")

        X_np = np.asarray(X, dtype=np.float32)
        X_scaled = self.scaler.transform(X_np)
        X_t = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_t)
            p1 = torch.sigmoid(logits).cpu().numpy()

        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def predict(self, X):
        p1 = self.predict_proba(X)[:, 1]
        return (p1 >= 0.5).astype(int)
