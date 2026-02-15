"""Linear probe for high-stakes detection.

StandardScaler + LogisticRegression, matching the paper's configuration:
  - C=1e-3 (L2 regularization)
  - fit_intercept=False (redundant with StandardScaler centering)
  - solver='lbfgs', max_iter=1000
"""

import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class LinearProbe:
    """Linear probe with standardization."""

    def __init__(self, C: float = 1e-3, random_state: int = 42):
        self.scaler     = StandardScaler()
        self.classifier = LogisticRegression(
            C=C,
            random_state=random_state,
            max_iter=1000,
            solver="lbfgs",
            fit_intercept=False,
        )
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearProbe":
        X_scaled = self.scaler.fit_transform(X)
        self.classifier.fit(X_scaled, y)
        self._fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit() first")
        X_scaled = self.scaler.transform(X)
        return self.classifier.predict_proba(X_scaled)[:, 1]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def save(self, path: Path):
        # pickle is needed here: sklearn models don't support JSON serialization.
        # These files are only produced and consumed locally by our own code.
        with open(path, "wb") as f:
            pickle.dump({"scaler": self.scaler, "classifier": self.classifier}, f)

    @classmethod
    def load(cls, path: Path) -> "LinearProbe":
        # Only load pickle files we produced ourselves (local cache)
        probe = cls()
        with open(path, "rb") as f:
            data = pickle.load(f)
        probe.scaler     = data["scaler"]
        probe.classifier = data["classifier"]
        probe._fitted    = True
        return probe
