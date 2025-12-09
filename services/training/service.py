"""Training Service - placeholder for task 6."""

from pathlib import Path
from typing import Any, Tuple

import numpy as np


class TrainingService:
    """Service for training ML models.
    
    Full implementation in task 6.
    """

    def load_data(self, data_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Load training data from storage."""
        raise NotImplementedError("Implement in task 6")

    def train_xgboost(self, X: np.ndarray, y: np.ndarray, params: dict[str, Any]) -> Any:
        """Train XGBoost model."""
        raise NotImplementedError("Implement in task 6")

    def train_pytorch(self, X: np.ndarray, y: np.ndarray, config: dict[str, Any]) -> Any:
        """Train PyTorch model."""
        raise NotImplementedError("Implement in task 6")

    def evaluate_model(
        self, model: Any, X_test: np.ndarray, y_test: np.ndarray
    ) -> dict[str, float]:
        """Evaluate model performance."""
        raise NotImplementedError("Implement in task 6")
