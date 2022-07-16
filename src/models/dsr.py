import numpy as np
from dso import DeepSymbolicRegressor

from src.models.base import BaseSymbolicModel
from ..utils import load_json


def get_dso_model(config_path: str) -> DeepSymbolicRegressor:
    """Creates DSR model from the config file"""

    config = load_json(config_path)
    model = DeepSymbolicRegressor(config)

    return model


class DSR(BaseSymbolicModel):
    def __init__(self, config_path: str) -> None:

        self.config_path = config_path
        self._model = self.init_model(config_path)

    def init_model(self, config_path: str) -> DeepSymbolicRegressor:
        """Creates DSR model from the config file"""

        return get_dso_model(config_path)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predicts the output for the given input."""

        return self._model.predict(x)

    def equation(self) -> str:
        if self._is_fit is False:
            raise ValueError("Model is not fitted yet")

        return self._model.program_.sympy_expr[0]

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """Trains the model on the given data."""

        self._model.fit(x, y)
        self._is_fit = True
