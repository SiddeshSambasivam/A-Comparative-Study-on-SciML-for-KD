import logging
from typing import List
import tempfile

from aifeynman import AIFeynmanRegressor

from src.models.base import BaseSymbolicModel


_OPERATIONS_MAP = {
    "add": "+",
    "sub": "-",
    "mul": "*",
    "div": "D",
    "neg": "~",
    "inv": "I",
    "log": "L",
    "exp": "E",
    "sin": "S",
    "cos": "C",
    "abs": "A",
    "arcsin": "N",
    "arctan": "T",
    "sqrt": "R",
    "pow": "P",
}


def convert_funcs_to_string(function_list: List[str]) -> str:

    funcs = []

    for func in function_list:
        if func in _OPERATIONS_MAP:
            funcs.append(_OPERATIONS_MAP[func])

    return "".join(funcs)


class AIFeynman(BaseSymbolicModel):
    def __init__(
        self,
        functions: str,
        BF_try_time: int = 60,
        polyfit_deg: int = 4,
        NN_epochs: int = 128,
        max_time: int = 2 * 60,
    ) -> None:

        self._is_fit = False

        self.function_list = functions.split(",")
        self.function_string = convert_funcs_to_string(self.function_list)

        self.file_handler = tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        )
        self.function_path = self.file_handler.name

        self.BF_try_time = BF_try_time
        self.polyfit_deg = polyfit_deg
        self.NN_epochs = NN_epochs
        self.max_time = max_time

        self._model = self.init_model()

    def __del__(self):
        """Deletes the temporary file created by the AIFeynmanRegressor constructor"""
        self.file_handler.close()

    def init_model(self):

        self.function_string = convert_funcs_to_string(self.function_list)
        with open(self.function_path, "w") as f:
            f.write(self.function_string)

        return AIFeynmanRegressor(
            BF_ops_file_type=self.function_path,
            BF_try_time=self.BF_try_time,
            polyfit_deg=self.polyfit_deg,
            NN_epochs=self.NN_epochs,
            max_time=self.max_time,
        )

    def fit(self, x, y):
        """Trains the model on the given data."""

        self._model.fit(x, y)
        self._is_fit = True

    def predict(self, x):
        """Predicts the output for the given input."""

        if self._is_fit is False:
            raise ValueError("Model is not fitted yet")

        return self._model.predict(x)

    def equation(self):
        if self._is_fit is False:
            raise ValueError("Model is not fitted yet")

        return self._model.best_model_
