from gplearn.genetic import SymbolicRegressor
from gplearn.functions import make_function

import sympy
import numpy as np
from sympy import sympify

from src.models.base import BaseSymbolicModel


def exp_fn_gplearn(x):
    """Custom exponential function for gplearn"""

    with np.errstate(over="ignore"):
        return np.where(np.abs(x) < 100, np.exp(x), 0.0)


exp = make_function(function=exp_fn_gplearn, name="exp", arity=1)

FUNCTION_SET = ("add", "sub", "mul", "div", "sqrt", "log", exp, "inv", "sin", "cos")


def get_gplearn_model(*args, **kwargs) -> SymbolicRegressor:
    """Creates Genetic programming model"""

    model = SymbolicRegressor(*args, **kwargs)

    return model


class Gplearn(BaseSymbolicModel):

    _LOCALS = {
        "add": lambda x, y: x + y,
        "sub": lambda x, y: x - y,
        "mul": lambda x, y: x * y,
        "div": lambda x, y: x / y,
        "sqrt": lambda x: sympy.sqrt(x),
        "log": lambda x: sympy.log(x),
        "inv": lambda x: 1 / x,
        "sin": lambda x: sympy.sin(x),
        "cos": lambda x: sympy.cos(x),
        "neg": lambda x: -x,
        "pow": lambda x, y: x ** y,
        "cos": lambda x: sympy.cos(x),
    }

    def __init__(self, *args, **kwargs) -> None:

        self._model = self.init_model(*args, **kwargs)
        self.args = [args, kwargs]

    def init_model(self, *args, **kwargs):

        return get_gplearn_model(*args, **kwargs)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predicts the output for the given input."""

        return self._model.predict(x)

    def equation(self) -> str:
        if self._is_fit is False:
            raise ValueError("Model is not fitted yet")

        eq_string = self._model._program.__str__()

        return sympify(eq_string, locals=self._LOCALS)

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """Trains the model on the given data."""

        self._model.fit(x, y)
        self._is_fit = True
