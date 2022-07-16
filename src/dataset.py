import logging
from dataclasses import dataclass
from typing import Callable, List

import numpy as np
import pandas as pd
from sympy import lambdify
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class Equation:
    expr: str
    variables: set
    support: dict
    number_of_points: int
    code: Callable = None

    def __post_init__(self):
        self.code = lambdify([*self.variables], expr=self.expr)
        self.x: np.ndarray = None
        self.y: np.ndarray = None

    def __repr__(self) -> str:
        return f"Equation(expr={self.expr}, number_of_points={self.number_of_points})"


class Dataset:
    """
    A class for dataset of symbolic equations.

    Args:
        equations: List of equations.
    """

    def __init__(self, equations: List[Equation], noise: float = 0.0) -> None:
        self.equations = equations
        self.noise = noise

    @staticmethod
    def evaluate_func(eq: Equation, X: np.ndarray):
        return eq.code(*X.T)

    def _generate_data_pts(self, eq: Equation):
        """Generates data points for the given equation."""

        input_data = []
        for var in eq.variables:
            _max = eq.support[var]["max"]
            _min = eq.support[var]["min"]
            input_data.append(np.random.uniform(_min, _max, eq.number_of_points))

        x = np.stack(input_data, axis=1)

        return x

    def generate_data(self):
        """Generates data for each equation."""

        for equation in self.equations:
            x = self._generate_data_pts(equation)

            y = equation.code(*x.T)

            generated_noise = np.random.normal(0, self.noise, equation.number_of_points)
            y += generated_noise

            equation.x = x
            equation.y = y

    def __iter__(self):
        return iter(self.equations)

    def __getitem__(self, index: int) -> Equation:
        return self.equations[index]

    def __len__(self) -> int:
        return len(self.equations)


def load_equations_dataframe(path: str) -> pd.DataFrame:
    """
    Loads the equations from the given path and returns a pandas dataframe.
    """

    EXPECTED_COLUMNS = ["eq", "support", "num_points"]

    df = pd.read_csv(path)
    if not all(x in df.columns for x in EXPECTED_COLUMNS):
        raise ValueError(
            "dataframe not compliant with the format. Ensure that it has eq, support and num_points as column name"
        )

    df = df[["eq", "support", "num_points"]]

    return df


def create_equation(eq: str, support: str, num_points: int) -> Equation:

    supp = eval(support)
    variables = set(supp.keys())

    eq = Equation(
        expr=eq,
        variables=variables,
        support=supp,
        number_of_points=num_points,
    )

    return eq
