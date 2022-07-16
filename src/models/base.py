from abc import ABC, abstractmethod

from ..metrics import compute_accuracy


class BaseSymbolicModel(ABC):
    def __init__(self) -> None:
        super().__init__()
        self._model = None
        self._is_fit = False

    @abstractmethod
    def init_model(*args, **kwargs):
        """Initializes the model."""
        raise NotImplementedError

    @abstractmethod
    def fit(self, x, y):
        """Trains the model on the given data."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, x):
        """Predicts the output of the model on the given data."""
        raise NotImplementedError

    @abstractmethod
    def equation(self) -> str:
        """Returns the equation of the model."""
        raise NotImplementedError

    def score(self, x, y):
        """Computes the accuracy of the model on the given data."""

        y_predict = self.predict(x)
        return compute_accuracy(y, y_predict)

    def __call__(self):
        return self
