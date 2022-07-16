import numpy as np


def compute_accuracy(
    y_true: np.ndarray, y_pred: np.ndarray, atol: float = 1e-3, rtol: float = 0.05
):
    """Compute accuracy of predictions"""

    closeness_to_truth = np.isclose(
        y_true, y_pred, rtol=rtol, atol=atol, equal_nan=True
    )
    accuracy = np.mean(closeness_to_truth)

    return accuracy
