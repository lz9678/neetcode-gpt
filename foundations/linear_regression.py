import numpy as np
from numpy.typing import NDArray

class Solution:

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        # X is (n, m), weights is (m,) -> return (n,) predictions
        # Round to 5 decimal places
        
        if X.shape[1] == weights.shape[0]:
        # try:
            result = np.dot(X, weights)
            return np.round(result, 5)
        # except ValueError as e:
        else:
            raise ValueError(
                f"Dimension mismatch: Matrix X has {X.shape[1]} columns,"
                f"but vector Weights has {X.shape[0]} elements."
            )


    def get_error(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> float:
        # Compute mean squared error between predictions and ground truth
        # Round to 5 decimal places
        error = model_prediction - ground_truth
        se = np.sum(error**2)/len(error)
        # se = np.sum(np.square(error))
        # se = np.dot(error, error)
        return np.round(se, 5)


