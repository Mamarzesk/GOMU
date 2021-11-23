from abc import ABC, abstractmethod

import numpy as np


class AbstractSimilarityComputer:
    @abstractmethod
    def __init__(self) -> None:
        ...

    @abstractmethod
    def compute(self):
        ...


class GradientsOrientationComputer(AbstractSimilarityComputer):
    def __init__(
        self, fixed_gradients: np.ndarray, moving_gradients: np.ndarray
    ) -> None:
        self.fixed_gradients = fixed_gradients
        self.moving_gradients = moving_gradients

    def compute(self) -> float:
        fixed_grads = self.normalize(self.fixed_gradients)
        moving_grads = self.normalize(self.moving_gradients)
        cosines = np.sum(np.multiply(fixed_grads, moving_grads), axis=1)
        return np.sum(cosines ** 2) / cosines.shape[0]

    @staticmethod
    def normalize(array: np.ndarray) -> np.ndarray:
        norms: np.ndarray = np.linalg.norm(array, axis=1, ord=None)
        tiled_norms: np.ndarray = np.tile(norms, (array.shape[1], 1)).T
        normalized_array = array / tiled_norms
        normalized_array = np.nan_to_num(normalized_array)
        return normalized_array
