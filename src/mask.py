from abc import ABC, abstractmethod

import numpy as np

from .typing import Arrays


class AbstractMaskGenerator(ABC):
    @abstractmethod
    def __init__(self) -> None:
        ...

    @abstractmethod
    def create(self):
        ...

    @abstractmethod
    def apply(self):
        ...


class GradientThresholdMaskGenerator(AbstractMaskGenerator):
    def __init__(self, gradients: Arrays, threshold: float) -> None:
        self.gradients = gradients
        self.masking_base = self.compute_magnitude(self.gradients)
        self.threshold = threshold
        self.mask = self.create()
        self.mask_indices: Arrays = tuple(reversed(np.where(self.mask)))

    def create(self) -> np.ndarray:
        non_zeros: np.ndarray = self.masking_base[self.masking_base > 0]
        threshold: float = np.percentile(non_zeros, 100.0 - self.threshold)
        return self.masking_base > threshold

    def apply(self, arrays: Arrays) -> np.ndarray:
        return np.array([array[self.mask] for array in arrays]).T

    @staticmethod
    def compute_magnitude(arrays: Arrays) -> np.ndarray:
        return np.add.reduce(tuple(array ** 2 for array in arrays))
