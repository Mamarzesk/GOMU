from typing import List

import numpy as np
from scipy.ndimage import gaussian_filter

from .typing import Arrays


class GradientComputer:
    def __init__(self, array: np.ndarray) -> None:
        self.array = array

    def compute(self, sigma: float) -> Arrays:
        dim: int = len(self.array.shape)
        array_range: float = np.max(self.array) - np.min(self.array)
        array = (self.array - np.min(self.array)) / array_range
        axes: List[List[int]] = np.identity(dim, dtype=int).tolist()
        return tuple(
            gaussian_filter(array, sigma, order=axis) for axis in axes
        )
