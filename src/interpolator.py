from abc import ABC, abstractmethod

import numpy as np


class AbstractInterpolator(ABC):
    @abstractmethod
    def __init__(self) -> None:
        ...

    def interpolate(self):
        ...


class NearestInterpolator(AbstractInterpolator):
    def __init__(
        self, array: np.ndarray, start: np.ndarray,
        end: np.ndarray, size: np.ndarray
    ) -> None:
        self.array = array
        self.start = start
        self.end = end
        self.size = size

    def interpolate(self, points: np.ndarray) -> np.ndarray:
        normalized_points = (points-self.start) / (self.end-self.start)
        indices: np.ndarray = np.round(self.size * normalized_points)
        indices = np.transpose(indices.astype(int))
        for idx in points.shape[1]:
            indices[idx][indices[idx] < self.start[idx]] = self.start[idx]
            indices[idx][indices[idx] > self.end[idx]] = self.end[idx]
