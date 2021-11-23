from abc import ABC, abstractmethod
from typing import List

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import gaussian_filter

from .typing import Arrays


class AbstractGradientComputer(ABC):
    @abstractmethod
    def __init__(self) -> None:
        ...

    @abstractmethod
    def compute(self):
        ...


class MINCGradientComputer(AbstractGradientComputer):
    def __init__(self, image: sitk.Image) -> None:
        self.image = image

    def compute(self, sigma: float) -> Arrays:
        array = sitk.GetArrayFromImage(self.image)
        dim: int = self.image.GetDimension()
        array_range: float = np.max(array) - np.min(array)
        array = (array - np.min(array)) / array_range
        axes: List[List[int]] = np.eye(dim, dtype=int).tolist()
        return tuple(gaussian_filter(array, sigma, order=axis) for axis in axes)
