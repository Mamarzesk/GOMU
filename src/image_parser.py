from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import SimpleITK as sitk

from .typing import Arrays


class AbstractImageParser(ABC):
    @abstractmethod
    def __init__(self) -> None:
        ...

    @abstractmethod
    def parse(self):
        ...


class MINCParser(AbstractImageParser):
    image_io = "MINCImageIO"

    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self.image = self.parse()

    def parse(self) -> sitk.Image:
        return sitk.ReadImage(self.file_path, imageIO=self.image_io)

    def create_image_space(self) -> Arrays:
        dim: int = self.image.GetDimension()
        origin: Tuple[float, ...] = self.image.GetOrigin()
        size: Tuple[int, ...] = self.image.GetSize()
        spacing: Tuple[float, ...] = self.image.GetSpacing()
        return tuple(
            np.linspace(origin[i], origin[i] + (size[i]-1)*spacing[i], size[i])
            for i in reversed(range(dim))
        )

    def indices_to_positions(self, indices: Arrays) -> Arrays:
        dim: int = self.image.GetDimension()
        origin: Tuple[float, ...] = self.image.GetOrigin()
        spacing: Tuple[float, ...] = self.image.GetSpacing()
        return np.transpose(np.array(tuple(
            origin[i] + indices[i]*spacing[i] for i in reversed(range(dim))
        )))
