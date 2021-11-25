from typing import List, Tuple

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import differential_evolution
from scipy.spatial.transform import Rotation

from src.image_parser import MINCParser
from src.mask import GradientThresholdMasker
from src.preprocessing import GradientComputer
from src.similarity import GradientsOrientation
from src.typing import Arrays


def create_rotation_matrix(angles: Tuple[float, float, float]) -> np.ndarray:
    return Rotation.from_euler('xyz', angles).as_matrix()


def apply_inverse_transform(
    rotation: np.ndarray, translation: np.ndarray, points: np.ndarray
) -> np.ndarray:
    return np.matmul(rotation.T, (points-translation).T).T


def interpolate_gradient(
    image_space: Arrays, gradient: np.ndarray, points: np.ndarray
) -> np.ndarray:
    interpolator = RegularGridInterpolator(
        image_space, gradient, bounds_error=False,
        fill_value=0.0, method='linear'
    )
    return interpolator(points)


def run(fixed_image_name, moving_image_name):
    fixed_image_parser = MINCParser(fixed_image_name)
    moving_image_parser = MINCParser(moving_image_name)
    fixed_array = fixed_image_parser.get_array()
    moving_array = moving_image_parser.get_array()
    fixed_gradients = GradientComputer(fixed_array).compute(2.0)
    moving_gradients = GradientComputer(moving_array).compute(2.0)
    mask_generator = GradientThresholdMasker(fixed_gradients, 20.0)
    masked_fixed_gradients = mask_generator.apply(fixed_gradients)
    fixed_mask_indices = mask_generator.mask_indices
    mask_position = fixed_image_parser.compute_positions(fixed_mask_indices)
    moving_space = moving_image_parser.create_image_space()

    def evaluate(args: List[float]) -> float:
        print(args)
        angles = args[0], args[1], args[2]
        translations = args[3], args[4], args[5]
        rotation_matrix = create_rotation_matrix(angles)
        trans = np.array(translations)
        moving_mask_position = apply_inverse_transform(
            rotation_matrix, trans, mask_position
        )
        masked_moving_gradients = np.array([
            interpolate_gradient(moving_space, grad, moving_mask_position)
            for grad in moving_gradients
        ])
        masked_moving_gradients = np.matmul(
            rotation_matrix, masked_moving_gradients).T
        f = GradientsOrientation(
            masked_fixed_gradients, masked_moving_gradients
        ).compute()
        return -f
    bounds = 3*[(-np.pi/30, np.pi/30)] + 3*[(-5, 5)]
    res = differential_evolution(
        evaluate, bounds=bounds, maxiter=2, popsize=3, disp=True
    )


if __name__ == '__main__':
    run(
        '../data/datasets/BITE/group2/01/us.mnc',
        '../data/datasets/BITE/group2/01/mr.mnc'
    )
