import numpy as np

DIM_BLOCK = 32


def create_gaussian_kernel(filter_width, standard_deviation):
    matrix = np.empty((filter_width, filter_width), np.float32)
    filter_half = filter_width // 2
    for i in range(-filter_half, filter_half + 1):
        for j in range(-filter_half, filter_half + 1):
            matrix[i + filter_half][j + filter_half] = (
                np.exp(-(i**2 + j**2) / (2 * standard_deviation**2))
                / (2 * np.pi * standard_deviation**2)
            )

    return matrix / matrix.sum()
