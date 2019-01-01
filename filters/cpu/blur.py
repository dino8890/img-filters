import math

import numpy as np


def _create_gaussian_kernel(filter_width, standard_deviation):
    matrix = np.empty((filter_width, filter_width))
    filter_half = filter_width // 2
    for i in range(-filter_half, filter_half + 1):
        for j in range(-filter_half, filter_half + 1):
            matrix[i + filter_half][j + filter_half] = (
                math.exp(-(i**2 + j**2) / (2 * standard_deviation**2))
                / (2 * math.pi * standard_deviation**2)
            )

    return matrix / matrix.sum()


def apply(source_array, standard_deviation, filter_width):
    result_array = source_array.copy()

    kernel = _create_gaussian_kernel(filter_width, standard_deviation)
    filter_half = filter_width // 2
    for i in range(result_array.shape[0]):
        for j in range(result_array.shape[1]):
            blur_red = 0.0
            blur_green = 0.0
            blur_blue = 0.0
            for k in range(-filter_half, filter_half + 1):
                for l in range(-filter_half, filter_half + 1):
                    x = max(0, min(source_array.shape[0] - 1, i + k))
                    y = max(0, min(source_array.shape[1] - 1, j + l))
                    r, g, b = (source_array[x][y]
                               * kernel[k + filter_half][l + filter_half])

                    blur_red += r
                    blur_green += g
                    blur_blue += b

            result_array[i][j] = (blur_red, blur_green, blur_blue)

    return result_array
