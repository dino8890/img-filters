import math

import numpy as np
import pycuda.autoinit
from pycuda import driver
from pycuda import compiler

DIM_BLOCK = 32
CUDA_KERNEL_CODE_PATH = 'filters/gpu/blur.cu'


def _nearest_multiple(a, b):
    return int(math.ceil(a / b) * b)


def _largest_factors(x):
    i = x - 1
    while x % i != 0:
        i -= 1

    return int(i), int(x / i)


def _create_gaussian_kernel(filter_width, standard_deviation):
    matrix = np.empty((filter_width, filter_width), np.float32)
    filter_half = filter_width // 2
    for i in range(-filter_half, filter_half + 1):
        for j in range(-filter_half, filter_half + 1):
            matrix[i + filter_half][j + filter_half] = (
                math.exp(-(i**2 + j**2) / (2 * standard_deviation**2))
                / (2 * math.pi * standard_deviation**2)
            )

    return matrix / matrix.sum()


def apply(source_array, standard_deviation, filter_width):
    result_array = np.empty_like(source_array)
    red_channel = source_array[:, :, 0].copy()
    green_channel = source_array[:, :, 1].copy()
    blue_channel = source_array[:, :, 2].copy()

    height, width = source_array.shape[:2]

    num_blocks = (_nearest_multiple(width, DIM_BLOCK)
                  * _nearest_multiple(height, DIM_BLOCK)
                  / DIM_BLOCK**2)

    max_num_blocks = (
            pycuda.autoinit.device.get_attribute(
                driver.device_attribute.MAX_GRID_DIM_X
            )
            * pycuda.autoinit.device.get_attribute(
                driver.device_attribute.MAX_GRID_DIM_Y
            )
    )

    if num_blocks > max_num_blocks:
        raise ValueError(
            'image dimensions too great, maximum block number exceeded'
        )

    dim_grid_x, dim_grid_y = _largest_factors(num_blocks)
    gaussian_kernel = _create_gaussian_kernel(filter_width, standard_deviation)

    mod = compiler.SourceModule(open(CUDA_KERNEL_CODE_PATH).read())
    apply_filter = mod.get_function('apply_filter')

    for channel in (red_channel, green_channel, blue_channel):
        apply_filter(
            driver.In(channel),
            driver.Out(channel),
            driver.In(np.full_like(channel, 255)),
            np.uint32(width),
            np.uint32(height),
            driver.In(gaussian_kernel),
            np.float32(standard_deviation),
            np.uint32(filter_width),
            block=(DIM_BLOCK, DIM_BLOCK, 1),
            grid=(dim_grid_x, dim_grid_y)
        )

    result_array[:, :, 0] = red_channel
    result_array[:, :, 1] = green_channel
    result_array[:, :, 2] = blue_channel

    return result_array
