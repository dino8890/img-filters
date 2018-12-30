import math

import numpy as np
import pycuda.autoinit
from pycuda import driver
from pycuda import compiler

DIM_BLOCK = 32


def _nearest_multiple(a, b):
    return int(math.ceil(a / b) * b)


def _largest_factors(x):
    i = x - 1
    while x % i != 0:
        i -= 1

    return int(i), int(x / i)


def apply(source_array):
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

    mod = compiler.SourceModule(open('filters/gpu/grayscale.cu').read())
    apply_filter = mod.get_function('apply_filter')

    apply_filter(
        driver.InOut(red_channel),
        driver.InOut(green_channel),
        driver.InOut(blue_channel),
        np.uint32(width),
        np.uint32(height),
        block=(DIM_BLOCK, DIM_BLOCK, 1),
        grid=(dim_grid_x, dim_grid_y)
    )

    result_array[:, :, 0] = red_channel
    result_array[:, :, 1] = green_channel
    result_array[:, :, 2] = blue_channel

    return result_array
