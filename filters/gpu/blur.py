import math

import numpy as np
import pycuda.autoinit
from pycuda import driver
from pycuda import compiler

from .. import shared

CUDA_KERNEL_CODE_PATH = 'filters/gpu/blur.cu'


def apply(source_array, standard_deviation, filter_width):
    result_array = np.empty_like(source_array)
    red_channel = source_array[:, :, 0].copy()
    green_channel = source_array[:, :, 1].copy()
    blue_channel = source_array[:, :, 2].copy()

    height, width = source_array.shape[:2]

    dim_grid_x = math.ceil(width / shared.DIM_BLOCK)
    dim_grid_y = math.ceil(height / shared.DIM_BLOCK)

    max_num_blocks = (
            pycuda.autoinit.device.get_attribute(
                driver.device_attribute.MAX_GRID_DIM_X
            )
            * pycuda.autoinit.device.get_attribute(
                driver.device_attribute.MAX_GRID_DIM_Y
            )
    )

    if (dim_grid_x * dim_grid_y) > max_num_blocks:
        raise ValueError(
            'image dimensions too great, maximum block number exceeded'
        )

    gaussian_kernel = shared.create_gaussian_kernel(
        filter_width,
        standard_deviation
    )

    mod = compiler.SourceModule(open(CUDA_KERNEL_CODE_PATH).read())
    apply_filter = mod.get_function('applyFilter')

    for channel in (red_channel, green_channel, blue_channel):
        apply_filter(
            driver.In(channel),
            driver.Out(channel),
            np.uint32(width),
            np.uint32(height),
            driver.In(gaussian_kernel),
            np.uint32(filter_width),
            block=(shared.DIM_BLOCK, shared.DIM_BLOCK, 1),
            grid=(dim_grid_x, dim_grid_y)
        )

    result_array[:, :, 0] = red_channel
    result_array[:, :, 1] = green_channel
    result_array[:, :, 2] = blue_channel

    return result_array
