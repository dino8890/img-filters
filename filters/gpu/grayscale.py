import math

import numpy as np
import pycuda.autoinit
from pycuda import driver
from pycuda import compiler

from .. import shared


def apply(source_array):
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

    mod = compiler.SourceModule(open('filters/gpu/grayscale.cu').read())
    apply_filter = mod.get_function('applyFilter')

    apply_filter(
        driver.InOut(red_channel),
        driver.InOut(green_channel),
        driver.InOut(blue_channel),
        np.uint32(width),
        np.uint32(height),
        block=(shared.DIM_BLOCK, shared.DIM_BLOCK, 1),
        grid=(dim_grid_x, dim_grid_y)
    )

    result_array[:, :, 0] = red_channel
    result_array[:, :, 1] = green_channel
    result_array[:, :, 2] = blue_channel

    return result_array
