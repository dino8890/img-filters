import multiprocessing

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


def get_cpu_count():
    try:
        cpu_count = multiprocessing.cpu_count()
    except NotImplementedError:
        cpu_count = 1

    return cpu_count


def get_segments(source_array):
    cpu_count = get_cpu_count()
    per_process_x = source_array.shape[1] // cpu_count
    per_process_y = source_array.shape[0] // cpu_count

    segments = []
    for i in range(cpu_count):
        start_y = i * per_process_y
        if i == (cpu_count - 1):
            end_y = source_array.shape[0]
        else:
            end_y = start_y + per_process_y

        for j in range(cpu_count):
            start_x = j * per_process_x

            if j == (cpu_count - 1):
                end_x = source_array.shape[1]
            else:
                end_x = start_x + per_process_x

            segments.append(((start_x, end_x), (start_y, end_y)))

    return segments
