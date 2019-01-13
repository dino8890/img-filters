import multiprocessing
import signal

import numpy as np

from .. import shared


def apply(source_array, standard_deviation, filter_width):
    # spawned processes will ignore SIGINT
    sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    raise_sigint = False
    with multiprocessing.Pool(shared.get_cpu_count()) as pool:
        # revert to original SIGINT handler on main process
        signal.signal(signal.SIGINT, sigint_handler)
        results = []
        for s in shared.get_segments(source_array):
            start_x, end_x = s[0]
            start_y, end_y = s[1]
            result = pool.apply_async(
                apply_filter,
                (source_array,
                 ((start_x, end_x),
                  (start_y, end_y)),
                 standard_deviation,
                 filter_width)
            )
            results.append((result, (start_x, end_x), (start_y, end_y)))

        result_array = np.empty_like(source_array)
        for r, (start_x, end_x), (start_y, end_y) in results:
            try:
                segment = r.get()
                result_array[start_y:end_y, start_x:end_x] = segment
            except KeyboardInterrupt:
                pool.terminate()
                raise_sigint = True
                break

        pool.close()
        pool.join()

    if raise_sigint:
        raise KeyboardInterrupt  # catch it in filter.py

    return result_array


def apply_filter(source_array, segment, standard_deviation, filter_width):
    start_x, end_x = segment[0]
    start_y, end_y = segment[1]

    result_array = np.empty_like(source_array[start_y:end_y, start_x:end_x])
    kernel = shared.create_gaussian_kernel(filter_width, standard_deviation)

    filter_half = filter_width // 2
    for i in range(start_y, end_y):
        for j in range(start_x, end_x):
            blur_red = 0.0
            blur_green = 0.0
            blur_blue = 0.0
            for k in range(-filter_half, filter_half + 1):
                for l in range(-filter_half, filter_half + 1):
                    x = max(0, min(source_array.shape[1] - 1, j + l))
                    y = max(0, min(source_array.shape[0] - 1, i + k))
                    r, g, b = (source_array[y][x]
                               * kernel[k + filter_half][l + filter_half])

                    blur_red += r
                    blur_green += g
                    blur_blue += b

            result_array[i - start_y][j - start_x] = (
                blur_red,
                blur_green,
                blur_blue
            )

    return result_array
