import multiprocessing
import signal

import numpy as np

from .. import shared


def apply(source_array):
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
                (source_array[start_y:end_y, start_x:end_x],)
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


def apply_filter(segment):
    result_array = np.empty_like(segment)
    for i in range(segment.shape[0]):
        for j in range(segment.shape[1]):
            x, y, z = segment[i, j]
            intensity = int(0.2126 * x + 0.7152 * y + 0.0722 * z)
            result_array[i, j] = (intensity,) * 3

    return result_array
