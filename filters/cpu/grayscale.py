def apply(source_array):
    result_array = source_array.copy()
    for i in range(source_array.shape[0]):
        for j in range(source_array.shape[1]):
            x, y, z = source_array[i, j]
            intensity = int(0.2126 * x + 0.7152 * y + 0.0722 * z)
            result_array[i, j] = (intensity,) * 3

    return result_array
