import numpy as np


class Helper:

    def array_equal(value, expected, decimals):
        def arr_prepare(arr):
            return np.around(arr[np.argsort(arr[:, 1])], decimals)

        value = arr_prepare(value)
        expected = arr_prepare(expected)

        return np.array_equal(value, expected)
