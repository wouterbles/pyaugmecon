import numpy as np


class Helper:

    def array_equal(value, expected, decimals):
        def arrPrepare(arr):
            return np.around(arr[np.argsort(arr[:, 1])], decimals)

        value = arrPrepare(value)
        expected = arrPrepare(expected)

        return np.array_equal(value, expected)
