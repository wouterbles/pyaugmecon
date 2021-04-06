import numpy as np
import pandas as pd


class Helper:

    def array_equal(value, expected, decimals):
        def arr_prepare(arr):
            return np.around(arr[np.argsort(arr[:, 1])], decimals)

        value = arr_prepare(value)
        expected = arr_prepare(expected)

        return np.array_equal(value, expected)

    def read_excel(file, sheet):
        return pd.read_excel(file, index_col=0, sheet_name=sheet)
