import numpy as np
import pandas as pd


class Helper:

    def array_equal(value, expected, decimals):
        def arr_prepare(arr):
            arr = np.array(arr)
            for i in reversed(range(np.shape(arr)[1] - 1)):
                arr = arr[arr[:, i].argsort(kind='mergesort')]

            return np.around(arr, decimals)

        value = arr_prepare(value)
        expected = arr_prepare(expected)

        pd.DataFrame(value).to_excel('val.xlsx')
        pd.DataFrame(expected).to_excel('expected.xlsx')

        return np.array_equal(value, expected)

    def read_excel(file, sheet):
        return pd.read_excel(file, index_col=0, sheet_name=sheet)
