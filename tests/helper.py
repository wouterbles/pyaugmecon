import numpy as np
import pandas as pd

from pyaugmecon.helper import Helper as PyAugmeconHelper


class Helper:
    def array_equal(value: dict | list | np.ndarray, expected, decimals):
        def arr_prepare(arr):
            if type(arr) is dict:
                arr = PyAugmeconHelper.keys_to_list(arr)

            arr = np.array(arr)
            arr = np.around(arr, decimals)

            for i in reversed(range(np.shape(arr)[1] - 1)):
                arr = arr[arr[:, i].argsort(kind="mergesort")]

            return arr

        value = arr_prepare(value)
        expected = arr_prepare(expected)

        return np.array_equal(value, expected)

    def read_excel(file, sheet):
        return pd.read_excel(file, index_col=0, sheet_name=sheet)
