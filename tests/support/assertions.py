import numpy as np


def array_equal(value, expected, decimals):
    """Compare numeric arrays and tuple-keyed solution maps after normalization."""

    def arr_prepare(arr):
        if isinstance(arr, dict):
            arr = [list(key) for key in arr]

        arr = np.array(arr)
        arr = np.around(arr, decimals)

        for i in reversed(range(np.shape(arr)[1] - 1)):
            arr = arr[arr[:, i].argsort(kind="mergesort")]

        return arr

    value = arr_prepare(value)
    expected = arr_prepare(expected)

    return np.array_equal(value, expected)


def read_reference_csv(dataset_dir, sheet: str):
    """Load a bundled CSV reference table by logical sheet name."""
    with dataset_dir.joinpath(f"{sheet}.csv").open("r", encoding="utf-8") as handle:
        data = np.genfromtxt(handle, delimiter=",", skip_header=1, filling_values=0.0)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data[:, 1:]
