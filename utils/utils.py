import numpy as np


def to_sum_1(array: np.ndarray):
    partial = array / np.min(array[np.nonzero(array)])
    return partial / partial.sum()
