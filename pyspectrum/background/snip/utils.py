import numpy as np


def ll_transform(y):
    return np.log(np.log(y + 1.0) + 1.0)


def inv_ll_transform(y):
    return np.exp(np.exp(y) - 1.0) - 1.0

