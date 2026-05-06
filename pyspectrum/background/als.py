import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from pyspectrum.background.base import BackgroundEstimator


class ALSBackground(BackgroundEstimator):
    def __init__(self, lam=1e5, p=0.01, max_iter=20):
        self.lam = lam
        self.p = p
        self.max_iter = max_iter

    def estimate(self, x, y):
        y = y.astype(float)
        n = len(y)

        # Second derivative matrix
        D = sparse.diags([1, -2, 1], [0, 1, 2], shape=(n - 2, n))
        DTD = D.T @ D

        w = np.ones(n)

        for _ in range(self.max_iter):
            W = sparse.diags(w, 0)

            Z = W + self.lam * DTD
            z = spsolve(Z, w * y)

            # asymmetry
            w = np.where(y > z, self.p, 1 - self.p)

        return z