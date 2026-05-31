import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from pyspectrum.background.base import BackgroundEstimator


class ALSBackground(BackgroundEstimator):
    """
    Asymmetric Least Squares (ALS) background estimator.

    Estimates a smooth background by iteratively fitting a weighted
    smoothing spline, penalizing points above the current estimate
    more lightly than points below it. This asymmetry drives the
    baseline to sit beneath the signal peaks.

    Parameters
    ----------
    lam : float
        Smoothness penalty. Larger values produce a smoother background.
        Typical range: 1e3 to 1e7. Default is 1e5.
    p : float
        Asymmetry parameter. Controls the weight given to points above
        the current estimate. Should be small (e.g. 0.001 to 0.1) so
        the baseline stays below peaks. Default is 0.01.
    max_iter : int
        Number of reweighting iterations. More iterations refine the
        baseline but increase compute time. Default is 20.

    References
    ----------
    Eilers, P.H.C. and Boelens, H.F.M. (2005).
    "Baseline correction with asymmetric least squares smoothing."
    """

    def __init__(self, lam=1e5, p=0.01, max_iter=20):
        self.lam = lam
        self.p = p
        self.max_iter = max_iter

    def estimate(self, axis: np.ndarray, counts: np.ndarray) -> np.ndarray:
        """
        Estimate the background of a spectrum using ALS.

        Parameters
        ----------
        axis : np.ndarray
            Axis values (not used by ALS, required by the interface).
        counts : np.ndarray
            Spectrum counts.

        Returns
        -------
        np.ndarray
            Estimated background, same shape as counts.
        """
        y = counts
        n = len(y)

        # Second-order difference matrix (n-2 x n) — penalizes curvature
        # in the background estimate. D.T @ D appears in the smoothness term.
        D = sparse.diags([1, -2, 1], [0, 1, 2], shape=(n - 2, n), dtype=float)
        DTD = D.T @ D

        # Initialize weights uniformly — all points treated equally
        w = np.ones(n)

        for _ in range(self.max_iter):
            # Diagonal weight matrix for the current iteration
            W = sparse.diags(w, 0)

            # Solve the weighted penalized least squares system:
            # (W + lam * D'D) z = W y
            Z = W + self.lam * DTD
            z = spsolve(Z.tocsc(), w * y)

            # Asymmetric reweighting: points above the baseline get weight p,
            # points below get weight 1-p. This pulls the baseline downward
            # toward the true background, away from peaks.
            w = np.where(y > z, self.p, 1 - self.p)

        return z