import numpy as np
from typing import Callable
from pyspectrum.utils.smoothing import adaptive_gaussian_smoothing
from pyspectrum.background.base import BackgroundEstimator
from .snip_utils import ll_transform, inv_ll_transform


class SNIPBackground(BackgroundEstimator):
    """
    Classical SNIP background estimation (Ryan, 1988).

    Parameters
    ----------
    iterations : int
        Number of SNIP clipping iterations.
    resolution : callable
        Maps axis values to FWHM in the same units.
        Required when smooth=True (the default).
    smooth : bool
        If True, apply resolution-adaptive Gaussian smoothing before SNIP.
        Default is True.
    """

    def __init__(self, iterations: int, resolution: Callable[[np.ndarray], np.ndarray] = None, smooth: bool = True):
        self.iterations = int(iterations)
        self.resolution = resolution
        self.smooth = smooth

    def estimate(self, axis: np.ndarray, counts: np.ndarray) -> np.ndarray:
        """
        Estimate background using SNIP.

        Parameters
        ----------
        axis : np.ndarray
            Axis values.
        counts : np.ndarray
            Spectrum counts.

        Returns
        -------
        np.ndarray
            Estimated background, same shape as counts.
        """
        return self._sinp(axis, counts)

    def _sinp(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        if self.smooth:
            if self.resolution is None:
                raise ValueError("resolution must be provided at construction when smooth=True.")
            y = np.asarray(y, dtype=float)
            z = ll_transform(y)
            z = adaptive_gaussian_smoothing(x, z, resolution=self.resolution)
        else:
            y = np.asarray(y, dtype=float)
            z = ll_transform(y)

        n = len(z)
        for k in range(1, self.iterations + 1):
            z_new = z.copy()
            for i in range(k, n - k):
                z_new[i] = min(z[i], 0.5 * (z[i - k] + z[i + k]))
            z = z_new

        return inv_ll_transform(z)
