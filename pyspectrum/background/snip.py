import numpy as np
import xarray as xr
from typing import Callable
from pyspectrum.utils.smoothing import adaptive_gaussian_smoothing
from pyspectrum.background.base import BackgroundEstimator
from .snip_utils import ll_transform, inv_ll_transform


class SNIPBackground(BackgroundEstimator):
    """
    Classical SNIP background estimation (Ryan, 1988).
    """

    def __init__(self, iterations: int):
        self.iterations = int(iterations)

    def estimate(self, axis, counts, resolution: Callable[[np.ndarray], np.ndarray], smoothing=False) -> xr.DataArray:
        """
        Perform SNIP background estimation for a spectrum.
        The spectrum is log-log smoothed before subtraction using the resolution calibration.

        Parameters
        ----------
        spectrum : Spectrum

        Returns
        -------
        bg : DataArray
            Estimated background counts.
        """
        background =  self.sinp(axis, counts, resolution=resolution)

        return background

    def sinp(self, x: np.ndarray, y: np.ndarray, resolution: Callable[[np.ndarray], np.ndarray] = None, smooth=True) -> np.ndarray:
        """
        Perform SNIP background estimation on a 1D spectrum.

        Parameters
        ----------
        x : array_like
            Axis values.
        y : array_like
            Spectrum counts.
        resolution : callable
            Resolution(x) -> FWHM in axis units.
        smooth : bool
            If True, apply resolution-adaptive Gaussian smoothing before SNIP.

        Returns
        -------
        bg : ndarray
            Estimated background counts.
        """
        y = np.asarray(y, dtype=float)
        z = ll_transform(y)

        if smooth:
            if resolution is None:
                raise ValueError("Resolution callable must be provided for smoothing.")
            z = adaptive_gaussian_smoothing(x, z, resolution=resolution)

        n = len(z)
        for k in range(1, self.iterations + 1):
            z_new = z.copy()
            for i in range(k, n - k):
                z_new[i] = min(
                    z[i],
                    0.5 * (z[i - k] + z[i + k]),
                )
            z = z_new

        return inv_ll_transform(z)

