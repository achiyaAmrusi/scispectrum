import numpy as np
import xarray as xr
from typing import Callable
from pyspectrum.utils.smoothing import resolution_adaptive_smoothing
from pyspectrum.background.base import BackgroundEstimator
from pyspectrum.core.spectrum import Spectrum
from .utils import ll_transform, inv_ll_transform


class SNIPBackground(BackgroundEstimator):
    """
    Classical SNIP background estimation (Ryan, 1988).
    """

    def __init__(self, iterations: int):
        self.iterations = int(iterations)

    def estimate(self, spectrum: Spectrum) -> xr.DataArray:
        """
        Perform SNIP background estimation on a Spectrum object.
        The spectrum is log-log smoothed before subtraction using the resolution calibration.

        Parameters
        ----------
        spectrum : Spectrum

        Returns
        -------
        bg : DataArray
            Estimated background counts.
        """
        y = spectrum.counts
        x = spectrum.axis
        background =  self.sinp(x,y, resolution=spectrum.resolution_calib)

        return xr.DataArray(background, coords=spectrum.coords)

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
            z = resolution_adaptive_smoothing(x, z, resolution=resolution)

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

