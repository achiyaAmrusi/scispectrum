import numpy as np
from pyspectrum.background.base import BackgroundEstimator
from pyspectrum.core.spectrum import Spectrum
from .snip_utils import ll_transform, inv_ll_transform
from pyspectrum.utils.smoothing import adaptive_gaussian_smoothing
from typing import Callable


class SASNIPBackground(BackgroundEstimator):
    """
    Step-Approximation SNIP (SASNIP) background estimator.

    Adaptive clipping window based on detector FWHM calibration.
    Iteration stops when background area converges.
    """

    def __init__(
            self,
            fwhm_calibration: Callable[[np.ndarray], np.ndarray],
            *,
            t_initial: float = 1.0,
            max_outer_iterations: int = 20,
            area_tol: float = 5e-3,
            min_window: int = 1,
    ):
        """
        Parameters
        ----------
        fwhm_calibration : callable
            Function FWHM(x) returning resolution in axis units.
            Required for adaptive SNIP.

        t_initial : float
            Initial proportionality factor between FWHM and clipping window.

        max_outer_iterations : int
            Maximum number of step-approximation iterations.

        area_tol : float
            Relative background area convergence threshold.

        min_window : int
            Minimum clipping window in channels.
        """
        if fwhm_calibration is None:
            raise ValueError("fwhm_calibration callable must be provided.")

        self.fwhm_calibration = fwhm_calibration
        self.t_initial = t_initial
        self.max_outer_iterations = max_outer_iterations
        self.area_tol = area_tol
        self.min_window = min_window

    def estimate(self, axis, counts) -> np.ndarray:
        """
        Estimate background using SASNIP.

        Parameters
        ----------
        axis : array_like
            Axis values.
        count : array_like
            Spectrum counts.
        smooth : bool
            If True, apply resolution-adaptive Gaussian smoothing before SASNIP.

        Returns
        -------
        bg : ndarray
            Estimated background.
        """
        bg_prev = counts
        area_prev = np.trapz(bg_prev, axis)
        t = self.t_initial

        for _ in range(self.max_outer_iterations):
            bg_new = self._adaptive_snip(axis, bg_prev, t)
            area_new = np.trapz(bg_new, axis)

            rel_diff = np.abs(area_new - area_prev) / max(area_prev, 1.0)
            if rel_diff < self.area_tol:
                break

            bg_prev = bg_new
            area_prev = area_new

        return bg_new

    def _adaptive_snip(self, axis: np.ndarray, counts: np.ndarray, t: float, smooth=True) -> np.ndarray:
        """
        Single adaptive SNIP step on log-log transformed data.

        Parameters
        ----------
        axis : array_like
            Axis values.
        counts : array_like
            Spectrum counts.
        t : float
            Scaling factor for clipping window.

        Returns
        -------
        bg : ndarray
            Background after single SNIP step.
        """
        z = ll_transform(counts)
        n = len(z)
        z_new = z.copy()

        if smooth:
            if (self.fwhm_calibration is None):
                raise ValueError("Resolution callable must be provided for smoothing.")
            z = adaptive_gaussian_smoothing(axis, z, resolution=self.fwhm_calibration)

        for i in range(n):
            fwhm = self.fwhm_calibration(axis[i])
            m = max(int(round(t * fwhm)), self.min_window)

            if i - m < 0 or i + m >= n:
                continue

            z_new[i] = min(z[i], 0.5 * (z[i - m] + z[i + m]))

        return inv_ll_transform(z_new)
