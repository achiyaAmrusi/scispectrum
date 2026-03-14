import numpy as np
from scipy.ndimage import gaussian_filter1d
from pyspectrum.utils.gaussian import fwhm_to_sigma
from typing import Callable

def adaptive_gaussian_smoothing(
    x,
    y,
    *,
    resolution: Callable[[np.ndarray], np.ndarray] = None,
    fwhm_scale: float = 1.0,
):
    """
    Gaussian smoothing with resolution-adaptive width.

    Parameters
    ----------
    x : array_like
        x-axis values (must be sorted, uniformly sampled).

    y : array_like
        y values to smooth.

    resolution : Resolution, optional
        Function resolution(x) -> FWHM(x) in x-units.
        If None, sigma is set to one x-bin.

    fwhm_scale : float
        Scaling factor applied to FWHM (default = 1.0).

    Returns
    -------
    y_smooth : ndarray
        Smoothed y values.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # --- Bin width ---
    dx = x[1] - x[0]

    # --- Determine Gaussian sigma in bins ---
    if resolution is not None:
        fwhm_x = resolution(x) * fwhm_scale
        sigma_x = fwhm_to_sigma(fwhm_x)

        # Convert to bins (robust)
        sigma_bins = np.median(sigma_x / dx)

        # Physical minimum: one bin
        sigma_bins = max(sigma_bins, 1.0)
    else:
        raise ValueError("A resolution callable must be provided for adaptive smoothing.")

    # --- Gaussian smoothing ---
    y_smooth = gaussian_filter1d(
        y,
        sigma=sigma_bins,
        mode="nearest",
    )

    return y_smooth

def gaussian_smoothing(x, y, fwhm):
    """
    Gaussian smoothing with constant width.

    Parameters
    ----------
    x : ndarray
        Axis values (uniformly spaced).

    y : ndarray
        Signal values.

    fwhm : float
        Gaussian FWHM in x-units.

    Returns
    -------
    ndarray
    """
    dx = x[1] - x[0]

    sigma_x = fwhm_to_sigma(fwhm)
    sigma_bins = sigma_x / dx

    return gaussian_filter1d(y, sigma=sigma_bins, mode="nearest")
