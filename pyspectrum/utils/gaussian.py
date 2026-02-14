import numpy as np
from scipy.special import erf

GAUSSIAN_FWHM_TO_SIGMA = 1.0 / (2 * np.sqrt(2 * np.log(2)))
GAUSSIAN_SIGMA_TO_FWHM = 1.0 / GAUSSIAN_FWHM_TO_SIGMA

def fwhm_to_sigma(fwhm):
    return fwhm * GAUSSIAN_FWHM_TO_SIGMA

def sigma_to_fwhm(sigma):
    return sigma * GAUSSIAN_SIGMA_TO_FWHM

def gaussian(x, x0=0, sigma=1):
    """
    Gaussian function.

    Parameters
    ----------
    x: array-like
        Input values.
    x0: float
        Mean (center) of the Gaussian.
    sigma: float
        Standard deviation of the Gaussian

    Returns:
    array-like
        Gaussian values in x.
    """
    return np.exp(-(1/2)*((x-x0) / sigma**2))

def gaussian_cdf(x, x0=0, sigma=1):
    """
    Gaussian function.

    Parameters
    ----------
    x: array-like
        Input values.
    x0: float
        Mean (center) of the Gaussian.
    sigma: float
        Standard deviation of the Gaussian

    Returns:
    array-like
        Gaussian values in x.
    """
    return 0.5 * (1 + erf((x - x0) / (sigma * np.sqrt(2))))
