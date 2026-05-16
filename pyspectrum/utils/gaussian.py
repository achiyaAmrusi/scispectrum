import numpy as np

GAUSSIAN_FWHM_TO_SIGMA = 1.0 / (2 * np.sqrt(2 * np.log(2)))
GAUSSIAN_SIGMA_TO_FWHM = 1.0 / GAUSSIAN_FWHM_TO_SIGMA

def fwhm_to_sigma(fwhm):
    return fwhm * GAUSSIAN_FWHM_TO_SIGMA

def sigma_to_fwhm(sigma):
    return sigma * GAUSSIAN_SIGMA_TO_FWHM


