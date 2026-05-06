import numpy as np
from scipy.special import erf
from pyspectrum.core.domain import Domain
from pyspectrum.domain_analysis.characterization.morphology import find_domain_peaks, domain_bases

def domain_erf_background(domain: Domain,
                          smooth=True,
                          smoothing_fwhm_scale=0.3,
                          prominence=None):
    """
    estimate the local bacgkround of a domain above SNR

    Note that this method works well for domain with 1 peak or dense multipeaks.
    If the peaks a sparse in space the method won't work well.

    Parameters
    ----------
    smooth
    domain

    Returns
    -------

    """
    # parameters for the background estimation
    if prominence is None:
        prominence = 0
    peaks, properties = find_domain_peaks(domain, smooth=smooth, smoothing_fwhm_scale=smoothing_fwhm_scale,
                                          prominence=prominence)
    prominences = properties["prominences"]
    weights = prominences / prominences.sum()
    resolutions = properties["fwhm"]/(2 * np.sqrt(2 * np.log(2)))
    local_axis = domain.spectrum.axis[domain.indices]
    height_left, height_right = domain_bases(domain)
    height_difference = height_left - height_right

    # Erf estimation for each peak
    normalized_erf = np.zeros((len(peaks), local_axis.shape[0]))
    for i, w, m, s in zip(range(len(peaks)), weights, peaks, resolutions):
        normalized_erf[i,:] = w*(erf(-(local_axis - m) / s) + 1)/2
    peak_baseline = min(height_right, height_left)

    return height_difference * normalized_erf.sum(axis=0) + peak_baseline

def domain_linear_background(domain: Domain):
    """
    estimate the local bacgkround of a domain above SNR

    Parameters
    ----------
    prominence
    domain

    Returns
    -------

    """
    # parameters for the background estimation
    local_axis = domain.spectrum.axis[domain.indices]
    height_left, height_right = domain_bases(domain)

    # linear estimation
    return height_left + (height_right - height_left) * (local_axis -local_axis[0])/(local_axis[-1]-local_axis[0])
