import numpy as np
from scipy.special import erf
from scispectrum.core.domain import Domain
from scispectrum.domain_analysis.morphology import find_domain_peaks, _base_width


class _BackgroundArray(np.ndarray):
    """Plain float background that also remembers how it was built from the domain
    values (``_model``), so ``Domain.data_with_errors`` can rebuild it from the same
    ufloats. Arrays derived from it do not keep the recipe."""
    def __array_finalize__(self, obj):
        self._model = None


def _with_model(domain, model):
    background = np.asarray(model(domain.data.values)).view(_BackgroundArray)
    background._model = model
    return background

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
    n = _base_width(domain)

    # Erf estimation for each peak
    normalized_erf = np.zeros((len(peaks), local_axis.shape[0]))
    for i, w, m, s in zip(range(len(peaks)), weights, peaks, resolutions):
        normalized_erf[i,:] = w*(erf(-(local_axis - m) / s) + 1)/2

    def model(values):
        height_left, height_right = values[:n].mean(), values[-n:].mean()
        height_difference = height_left - height_right
        # compare nominal values (ordering ufloats is deprecated in uncertainties)
        nominal = lambda h: getattr(h, "nominal_value", h)
        peak_baseline = height_right if nominal(height_right) <= nominal(height_left) else height_left
        return height_difference * normalized_erf.sum(axis=0) + peak_baseline

    return _with_model(domain, model)

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
    n = _base_width(domain)

    # linear estimation
    def model(values):
        height_left, height_right = values[:n].mean(), values[-n:].mean()
        return height_left + (height_right - height_left) * (local_axis -local_axis[0])/(local_axis[-1]-local_axis[0])

    return _with_model(domain, model)
