import numpy as np
from scipy.signal import peak_widths
from pyspectrum.domain_analysis.find_peaks import find_domain_peaks

def count_peaks(domain, **kwargs):
    """
    Count peaks inside a Domain.

    Parameters
    ----------
    domain : Domain

    Returns
    -------
    int
    """
    peaks, _ = find_domain_peaks(domain, **kwargs)
    return len(peaks)


def peak_positions(domain, **kwargs):
    """
    Return peak positions in axis units.

    Returns
    -------
    ndarray
    """
    peaks, _ = find_domain_peaks(domain, **kwargs)

    return peaks


def peak_fwhm(domain, **kwargs):
    """
    Estimate FWHM of detected peaks using scipy.signal.peak_widths.

    Returns
    -------
    ndarray
        FWHM estimates in axis units.
    """

    peaks, properties = find_domain_peaks(domain, **kwargs)

    if len(peaks) == 0:
        return np.array([])

    y = domain.data.values
    x = domain.data.coords[domain.spectrum.axis_name].values
    dx = x[1] - x[0]

    peak_indices = np.searchsorted(x, peaks)

    widths, *_ = peak_widths(y, peak_indices, rel_height=0.5)

    return widths * dx

def domain_bases(domain):
    """
    Estimate the left and right bases height by averaging a resolution size of the sides

    Returns
    -------
    tuple
        The height if each side
    """

    # Local resolution
    local_axis = domain.spectrum.axis[domain.indices]
    local_resolution = domain.spectrum.resolution_calib(local_axis).mean()/(2 * np.sqrt(2 * np.log(2)))
    local_resolution_channel = max(int(local_resolution / (domain.spectrum.axis[1] - domain.spectrum.axis[0]) / 2), 3)

    #
    height_left = domain.data[:local_resolution_channel].mean().item()
    height_right = domain.data[-local_resolution_channel:-1].mean().item()

    return height_left, height_right