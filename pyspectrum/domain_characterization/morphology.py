import numpy as np
from scipy.signal import find_peaks, peak_widths, find_peaks_cwt
from pyspectrum.utils.smoothing import gaussian_smoothing
from copy import copy

def find_domain_peaks(
    domain,
    *,
    smooth=True,
    smoothing_fwhm_scale=0.3,
    prominence=None,
):
    """
    Detect peaks within a Domain and return properties in axis units.
    This function is a convenience wrapper around ``scipy.signal.find_peaks``.
    Note that after smoothing the fwhm and other properties may alter.
    Parameters
    ----------
    domain : Domain
        Domain to analyze.

    smooth : bool
        Apply Gaussian smoothing before detection.

    smoothing_fwhm_scale : float
        Scaling applied to detector resolution during smoothing.
        Default 0.3
        Warning: Large values of smoothing_fwhm_scale cause changes in the peaks properties
    prominence : float, optional
        Minimum peak prominence.

    Returns
    -------
    peak_positions : ndarray
        Peak locations in axis units.

    properties : dict
        Peak properties expressed in axis units when applicable.
    """

    da = domain.data
    axis_name = domain.spectrum.axis_name

    x = da.coords[axis_name].values
    y = da.values

    dx = x[1] - x[0]

    if smooth:
        y = gaussian_smoothing(x, y, fwhm=smoothing_fwhm_scale * domain.local_resolution)

    kwargs = {}
    if prominence is not None:
        kwargs["prominence"] = prominence

    peaks, properties = find_peaks(y, **kwargs)

    # --- Convert peak indices to axis units ---
    peak_positions = x[peaks]

    # --- Convert index-based properties to axis units ---
    converted = copy(properties)

    if "left_bases" in properties:
        converted["left_bases"] = x[properties["left_bases"]]

    if "right_bases" in properties:
        converted["right_bases"] = x[properties["right_bases"]]

    # --- Peak widths ---
    if len(peaks) > 0:
        widths, height, left_ips, right_ips = peak_widths(y, peaks, rel_height=0.5)

        converted["fwhm"] = widths * dx
        converted["left_ips"] = x[0] + left_ips * dx
        converted["right_ips"] = x[0] + right_ips * dx

    return peak_positions, converted

def find_domain_peaks_cwt(
        domain,
        *,
        width_scale_factor=2.0,
        number_of_widths=50,
        max_distances=None,
        gap_thresh=None,
        min_length=None,
        min_snr=1,
        noise_perc=10,
        window_size=None,
):
    """
    Detect peaks within a Domain using a Continuous Wavelet Transform (CWT).

    This function is a convenience wrapper around ``scipy.signal.find_peaks_cwt``.
    It automatically derives suitable peak-width scales from the local
    resolution stored in the Domain object.

    The method is particularly useful for detecting peaks with unknown widths
    or partially overlapping peaks.

    Parameters
    ----------
    domain : Domain
        Domain to analyze.
        The minimum widthof cwt is take to be 10% of the local resolution, so make sure you local resolution is correct/

    width_scale_factor : float, default=2.0
        Maximum width scale relative to the detector resolution used when
        constructing the wavelet scales.If the peaks are much larger than the resolution increase it appropriately

    number_of_widths : int, default=50
        Number of wavelet widths sampled between the minimum and maximum width.

    max_distances : array-like, optional
        Maximum horizontal distance allowed when linking ridge points across
        adjacent wavelet scales. If None, defaults to ``widths / 4`` .

    gap_thresh : int, optional
        Maximum number of consecutive missing ridge points allowed when
        constructing ridge lines. If None, defaults to ``len(widths) / 4``.

    min_length : int, optional
        Minimum ridge length (number of scales) required for a peak to be
        accepted. If None, defaults to ``len(widths) / 3``.

    min_snr : float, default=1
        Minimum signal-to-noise ratio required for a ridge to be considered
        a peak.

    noise_perc : float, default=10
        Percentile used to estimate noise along each ridge.

    window_size : int, optional
        Window size used internally by the CWT peak detection algorithm
        for noise estimation.

    Returns
    -------
    peak_positions : ndarray or None
        Array of peak positions in axis units. Returns ``None`` if no peaks
        are detected.

    Notes
    -----
    The wavelet widths are automatically derived from the detector resolution
    of the domain:

    ``widths ≈ [0.2 × resolution, ..., width_scale_factor × resolution]``

    expressed in sample units.

    Smaller widths help resolve overlapping peaks, while larger widths
    stabilize ridge detection.

    This function is intended for domains containing a small number of peaks
    (e.g., isolated spectral regions).
    """

    da = domain.data
    axis_name = domain.spectrum.axis_name

    x = da.coords[axis_name].values
    y = da.values

    dx = x[1] - x[0]

    # Convert detector resolution to number of samples
    resolution_samples = domain.local_resolution / dx

    # Construct wavelet widths
    widths = np.linspace(
        0.2 * resolution_samples,
        width_scale_factor * resolution_samples,
        number_of_widths,
    )

    # Default ridge-linking parameters
    if max_distances is None:
        max_distances = widths / 4

    if gap_thresh is None:
        gap_thresh = int(len(widths) / 4)

    if min_length is None:
        min_length = int(len(widths) / 3)

    # Run CWT peak detection
    peaks = find_peaks_cwt(
        y,
        widths=widths,
        max_distances=max_distances,
        gap_thresh=gap_thresh,
        min_length=min_length,
        min_snr=min_snr,
        noise_perc=noise_perc,
        window_size=window_size,
    )

    if len(peaks) == 0:
        return None

    # Convert indices to axis units
    peak_positions = x[peaks]

    return peak_positions

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