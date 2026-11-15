import numpy as np
from uncertainties.unumpy import nominal_values
from uncertainties import nominal_value, unumpy
from scispectrum.core import Domain
import warnings

def _interpolate_crossing(coords, data, i_low, i_high, threshold):
    """Linear interpolation between two channels to find exact threshold crossing.
    We use this instead of numpy interpolation because the peak might be noisy."""
    if i_low < 0 or i_high >= len(data):
        return coords[max(i_low, 0)]
    x0, x1 = coords[i_low], coords[i_high]
    y0, y1 = data[i_low], data[i_high]
    return x0 + (threshold - y0) * (x1 - x0) / (y1 - y0)


def _parabolic_maximum(coords, data):
    """Fit parabola through 3 channels around the maximum to sub-channel estimate
    of the peak position and value. This is a numerical helper, not a physical estimator.
    Closed form for any spacing (not polyfit), so ufloat data propagates."""
    i = nominal_values(data).argmax()
    if i == 0 or i == len(data) - 1:
        return coords[i], data[i]
    (x0, x1, x2), (y0, y1, y2) = coords[i-1:i+2], data[i-1:i+2]
    denom = (x0 - x1) * (x0 - x2) * (x1 - x2)
    a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / denom
    b = (x2**2 * (y0 - y1) + x1**2 * (y2 - y0) + x0**2 * (y1 - y2)) / denom
    c = (x1 * x2 * (x1 - x2) * y0 + x2 * x0 * (x2 - x0) * y1 + x0 * x1 * (x0 - x1) * y2) / denom
    center     = -b / (2 * a)
    peak_value = c - b**2 / (4 * a)
    return center, peak_value


def fwhm_edges(single_peak_domain: Domain):
    """Find FWHM edges using parabolic maximum and linear interpolation at crossings.
    With counts_err, half_max and the edges are ufloats that depend on the counts;
    only the choice of channels (maximum, crossings) is made on nominal values.

    Note: every access to data_with_errors creates new, independent ufloats, so the
    edges returned here are uncorrelated with the counts of a later, separate call
    (e.g. sum_under(domain, left_edge, right_edge)). Use sum_under_fwhm or
    center_estimator, which use one set of ufloats for edges and counts."""
    data = _get_data(single_peak_domain)
    return _fwhm_edges(data.coords[single_peak_domain.spectrum.axis_name].values, data.values)


def _fwhm_edges(coords, values):
    """fwhm_edges on given coords and values (one set of ufloats shared with the caller)."""
    _, peak_value = _parabolic_maximum(coords, values)
    half_max  = peak_value / 2

    above     = np.where(nominal_values(values) >= nominal_value(half_max))[0]
    left_idx  = above[0]
    right_idx = above[-1]
    left_edge  = _interpolate_crossing(coords, values, left_idx - 1, left_idx,      half_max)
    right_edge = _interpolate_crossing(coords, values, right_idx,    right_idx + 1, half_max)

    return left_edge, right_edge, half_max

def _get_data(single_peak_domain: Domain):
    if single_peak_domain.spectrum.counts_err is not None:
        return single_peak_domain.data_with_errors
    return single_peak_domain.data


def center_estimator(single_peak_domain: Domain, center_tolerance=1e-1):
    """
    Estimate the peak center as a weighted centroid within the FWHM region:
        center = integral(x * I) / integral(I)
    integrated over channels (trapezoid rule) from the left to the right FWHM
    edge, with the counts linearly interpolated at the edges (= half maximum).

    Parameters
    ----------
    single_peak_domain : Domain
        Spectral domain containing a single isolated peak.
        Must have a well-defined maximum and FWHM region.
    center_tolerance : float, optional
        Maximum acceptable fractional disagreement between the weighted
        centroid and the parabolic center, expressed as a fraction of
        the FWHM. Default is 0.1 (10% of FWHM).
        If the values don't disagree, issue a warning

    Returns
    -------
    float
        Estimated peak center position in the spectrum's axis units.

    Warns
    -----
    UserWarning
        If the weighted centroid and parabolic center disagree by more
        than `center_tolerance * FWHM`. This may indicate an asymmetric
        or noisy peak. Consider using a Gaussian fit instead.

    Notes
    -----
    - The parabolic center is estimated from the three points near
      the maximum and is less sensitive to the overall peak shape.
    """
    data   = _get_data(single_peak_domain)
    values = data.values
    coords = data.coords[single_peak_domain.spectrum.axis_name].values

    left_edge, right_edge, _ = _fwhm_edges(coords, values)
    parabolic_center, _      = _parabolic_maximum(coords, nominal_values(values))

    # interior coords are coords[first:last + 1] (chosen from the nominal edges); each
    # edge is interpolated between an interior coord and its outer neighbour
    first = np.searchsorted(coords, nominal_value(left_edge), side="right")
    last  = np.searchsorted(coords, nominal_value(right_edge), side="left") - 1

    left_frac  = (left_edge - coords[first - 1]) / (coords[first] - coords[first - 1])
    right_frac = (right_edge - coords[last]) / (coords[last + 1] - coords[last])
    left_value  = values[first - 1] + left_frac * (values[first] - values[first - 1])
    right_value = values[last] + right_frac * (values[last + 1] - values[last])

    channels = np.array([first - 1 + left_frac, *range(first, last + 1), last + right_frac])
    axis     = np.array([left_edge, *coords[first:last + 1], right_edge])
    counts   = np.array([left_value, *values[first:last + 1], right_value])
    weighted_center = np.trapezoid(axis * counts, channels) / np.trapezoid(counts, channels)

    fwhm = nominal_value(right_edge - left_edge)
    tolerance = center_tolerance * fwhm  # warn if centers disagree by more than 10% of FWHM
    if not np.isclose(
            nominal_value(weighted_center),
            parabolic_center,
            atol=tolerance,
            rtol=0):  # purely absolute — tolerance already scaled to FWHM
        warnings.warn(
            f"Weighted centroid and parabolic center disagree by more than "
            f"{tolerance:.2f} — peak may be asymmetric or noisy. "
            f"Consider using a Gaussian fit instead."
        )

    return weighted_center

def accurate_center_estimator(single_peak_domain: Domain, top_fraction=0.3):
    """
    Estimate the peak center from a parabola fitted to the log of the
    background-subtracted counts near the top of the peak:
        ln I(x) = ln A - (x - center)^2 / (2 sigma^2)
    This is exact for a Gaussian, so (unlike a parabola on I itself) using more
    channels does not bias the center, it only lowers its error.

    Parameters
    ----------
    single_peak_domain : Domain
        Domain containing a single isolated, Gaussian-like peak, with a background
        attached (``domain.subtract_background(...)``). The background is not
        estimated here; an unsubtracted background biases the result.
    top_fraction : float, optional
        The fit uses the contiguous channels around the maximum whose counts are
        at least ``top_fraction * max``. Default is 0.3; a narrower window
        (e.g. 0.8) gives a larger error, a wider one relies more on the tails,
        where the background subtraction matters most.

    Returns
    -------
    float or ufloat
        Estimated peak center in the spectrum's axis units; a ufloat if the
        spectrum has counts_err (including the background's error, if any).

    Raises
    ------
    ValueError
        If the domain has no background, the maximum is not positive, or fewer
        than 3 channels are above ``top_fraction * max``.
    """
    if single_peak_domain.background is None:
        raise ValueError("accurate_center_estimator requires a domain with a background; "
                         "attach one with domain.subtract_background(...)")

    data    = _get_data(single_peak_domain)
    values  = data.values
    coords  = data.coords[single_peak_domain.spectrum.axis_name].values
    nominal = nominal_values(values)

    # window: contiguous run around the maximum above top_fraction * max (nominal choice)
    peak = nominal.argmax()
    if nominal[peak] <= 0:
        raise ValueError("The background-subtracted maximum is not positive")
    above = nominal >= top_fraction * nominal[peak]
    lo, hi = peak, peak
    while lo > 0 and above[lo - 1]:
        lo -= 1
    while hi < len(nominal) - 1 and above[hi + 1]:
        hi += 1
    if hi - lo + 1 < 3:
        raise ValueError(f"Fewer than 3 channels above top_fraction * max ({top_fraction}); "
                         f"lower top_fraction or use center_estimator")

    # least squares ln I = a u^2 + b u + c, with u = x - x_peak for conditioning;
    # the solution is linear in ln I, so ufloats pass through
    u = coords[lo:hi + 1] - coords[peak]
    log = unumpy.log if values.dtype == object else np.log
    a, b, _ = np.linalg.pinv(np.vstack([u**2, u, np.ones_like(u)]).T) @ log(values[lo:hi + 1])
    return coords[peak] - b / (2 * a)

def fwhm_estimator(single_peak_domain: Domain):
    """Estimate FWHM via interpolated half-maximum crossings.
    Not suitable for noisy peaks — use a fit instead."""
    left_edge, right_edge, _= fwhm_edges(single_peak_domain)
    return right_edge - left_edge

def sum_under(single_peak_domain: Domain, left_edge: float, right_edge: float):
    """Sum counts between left_edge and right_edge:
        integral over channels of the linearly interpolated counts, from left_edge to right_edge
    (trapezoid rule over the edges and the coords strictly between them).

    The edges are given in axis units and converted to fractional channels, each
    within its own interval, so the result is in counts for any axis calibration.
    Which coords are interior is chosen from the nominal edges; the edges themselves
    enter the arithmetic, so if they are ufloats their uncertainty (the window's)
    propagates into the sum.

    Note: the counts here are new ufloats, independent of the ones the edges were
    computed from (see fwhm_edges), so the correlation between them is lost. For the
    FWHM window use sum_under_fwhm, which keeps it.
    """
    data = _get_data(single_peak_domain)
    return _sum_under(data.coords[single_peak_domain.spectrum.axis_name].values,
                      data.values, left_edge, right_edge)


def _sum_under(coords, values, left_edge, right_edge):
    """sum_under on given coords and values (one set of ufloats shared with the caller)."""
    # interior coords are coords[first:last + 1]; each edge lies between an interior
    # coord and its outer neighbour, and is interpolated between those two
    first = np.searchsorted(coords, nominal_value(left_edge), side="right")
    last  = np.searchsorted(coords, nominal_value(right_edge), side="left") - 1

    left_frac  = (left_edge - coords[first - 1]) / (coords[first] - coords[first - 1])
    right_frac = (right_edge - coords[last]) / (coords[last + 1] - coords[last])
    left_value  = values[first - 1] + left_frac * (values[first] - values[first - 1])
    right_value = values[last] + right_frac * (values[last + 1] - values[last])

    channels = np.array([first - 1 + left_frac, *range(first, last + 1), last + right_frac])
    counts   = np.array([left_value, *values[first:last + 1], right_value])
    return np.trapezoid(counts, channels)


def sum_under_fwhm(single_peak_domain: Domain):
    """Sum counts within the FWHM region. Not suitable for noisy peaks — use a fit instead."""
    data   = _get_data(single_peak_domain)
    coords = data.coords[single_peak_domain.spectrum.axis_name].values
    left_edge, right_edge, _ = _fwhm_edges(coords, data.values)
    return _sum_under(coords, data.values, left_edge, right_edge)