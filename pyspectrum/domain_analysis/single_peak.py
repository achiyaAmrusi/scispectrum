import numpy as np
from uncertainties.unumpy import nominal_values
from uncertainties import nominal_value
from pyspectrum.core import Domain
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
    of the peak position and value. This is a numerical helper, not a physical estimator."""
    i = data.argmax()
    if i == 0 or i == len(data) - 1:
        return coords[i], data[i]
    a, b, c = np.polyfit(coords[i-1:i+2], data[i-1:i+2], 2)
    center     = -b / (2 * a)
    peak_value = a * center**2 + b * center + c
    return center, peak_value


def fwhm_edges(single_peak_domain: Domain):
    """Find FWHM edges using parabolic maximum and linear interpolation at crossings."""
    data = _get_data(single_peak_domain)
    nominal_data = nominal_values(data)
    coords = data.coords[single_peak_domain.spectrum.axis_name].values

    _, peak_value = _parabolic_maximum(coords, nominal_data)
    half_max  = peak_value / 2

    left_idx  = np.where(nominal_data >= half_max)[0][0]
    right_idx = np.where(nominal_data >= half_max)[0][-1]
    left_edge  = _interpolate_crossing(coords, nominal_data, left_idx - 1, left_idx,      half_max)
    right_edge = _interpolate_crossing(coords, nominal_data, right_idx,    right_idx + 1, half_max)

    return left_edge, right_edge, half_max

def _get_data(single_peak_domain: Domain):
    if single_peak_domain.spectrum.data_with_errors is not None:
        return single_peak_domain.data_with_errors
    return single_peak_domain.data


def center_estimator(single_peak_domain: Domain):
    """Estimate peak center as weighted centroid within the FWHM region.
    Not suitable for noisy peaks — use a fit instead."""
    data = _get_data(single_peak_domain)
    nominal_data = nominal_values(data)
    coords = data.coords[single_peak_domain.spectrum.axis_name].values
    daxis  = coords[1] - coords[0]

    left_edge, right_edge, half_max = fwhm_edges(single_peak_domain)
    parabolic_center, _             = _parabolic_maximum(coords, nominal_data)

    fwhm_mask   = (coords >= left_edge) & (coords <= right_edge)
    fwhm_coords = np.concatenate([[left_edge], coords[fwhm_mask], [right_edge]])
    fwhm_data   = np.concatenate([[half_max],  data[fwhm_mask], [half_max]])
    weighted_center = ((fwhm_data * fwhm_coords).sum() / fwhm_data.sum())

    if abs(nominal_value(weighted_center) - parabolic_center) / daxis > 2:
        warnings.warn("Peak center estimation may be unreliable — peak might be noisy or asymmetric. "
                      "Consider using a Gaussian fit instead.")

    return weighted_center

def fwhm_estimator(single_peak_domain: Domain):
    """Estimate FWHM via interpolated half-maximum crossings.
    Not suitable for noisy peaks — use a fit instead."""
    left_edge, right_edge, _= fwhm_edges(single_peak_domain)
    return right_edge - left_edge

def sum_under(single_peak_domain: Domain, left_edge: float, right_edge: float):
    """Sum counts between left_edge and right_edge, including fractional edge channels."""
    data   = _get_data(single_peak_domain)
    coords = data.coords[single_peak_domain.spectrum.axis_name].values
    daxis  = coords[1] - coords[0]

    fwhm_mask    = (coords > left_edge) & (coords < right_edge)
    interior_sum = data[fwhm_mask].sum()

    left_channel   = data[(coords >= left_edge)][0]
    right_channel  = data[(coords <= right_edge)][-1]
    left_fraction  = (coords[np.where(coords >= left_edge)[0][0]]  - left_edge)  / daxis
    right_fraction = (right_edge - coords[np.where(coords <= right_edge)[0][-1]]) / daxis

    return (interior_sum + left_fraction * left_channel + right_fraction * right_channel).item()


def sum_under_fwhm(single_peak_domain: Domain):
    """Sum counts within the FWHM region. Not suitable for noisy peaks — use a fit instead."""
    left_edge, right_edge, _ = fwhm_edges(single_peak_domain)
    return sum_under(single_peak_domain, left_edge, right_edge)