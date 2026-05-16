import numpy as np


def centroid(domain):
    """
    First moment (center of mass).

    Returns
    -------
    float
    """

    da = domain.data

    x = da.coords[domain.spectrum.axis_name].values
    y = da.values

    total = np.sum(y)

    if total == 0:
        return np.nan

    return np.sum(x * y) / total


def variance(domain):
    """
    Second central moment.

    Returns
    -------
    float
    """

    da = domain.data

    x = da.coords[domain.spectrum.axis_name].values
    y = da.values

    c = centroid(domain)

    total = np.sum(y)

    if total == 0:
        return np.nan

    return np.sum(y * (x - c) ** 2) / total


def skewness(domain):
    """
    Third standardized moment.

    Returns
    -------
    float
    """

    da = domain.data

    x = da.coords[domain.spectrum.axis_name].values
    y = da.values

    c = centroid(domain)
    var = variance(domain)

    if var == 0 or np.isnan(var):
        return np.nan

    std = np.sqrt(var)

    total = np.sum(y)

    return np.sum(y * ((x - c) / std) ** 3) / total