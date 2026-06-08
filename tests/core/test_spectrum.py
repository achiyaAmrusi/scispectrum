import numpy as np
import xarray as xr
import pytest

from scispectrum.core.spectrum import Spectrum
from scispectrum.calibration import ResolutionCalibration, AxisCalibration

BG =10
AMP = (100, 500)
CENTERS = (100, 500)
SIGMA = (10, 10)
DOMAIN_SIGMA_NUM = 4

# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def spectrum():
    channels = np.linspace(0, 1000, 1001)
    counts = np.zeros_like(channels)

    x0, a, s = (np.asarray(v)[:, None] for v in (CENTERS, AMP, SIGMA))

    counts += (a * np.exp(-(channels - x0) ** 2 / (2 * s ** 2))).sum(axis=0)
    counts += BG
    counts = np.random.poisson(counts).astype(float)

    res = ResolutionCalibration(lambda x: SIGMA[0])
    poly = np.poly1d([2.0, 0 ])
    axis_calib = AxisCalibration(func=poly, name="channel")
    return Spectrum(
        counts=counts,
        resolution_calib=res,
        axis_calib=axis_calib
    )
# -----------------------------------------------------------------------------
# Construction & identity
# -----------------------------------------------------------------------------

def test_spectrum_creation(spectrum):
    assert isinstance(spectrum, Spectrum)
    assert isinstance(spectrum.resolution_calib, ResolutionCalibration)
    assert isinstance(spectrum.axis_calib, AxisCalibration)
    assert isinstance(spectrum.channels, np.ndarray)
    assert isinstance(spectrum.counts, np.ndarray)
    assert np.all(np.isclose(spectrum.axis_calib.apply(spectrum.channels),spectrum.axis))
    assert spectrum.axis_name == "channel"


# -----------------------------------------------------------------------------
# Data interface
# -----------------------------------------------------------------------------

def test_spectrum_data_is_xarray(spectrum):
    da = spectrum.data
    assert isinstance(da, xr.DataArray)


# -----------------------------------------------------------------------------
# Background subtraction
# -----------------------------------------------------------------------------
#
def test_background_subtraction_returns_new_domain(spectrum: Spectrum):
    bg = np.ones_like(spectrum.channels) * BG
    sub =spectrum-bg
    assert isinstance(sub, Spectrum)

def test_background_length_mismatch_raises(spectrum):
    wrong_bg = np.ones(len(spectrum.channels) - 1)

    with pytest.raises(ValueError):
        spectrum - wrong_bg

# -----------------------------------------------------------------------------
# Error propagation
# -----------------------------------------------------------------------------

def test_addition_error_propagation():
    counts1 = np.array([10., 20., 30.])
    err1 = np.array([1., 2., 3.])

    counts2 = np.array([5., 7., 9.])
    err2 = np.array([0.5, 0.7, 0.9])

    s1 = Spectrum(counts=counts1, counts_err=err1)
    s2 = Spectrum(counts=counts2, counts_err=err2)

    result = s1 + s2

    expected_counts = counts1 + counts2
    expected_err = np.sqrt(err1**2 + err2**2)

    assert np.allclose(result.counts, expected_counts)
    assert np.allclose(result.counts_err, expected_err)


def test_subtraction_error_propagation():
    counts1 = np.array([10., 20., 30.])
    err1 = np.array([1., 2., 3.])

    counts2 = np.array([5., 7., 9.])
    err2 = np.array([0.5, 0.7, 0.9])

    s1 = Spectrum(counts=counts1, counts_err=err1)
    s2 = Spectrum(counts=counts2, counts_err=err2)

    result = s1 - s2

    expected_counts = counts1 - counts2
    expected_err = np.sqrt(err1**2 + err2**2)

    assert np.allclose(result.counts, expected_counts)
    assert np.allclose(result.counts_err, expected_err)


def test_multiplication_error_propagation():
    counts1 = np.array([10., 20., 30.])
    err1 = np.array([1., 2., 3.])

    counts2 = np.array([2., 4., 5.])
    err2 = np.array([0.2, 0.4, 0.5])

    s1 = Spectrum(counts=counts1, counts_err=err1)
    s2 = Spectrum(counts=counts2, counts_err=err2)

    result = s1 * s2

    expected_counts = counts1 * counts2

    expected_err = expected_counts * np.sqrt(
        (err1 / counts1) ** 2 +
        (err2 / counts2) ** 2
    )

    assert np.allclose(result.counts, expected_counts)
    assert np.allclose(result.counts_err, expected_err)


def test_division_error_propagation():
    counts1 = np.array([10., 20., 30.])
    err1 = np.array([1., 2., 3.])

    counts2 = np.array([2., 4., 5.])
    err2 = np.array([0.2, 0.4, 0.5])

    s1 = Spectrum(counts=counts1, counts_err=err1)
    s2 = Spectrum(counts=counts2, counts_err=err2)

    result = s1 / s2

    expected_counts = counts1 / counts2

    expected_err = expected_counts * np.sqrt(
        (err1 / counts1) ** 2 +
        (err2 / counts2) ** 2
    )

    assert np.allclose(result.counts, expected_counts)
    assert np.allclose(result.counts_err, expected_err)


def test_scalar_multiplication_error_propagation():
    counts = np.array([10., 20., 30.])
    errs = np.array([1., 2., 3.])

    s = Spectrum(counts=counts, counts_err=errs)

    result = s * 2

    assert np.allclose(result.counts, counts * 2)
    assert np.allclose(result.counts_err, errs * 2)


def test_scalar_division_error_propagation():
    counts = np.array([10., 20., 30.])
    errs = np.array([1., 2., 3.])

    s = Spectrum(counts=counts, counts_err=errs)

    result = s / 2

    assert np.allclose(result.counts, counts / 2)
    assert np.allclose(result.counts_err, errs / 2)


def test_operation_without_errors_returns_no_errors():
    s1 = Spectrum(counts=np.array([1., 2., 3.]))
    s2 = Spectrum(counts=np.array([4., 5., 6.]))

    result = s1 + s2

    assert result.counts_err is None


def test_mixed_error_and_no_error():
    counts1 = np.array([10., 20., 30.])
    err1 = np.array([1., 2., 3.])

    counts2 = np.array([5., 7., 9.])

    s1 = Spectrum(counts=counts1, counts_err=err1)
    s2 = Spectrum(counts=counts2)

    result = s1 + s2

    assert np.allclose(result.counts, counts1 + counts2)
    assert np.allclose(result.counts_err, err1)