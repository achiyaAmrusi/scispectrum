import numpy as np
import pytest
import xarray as xr
from uncertainties.unumpy import nominal_values, std_devs

from scispectrum.core.spectrum import Spectrum
from scispectrum.core.domain import Domain
from scispectrum.calibration import ResolutionCalibration
from scispectrum.domain_analysis.find_peaks import find_domain_peaks

BG =10
AMP = (100, 500)
CENTERS = (100, 500)
SIGMA = (10, 10)
DOMAIN_SIGMA_NUM = 4

# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def simple_spectrum():
    channels = np.linspace(0, 1000, 1001)
    counts = np.zeros_like(channels)

    x0, a, s = (np.asarray(v)[:, None] for v in (CENTERS, AMP, SIGMA))

    counts += (a * np.exp(-(channels - x0) ** 2 / (2 * s ** 2))).sum(axis=0)
    counts += BG
    counts = np.random.poisson(counts).astype(float)

    res = ResolutionCalibration(lambda x: SIGMA[0])

    return Spectrum(
        counts=counts,
        metadata={"source": "test"},
        resolution_calib=res
    )

@pytest.fixture
def simple_domain(simple_spectrum):
    return Domain(
        spectrum=simple_spectrum,
        start=CENTERS[1] - DOMAIN_SIGMA_NUM * SIGMA[1],
        stop=CENTERS[1] + DOMAIN_SIGMA_NUM * SIGMA[1]
    )


# -----------------------------------------------------------------------------
# Construction & identity
# -----------------------------------------------------------------------------

def test_domain_creation(simple_domain):
    assert isinstance(simple_domain, Domain)
    assert simple_domain.start == CENTERS[1] - DOMAIN_SIGMA_NUM * SIGMA[1]
    assert simple_domain.stop == CENTERS[1] + DOMAIN_SIGMA_NUM * SIGMA[1]
    assert len(simple_domain.indices) == DOMAIN_SIGMA_NUM * 2 * SIGMA[1]

def test_invalid_bounds(simple_spectrum):
    with pytest.raises(ValueError):
        Domain(simple_spectrum, start=-1, stop=10)

    with pytest.raises(ValueError):
        Domain(simple_spectrum, start=10, stop=10)


# -----------------------------------------------------------------------------
# Data interface
# -----------------------------------------------------------------------------

def test_domain_data_is_xarray(simple_domain):
    da = simple_domain.data

    assert np.allclose(da.coords["channel"].values, simple_domain.spectrum.channels[simple_domain.indices])
    assert isinstance(da, xr.DataArray)


def test_domain_indices(simple_domain):
    indices = simple_domain.indices
    assert indices[0] == simple_domain.start
    assert indices[-1] == simple_domain.stop - 1

# -----------------------------------------------------------------------------
# Background subtraction
# -----------------------------------------------------------------------------
#
def test_background_subtraction_returns_new_domain(simple_domain: Domain):
    bg = np.ones_like(simple_domain.indices) * BG
    sub = simple_domain.subtract_background(bg)
    assert isinstance(sub, Domain)

def test_background_subtraction_does_not_modify_original(simple_domain):

    bg = np.ones_like(simple_domain.indices) * BG
    sub = simple_domain.subtract_background(bg)

    original = simple_domain.data.values
    modified = sub.data.values

    assert not np.allclose(original, modified)
    assert np.allclose(modified, original - 10)


def test_background_length_mismatch_raises(simple_domain):
    wrong_bg = np.ones(len(simple_domain.indices) - 1)

    with pytest.raises(ValueError):
        simple_domain.subtract_background(wrong_bg)


def test_background_none_restores_original(simple_domain):
    bg = np.ones_like(simple_domain.indices) * BG

    sub = simple_domain.subtract_background(bg)
    restored = sub.subtract_background(None)

    assert np.allclose(restored.data.values, simple_domain.data.values)

# -----------------------------------------------------------------------------
# Peaks detection
# -----------------------------------------------------------------------------
#
def test_finding_peak_in_domain(simple_domain):
    positions, props = find_domain_peaks(simple_domain, smooth=True, prominence=AMP[1]/2)
    assert len(positions) == 1
    assert np.allclose(positions[0], CENTERS[1], CENTERS[1]*0.1)


# -----------------------------------------------------------------------------
# Background error propagation
# -----------------------------------------------------------------------------

@pytest.fixture
def spectrum_with_errors():
    counts = np.full(200, 100.0)
    counts_err = np.full(200, 10.0)
    return Spectrum(counts=counts, counts_err=counts_err)


def test_subtract_background_with_err_returns_new_domain(spectrum_with_errors):
    domain = Domain(spectrum_with_errors, start=50, stop=150)
    bg = np.full(100, 20.0)
    bg_err = np.full(100, 2.0)
    sub = domain.subtract_background(bg, bg_err)
    assert isinstance(sub, Domain)
    assert sub.background_err is not None


def test_background_err_length_mismatch_raises(spectrum_with_errors):
    domain = Domain(spectrum_with_errors, start=50, stop=150)
    bg = np.full(100, 20.0)
    wrong_err = np.full(99, 2.0)
    with pytest.raises(ValueError):
        domain.subtract_background(bg, wrong_err)


def test_background_err_without_background_raises(spectrum_with_errors):
    domain = Domain(spectrum_with_errors, start=50, stop=150)
    bg_err = np.full(100, 2.0)
    with pytest.raises(ValueError):
        domain.subtract_background(None, bg_err)


def test_data_with_errors_propagates_background_uncertainty(spectrum_with_errors):
    domain = Domain(spectrum_with_errors, start=50, stop=150)
    bg = np.full(100, 20.0)
    bg_err = np.full(100, 2.0)
    sub = domain.subtract_background(bg, bg_err)

    result = sub.data_with_errors
    vals = nominal_values(result.values)
    errs = std_devs(result.values)

    # nominal: 100 - 20 = 80
    assert np.allclose(vals, 80.0)
    # error in quadrature: sqrt(10^2 + 2^2) = sqrt(104)
    assert np.allclose(errs, np.sqrt(10**2 + 2**2))


def test_data_with_errors_no_background_err_unchanged(spectrum_with_errors):
    domain = Domain(spectrum_with_errors, start=50, stop=150)
    bg = np.full(100, 20.0)
    sub = domain.subtract_background(bg)

    result = sub.data_with_errors
    errs = std_devs(result.values)

    # Only Poisson error — background has no uncertainty
    assert np.allclose(errs, 10.0)


def test_background_err_stored_as_separate_attribute(spectrum_with_errors):
    domain = Domain(spectrum_with_errors, start=50, stop=150)
    bg = np.full(100, 20.0)
    bg_err = np.full(100, 2.0)
    sub = domain.subtract_background(bg, bg_err)

    assert np.allclose(sub.background, bg)
    assert np.allclose(sub.background_err, bg_err)

