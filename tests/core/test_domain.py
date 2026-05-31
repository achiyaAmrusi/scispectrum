import numpy as np
import pytest
import xarray as xr

from pyspectrum.core.spectrum import Spectrum
from pyspectrum.core.domain import Domain
from pyspectrum.calibration import ResolutionCalibration
from pyspectrum.domain_analysis.find_peaks import find_domain_peaks

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

