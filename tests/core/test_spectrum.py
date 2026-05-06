import numpy as np
import xarray as xr
import pytest

from pyspectrum.core.spectrum import Spectrum
from pyspectrum.calibration import ResolutionCalibration, AxisCalibration

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
        channels=channels,
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

