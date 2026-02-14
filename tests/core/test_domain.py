import numpy as np
import pytest
import xarray as xr

from pyspectrum.core.spectrum import Spectrum
from pyspectrum.core.domain import Domain
from pyspectrum.utils.gaussian import gaussian


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def simple_spectrum():
    channels = np.linspace(0, 1000, 1001)
    counts = np.zeros_like(channels)

    counts += 1000 * gaussian(channels, x0=500, sigma=5)
    counts += 10  # flat background

    return Spectrum(
        counts=counts,
        channels=channels,
        metadata={"source": "test"}
    )


@pytest.fixture
def simple_domain(simple_spectrum):
    return Domain(
        spectrum=simple_spectrum,
        start=480,
        stop=520,
        metadata={"snr": 12.3},
        score=5.6,
        method="test-finder",
    )


# -----------------------------------------------------------------------------
# Construction & identity
# -----------------------------------------------------------------------------

def test_domain_creation(simple_domain):
    assert isinstance(simple_domain, Domain)
    assert simple_domain.start == 480
    assert simple_domain.stop == 520
    assert simple_domain.width == 40


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
    assert isinstance(da, xr.DataArray)


def test_domain_data_slice(simple_domain):
    da = simple_domain.data

    assert da.sizes["channel"] == simple_domain.width
    assert np.allclose(
        da.coords["channel"].values,
        simple_domain.spectrum.channels[simple_domain.start : simple_domain.stop]
    )


def test_domain_indices(simple_domain):
    indices = simple_domain.indices
    assert indices[0] == simple_domain.start
    assert indices[-1] == simple_domain.stop - 1
    assert len(indices) == simple_domain.width


# -----------------------------------------------------------------------------
# Metadata propagation
# -----------------------------------------------------------------------------

def test_domain_attrs(simple_domain):
    da = simple_domain.data
    attrs = da.attrs

    assert attrs["domain_start"] == simple_domain.start
    assert attrs["domain_stop"] == simple_domain.stop
    assert attrs["domain_width"] == simple_domain.width
    assert attrs["domain_method"] == "test-finder"
    assert attrs["domain_score"] == 5.6
    assert attrs["snr"] == 12.3


# -----------------------------------------------------------------------------
# Background subtraction
# -----------------------------------------------------------------------------

def test_background_subtraction_returns_new_domain(simple_domain):
    bg = np.ones(simple_domain.width) * 10

    sub = simple_domain.subtract_background(bg)

    assert isinstance(sub, Domain)
    assert sub is not simple_domain


def test_background_subtraction_does_not_modify_original(simple_domain):
    bg = np.ones(simple_domain.width) * 10

    sub = simple_domain.subtract_background(bg)

    original = simple_domain.data.values
    modified = sub.data.values

    assert not np.allclose(original, modified)
    assert np.allclose(modified, original - 10)


def test_background_callable(simple_domain):
    def bg_func(domain):
        return np.ones(domain.width) * 10

    sub = simple_domain.subtract_background(bg_func)

    assert np.allclose(
        sub.data.values,
        simple_domain.data.values - 10
    )


# -----------------------------------------------------------------------------
# Identity preservation
# -----------------------------------------------------------------------------

def test_background_preserves_identity(simple_domain):
    bg = np.zeros(simple_domain.width)
    sub = simple_domain.subtract_background(bg)

    assert sub.start == simple_domain.start
    assert sub.stop == simple_domain.stop
    assert sub.spectrum is simple_domain.spectrum
    assert sub.metadata == simple_domain.metadata


# -----------------------------------------------------------------------------
# Representation & length
# -----------------------------------------------------------------------------

def test_len(simple_domain):
    assert len(simple_domain) == simple_domain.width


def test_repr(simple_domain):
    r = repr(simple_domain)
    assert "Domain(" in r
    assert "start=480" in r
    assert "stop=520" in r
