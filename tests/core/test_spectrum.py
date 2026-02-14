import pytest
import numpy as np
from pyspectrum.core.spectrum import Spectrum
from pyspectrum.utils.gaussian import gaussian

@pytest.fixture
def simple_spectrum():
    channels = np.linspace(0, 1000, 1001)
    counts = np.zeros_like(channels)

    # single clean Gaussian peak
    counts += 1000 * gaussian(channels, x0=500, sigma=5)
    counts += 10  # flat background

    return Spectrum(
        counts=counts,
        channels=channels,
        metadata={"test": True}
    )

def test_spectrum_creation(simple_spectrum):
    """Test that a Spectrum object is created and has expected attributes."""

    # Check type
    assert isinstance(simple_spectrum, Spectrum)

    # Check channels and counts shape
    assert simple_spectrum.channels.shape == simple_spectrum.counts.shape
    assert simple_spectrum.channels.size == 1001

    # Check metadata
    assert "test" in simple_spectrum.metadata
    assert simple_spectrum.metadata["test"] is True

def test_spectrum_op(simple_spectrum):
    """Test that a Spectrum object is created and has expected attributes."""
    # Check type
    assert isinstance(simple_spectrum+simple_spectrum, Spectrum)

