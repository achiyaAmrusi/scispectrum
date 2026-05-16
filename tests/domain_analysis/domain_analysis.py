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
        channels=channels,
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
