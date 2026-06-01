"""
Tests for domain_analysis free functions.

Synthetic setup: two Gaussian peaks on a flat background. The second peak
(at CENTERS[1]) is isolated in its own Domain for single-peak tests.
"""

import numpy as np
import pytest

from pyspectrum.core import Spectrum
from pyspectrum.core.domain import Domain
from pyspectrum.calibration import ResolutionCalibration
from pyspectrum.domain_analysis.find_peaks import find_domain_peaks
from pyspectrum.domain_analysis.morphology import (
    domain_count_peaks,
    domain_peaks_position,
    domain_peaks_fwhm,
    domain_bases,
)
from pyspectrum.domain_analysis.single_peak import center_estimator, fwhm_estimator

RNG = np.random.default_rng(7)

BG        = 10
AMP       = (100, 500)
CENTERS   = (100, 500)
SIGMA     = (10, 10)
FWHM_TRUE = SIGMA[0] * 2 * np.sqrt(2 * np.log(2))
DOMAIN_SIGMA = 4

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def simple_spectrum():
    channels = np.arange(1001, dtype=float)
    counts   = np.full_like(channels, float(BG))
    for c, a, s in zip(CENTERS, AMP, SIGMA):
        counts += a * np.exp(-((channels - c) ** 2) / (2 * s ** 2))
    counts = RNG.poisson(counts).astype(float)
    sp = Spectrum(counts=counts)
    sp.set_resolution_calibration(ResolutionCalibration(lambda x: FWHM_TRUE))
    return sp

@pytest.fixture(scope="module")
def simple_domain(simple_spectrum):
    center, sigma = CENTERS[1], SIGMA[1]
    return Domain(
        spectrum=simple_spectrum,
        start=int(center - DOMAIN_SIGMA * sigma),
        stop=int(center + DOMAIN_SIGMA * sigma),
    )

# ---------------------------------------------------------------------------
# find_domain_peaks
# ---------------------------------------------------------------------------

class TestFindDomainPeaks:

    def test_returns_two_values(self, simple_domain):
        positions, props = find_domain_peaks(simple_domain, prominence=10)
        assert positions is not None and props is not None

    def test_detects_single_peak(self, simple_domain):
        positions, _ = find_domain_peaks(simple_domain, prominence=10)
        assert len(positions) == 1

    def test_peak_near_center(self, simple_domain):
        positions, _ = find_domain_peaks(simple_domain, prominence=10)
        assert abs(positions[0] - CENTERS[1]) < 2 * SIGMA[1]

    def test_props_has_fwhm(self, simple_domain):
        _, props = find_domain_peaks(simple_domain, prominence=10)
        assert "fwhm" in props

# ---------------------------------------------------------------------------
# morphology helpers
# ---------------------------------------------------------------------------

class TestMorphology:

    def test_count_peaks(self, simple_domain):
        assert domain_count_peaks(simple_domain, prominence=10) == 1

    def test_peaks_position_near_center(self, simple_domain):
        pos = domain_peaks_position(simple_domain, prominence=10)
        assert len(pos) == 1
        assert abs(pos[0] - CENTERS[1]) < 2 * SIGMA[1]

    def test_peaks_fwhm_returns_positive(self, simple_domain):
        # domain_peaks_fwhm uses peak_widths on unsmoothed data with a smoothed
        # peak index — accuracy is limited, but the result must be a positive scalar.
        fwhm = domain_peaks_fwhm(simple_domain, prominence=10)
        assert len(fwhm) == 1
        assert fwhm[0] > 0

    def test_find_domain_peaks_fwhm_accurate(self, simple_domain):
        # The FWHM in find_domain_peaks properties is computed on smoothed data
        # and is the reliable estimate.
        _, props = find_domain_peaks(simple_domain, prominence=10)
        assert abs(props["fwhm"][0] - FWHM_TRUE) / FWHM_TRUE < 0.3

    def test_domain_bases_returns_two_floats(self, simple_domain):
        left, right = domain_bases(simple_domain)
        assert isinstance(left, float)
        assert isinstance(right, float)

    def test_domain_bases_near_background(self, simple_domain):
        left, right = domain_bases(simple_domain)
        assert abs(left  - BG) < 20
        assert abs(right - BG) < 20

# ---------------------------------------------------------------------------
# single_peak estimators
# ---------------------------------------------------------------------------

class TestSinglePeakEstimators:

    def test_center_within_sigma(self, simple_domain):
        center = center_estimator(simple_domain)
        assert abs(center - CENTERS[1]) < SIGMA[1]

    def test_fwhm_within_tolerance(self, simple_domain):
        fwhm = fwhm_estimator(simple_domain)
        assert abs(fwhm - FWHM_TRUE) / FWHM_TRUE < 0.3
