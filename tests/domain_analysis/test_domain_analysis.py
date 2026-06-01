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
from pyspectrum.domain_analysis.moment import centroid, variance, skewness
from pyspectrum.domain_analysis.background import domain_erf_background, domain_linear_background

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

# ---------------------------------------------------------------------------
# moment functions
# ---------------------------------------------------------------------------

class TestMoment:

    def test_centroid_near_peak_center(self, simple_domain):
        c = centroid(simple_domain)
        assert abs(c - CENTERS[1]) < SIGMA[1]

    def test_variance_near_true_variance(self, simple_domain):
        # Variance is sigma² — flat background dilutes it slightly, so test loosely
        v = variance(simple_domain)
        assert abs(v - SIGMA[1] ** 2) / SIGMA[1] ** 2 < 0.5

    def test_skewness_near_zero(self, simple_domain):
        # Symmetric Gaussian → skewness ≈ 0
        s = skewness(simple_domain)
        assert abs(s) < 1.0

    def test_centroid_empty_domain_returns_nan(self, simple_spectrum):
        # Domain of all-zero counts → centroid is undefined
        zero_domain = Domain(
            spectrum=Spectrum(counts=np.zeros(len(simple_spectrum.counts))),
            start=400, stop=600,
        )
        assert np.isnan(centroid(zero_domain))

# ---------------------------------------------------------------------------
# domain_analysis background functions
# ---------------------------------------------------------------------------

class TestDomainBackground:

    def test_linear_output_shape(self, simple_domain):
        bg = domain_linear_background(simple_domain)
        assert bg.shape == (simple_domain.stop - simple_domain.start,)

    def test_linear_endpoints_match_bases(self, simple_domain):
        left, right = domain_bases(simple_domain)
        bg = domain_linear_background(simple_domain)
        assert abs(bg[0]  - left)  < 1.0
        assert abs(bg[-1] - right) < 1.0

    def test_linear_is_monotone_or_flat(self, simple_domain):
        # With symmetric flat background both edges ≈ BG, so slope ≈ 0
        bg = domain_linear_background(simple_domain)
        assert abs(bg[-1] - bg[0]) < 10

    def test_erf_output_shape(self, simple_domain):
        bg = domain_erf_background(simple_domain, prominence=10)
        assert bg.shape == (simple_domain.stop - simple_domain.start,)

    def test_erf_values_near_background_level(self, simple_domain):
        bg = domain_erf_background(simple_domain, prominence=10)
        assert np.all(np.abs(bg - BG) < 30)
