"""
Tests for domain_analysis free functions.

Two synthetic setups:
  single_domain  — one Gaussian peak at CENTER, used for single-peak estimators
  two_peak_domain — two well-separated peaks, used to verify multi-peak detection
"""

import numpy as np
import pytest

from scispectrum.core import Spectrum
from scispectrum.core.domain import Domain
from scispectrum.calibration import ResolutionCalibration
from scispectrum.domain_analysis.find_peaks import find_domain_peaks
from scispectrum.domain_analysis.morphology import (
    domain_count_peaks,
    domain_peaks_position,
    domain_peaks_fwhm,
    domain_bases,
)
from scispectrum.domain_analysis.single_peak import center_estimator, fwhm_estimator
from scispectrum.domain_analysis.moment import centroid, variance, skewness
from scispectrum.domain_analysis.background import domain_erf_background, domain_linear_background

RNG = np.random.default_rng(7)

BG           = 10
SIGMA        = 10
FWHM_TRUE    = SIGMA * 2 * np.sqrt(2 * np.log(2))

# Single-peak constants
CENTER       = 500
AMP          = 500

# Two-peak constants
CENTERS_2    = (300, 370)   # ~3 FWHM apart — close but clearly separable
AMPS_2       = (300, 500)

N_CHANNELS   = 1001

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolution_calib():
    return ResolutionCalibration(lambda x: FWHM_TRUE)

def _gaussian(channels, center, amp, sigma):
    return amp * np.exp(-((channels - center) ** 2) / (2 * sigma ** 2))

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def single_spectrum():
    channels = np.arange(N_CHANNELS, dtype=float)
    counts   = np.full_like(channels, float(BG))
    counts  += _gaussian(channels, CENTER, AMP, SIGMA)
    counts   = RNG.poisson(counts).astype(float)
    sp = Spectrum(counts=counts)
    sp.set_resolution_calibration(_resolution_calib())
    return sp

@pytest.fixture(scope="module")
def two_peak_spectrum():
    channels = np.arange(N_CHANNELS, dtype=float)
    counts   = np.full_like(channels, float(BG))
    for c, a in zip(CENTERS_2, AMPS_2):
        counts += _gaussian(channels, c, a, SIGMA)
    counts = RNG.poisson(counts).astype(float)
    sp = Spectrum(counts=counts)
    sp.set_resolution_calibration(_resolution_calib())
    return sp

@pytest.fixture(scope="module")
def single_domain(single_spectrum):
    half = 4 * SIGMA
    return Domain(spectrum=single_spectrum,
                  start=int(CENTER - half), stop=int(CENTER + half))

@pytest.fixture(scope="module")
def two_peak_domain(two_peak_spectrum):
    start = int(min(CENTERS_2) - 4 * SIGMA)
    stop  = int(max(CENTERS_2) + 4 * SIGMA)
    return Domain(spectrum=two_peak_spectrum, start=start, stop=stop)

# ---------------------------------------------------------------------------
# find_domain_peaks
# ---------------------------------------------------------------------------

class TestFindDomainPeaks:

    def test_returns_positions_and_props(self, single_domain):
        positions, props = find_domain_peaks(single_domain, prominence=10)
        assert positions is not None and props is not None

    def test_single_peak_detected(self, single_domain):
        positions, _ = find_domain_peaks(single_domain, prominence=10)
        assert len(positions) == 1

    def test_single_peak_near_center(self, single_domain):
        positions, _ = find_domain_peaks(single_domain, prominence=10)
        assert abs(positions[0] - CENTER) < 2 * SIGMA

    def test_two_peaks_detected(self, two_peak_domain):
        positions, _ = find_domain_peaks(two_peak_domain, prominence=10)
        assert len(positions) == 2

    def test_two_peaks_near_true_centers(self, two_peak_domain):
        positions, _ = find_domain_peaks(two_peak_domain, prominence=10)
        for true_c in CENTERS_2:
            assert any(abs(p - true_c) < 2 * SIGMA for p in positions)

    def test_props_has_fwhm(self, single_domain):
        _, props = find_domain_peaks(single_domain, prominence=10)
        assert "fwhm" in props

    def test_fwhm_accurate(self, single_domain):
        _, props = find_domain_peaks(single_domain, prominence=10)
        assert abs(props["fwhm"][0] - FWHM_TRUE) / FWHM_TRUE < 0.3

# ---------------------------------------------------------------------------
# morphology helpers
# ---------------------------------------------------------------------------

class TestMorphology:

    def test_count_single(self, single_domain):
        assert domain_count_peaks(single_domain, prominence=10) == 1

    def test_count_two(self, two_peak_domain):
        assert domain_count_peaks(two_peak_domain, prominence=10) == 2

    def test_position_single(self, single_domain):
        pos = domain_peaks_position(single_domain, prominence=10)
        assert len(pos) == 1
        assert abs(pos[0] - CENTER) < 2 * SIGMA

    def test_position_two(self, two_peak_domain):
        pos = domain_peaks_position(two_peak_domain, prominence=10)
        assert len(pos) == 2
        for true_c in CENTERS_2:
            assert any(abs(p - true_c) < 2 * SIGMA for p in pos)

    def test_fwhm_length_matches_peak_count(self, two_peak_domain):
        fwhm = domain_peaks_fwhm(two_peak_domain, prominence=10)
        assert len(fwhm) == 2

    def test_domain_bases_near_background(self, single_domain):
        left, right = domain_bases(single_domain)
        assert abs(left  - BG) < 20
        assert abs(right - BG) < 20

# ---------------------------------------------------------------------------
# single_peak estimators
# ---------------------------------------------------------------------------

class TestSinglePeakEstimators:

    def test_center_within_sigma(self, single_domain):
        assert abs(center_estimator(single_domain) - CENTER) < SIGMA

    def test_fwhm_within_tolerance(self, single_domain):
        fwhm = fwhm_estimator(single_domain)
        assert abs(fwhm - FWHM_TRUE) / FWHM_TRUE < 0.3

# ---------------------------------------------------------------------------
# moment functions
# ---------------------------------------------------------------------------

class TestMoment:

    def test_centroid_near_peak_center(self, single_domain):
        assert abs(centroid(single_domain) - CENTER) < SIGMA

    def test_variance_near_true_variance(self, single_domain):
        v = variance(single_domain)
        assert abs(v - SIGMA ** 2) / SIGMA ** 2 < 0.5

    def test_skewness_near_zero(self, single_domain):
        assert abs(skewness(single_domain)) < 1.0

    def test_centroid_empty_domain_returns_nan(self, single_spectrum):
        zero_domain = Domain(
            spectrum=Spectrum(counts=np.zeros(N_CHANNELS)),
            start=400, stop=600,
        )
        assert np.isnan(centroid(zero_domain))

# ---------------------------------------------------------------------------
# domain_analysis background functions
# ---------------------------------------------------------------------------

class TestDomainBackground:

    def test_linear_shape(self, single_domain):
        bg = domain_linear_background(single_domain)
        assert bg.shape == (single_domain.stop - single_domain.start,)

    def test_linear_endpoints_match_bases(self, single_domain):
        left, right = domain_bases(single_domain)
        bg = domain_linear_background(single_domain)
        assert abs(bg[0]  - left)  < 1.0
        assert abs(bg[-1] - right) < 1.0

    def test_linear_near_flat_for_flat_background(self, single_domain):
        bg = domain_linear_background(single_domain)
        assert abs(bg[-1] - bg[0]) < 10

    def test_erf_shape(self, single_domain):
        bg = domain_erf_background(single_domain, prominence=10)
        assert bg.shape == (single_domain.stop - single_domain.start,)

    def test_erf_values_near_background_level(self, single_domain):
        bg = domain_erf_background(single_domain, prominence=10)
        assert np.all(np.abs(bg - BG) < 30)
