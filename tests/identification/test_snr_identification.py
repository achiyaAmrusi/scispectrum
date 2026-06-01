"""Tests for SNRFinder."""

import numpy as np
import pytest

from pyspectrum.core import Spectrum
from pyspectrum.core.domain import Domain
from pyspectrum.calibration import ResolutionCalibration
from pyspectrum.identification.convolution import Convolution
from pyspectrum.identification.snr import SNRFinder

# ---------------------------------------------------------------------------
# Synthetic spectrum: two well-separated Gaussian peaks
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(1)

N       = 1000
CENTERS = [200, 700]
SIGMA   = 10.0
FWHM    = SIGMA * 2 * np.sqrt(2 * np.log(2))
AMP     = 2000.0
BG      = 20.0


def _build_counts():
    counts = np.full(N, BG)
    for c in CENTERS:
        counts += AMP * np.exp(-((np.arange(N) - c) ** 2) / (2 * SIGMA ** 2))
    return RNG.poisson(counts).astype(float)


@pytest.fixture(scope="module")
def spectrum():
    s = Spectrum(counts=_build_counts())
    s.set_resolution_calibration(ResolutionCalibration(lambda x: FWHM))
    return s


@pytest.fixture(scope="module")
def finder(spectrum):
    conv = Convolution(resolution=spectrum.resolution_calib)
    return SNRFinder(conv, n_sigma_signal_threshold=4.0, n_sigma_bg_threshold=2.0)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSNRFinderFind:

    def test_returns_list(self, finder, spectrum):
        assert isinstance(finder.find(spectrum), list)

    def test_returns_domains(self, finder, spectrum):
        domains = finder.find(spectrum)
        assert all(isinstance(d, Domain) for d in domains)

    def test_detects_both_peaks(self, finder, spectrum):
        domains = finder.find(spectrum)
        assert len(domains) >= 2

    def test_domains_cover_known_centers(self, finder, spectrum):
        domains = finder.find(spectrum)
        for center in CENTERS:
            covered = any(d.start <= center < d.stop for d in domains)
            assert covered, f"No domain covers peak at channel {center}"

    def test_domain_bounds_valid(self, finder, spectrum):
        domains = finder.find(spectrum)
        for d in domains:
            assert d.start >= 0
            assert d.stop <= N
            assert d.start < d.stop


class TestSNRFinderDomain:

    def test_domain_method_returns_domain(self, finder, spectrum):
        d = finder.domain(spectrum, axis_value=float(CENTERS[0]))
        assert isinstance(d, Domain)

    def test_domain_covers_requested_center(self, finder, spectrum):
        center = CENTERS[1]
        d = finder.domain(spectrum, axis_value=float(center))
        assert d.start <= center < d.stop


class TestSNRFinderCaching:

    def test_n_sigma_cached_on_second_call(self, finder, spectrum):
        finder.find(spectrum)
        ns1 = finder._cached_n_sigma
        finder.find(spectrum)
        ns2 = finder._cached_n_sigma
        assert ns1 is ns2   # same object — not recomputed
