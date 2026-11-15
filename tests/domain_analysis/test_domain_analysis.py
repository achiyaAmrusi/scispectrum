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


# ---------------------------------------------------------------------------
# Background errors: exact for any sum over the domain (bases inside the domain)
# ---------------------------------------------------------------------------

from uncertainties.unumpy import nominal_values

def _poisson_domain(scale=1.0, low=False):
    """Poisson spectrum in counts (scale=1) or normalised (e.g. cps), errors max(sqrt(N),1)."""
    ch = np.arange(N_CHANNELS, dtype=float)
    lam = (0.02 if low else 500) + (2 if low else 25000) * np.exp(-((ch - CENTER) ** 2) / (2 * SIGMA ** 2))
    n = np.random.default_rng(11).poisson(lam).astype(float)
    sp = Spectrum(counts=n / scale, counts_err=np.maximum(np.sqrt(n), 1) / scale)
    sp.set_resolution_calibration(_resolution_calib())
    return Domain(spectrum=sp, start=CENTER - 4 * SIGMA, stop=CENTER + 4 * SIGMA)

def _reference_std(domain, background, window):
    """Independent error of sum(net[window]): numerical Jacobian of the background on
    floats, then plain error propagation over independent channels."""
    x = domain.data.values
    W = np.array([(background._model(x + np.eye(len(x))[j]) - background._model(x)) for j in range(len(x))]).T
    sel = np.zeros(len(x)); sel[window] = 1
    a = sel - sel @ W
    return np.sqrt((a ** 2 * domain.spectrum.counts_err[domain.indices] ** 2).sum())

@pytest.mark.parametrize("scale", [1.0, 3600.0])
@pytest.mark.parametrize("low", [False, True])
@pytest.mark.parametrize("window", [slice(None), slice(20, -20), slice(2, 40)])  # whole / no bases / partial
def test_linear_background_error_is_exact(scale, low, window):
    d = _poisson_domain(scale, low)
    bg = domain_linear_background(d)
    net = d.subtract_background(bg).data_with_errors.values[window].sum()
    assert net.s == pytest.approx(_reference_std(d, bg, window), rel=1e-9)

def test_erf_background_error_is_exact_and_peaks_found_once(monkeypatch):
    import scispectrum.domain_analysis.background as bgmod
    calls = []
    monkeypatch.setattr(bgmod, "find_domain_peaks", lambda *a, **k: calls.append(1) or find_domain_peaks(*a, **k))
    d = _poisson_domain()
    bg = domain_erf_background(d, prominence=10)
    sub = d.subtract_background(bg)
    net = sub.data_with_errors.sum().item()
    assert net.s == pytest.approx(_reference_std(d, bg, slice(None)), rel=1e-9)
    assert len(calls) == 1

def test_whole_domain_poisson_formula():
    """Var(net) = Var(C) + Var(fit) - 2B for Poisson counts summed over the whole domain."""
    d = _poisson_domain()
    bg = domain_linear_background(d)
    net = d.subtract_background(bg).data_with_errors.sum().item()
    fit = bg._model(d.data_with_errors.values).sum()
    var_c = (d.spectrum.counts_err[d.indices] ** 2).sum()
    assert net.s ** 2 == pytest.approx(var_c + fit.s ** 2 - 2 * fit.n, rel=1e-9)

def test_background_is_plain_float_and_data_consistent(single_domain):
    for d in (single_domain, _poisson_domain()):
        bg = domain_linear_background(d)
        assert bg.dtype == float
        sub = d.subtract_background(bg)
        assert np.allclose(sub.data.values, d.data.values - bg)
    assert np.allclose(nominal_values(sub.data_with_errors.values), sub.data.values)

def test_edited_or_foreign_background_falls_back_to_plain_subtraction():
    d = _poisson_domain()
    bg = domain_linear_background(d)
    edited = bg.copy(); edited[5] += 100
    other = Domain(d.spectrum, start=d.start + 1, stop=d.stop + 1)
    for dom, b in ((d, edited), (d, bg * 1.1), (other, bg)):
        sub = dom.subtract_background(b)
        assert sub._background_model is None
        assert np.allclose(nominal_values(sub.data_with_errors.values), sub.data.values)

def test_new_counts_err_is_used():
    d = _poisson_domain()
    sub = d.subtract_background(domain_linear_background(d))
    before = sub.data_with_errors.sum().item().s
    d.spectrum.counts_err = d.spectrum.counts_err * 2
    assert sub.data_with_errors.sum().item().s == pytest.approx(2 * before)

def test_right_base_uses_last_channels(single_domain):
    from scispectrum.domain_analysis.morphology import _base_width
    n = _base_width(single_domain)
    assert domain_bases(single_domain)[1] == pytest.approx(single_domain.data.values[-n:].mean())

