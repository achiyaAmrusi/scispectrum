"""
Tests for all five background estimators.

Synthetic setup: flat background of BG_LEVEL counts under a single Gaussian
peak of amplitude PEAK_AMP at PEAK_CENTER. All background estimators share
two properties that can be reliably tested:

1. Output shape matches input.
2. The estimate at the peak center is below the measured counts there
   (the estimator should not "follow" the peak).

Edge-accuracy tests are only applied to SNIP, which is designed to recover a
flat continuum. ALS underestimates by design (p << 1 pulls the baseline below
the true level) and IterativePolyFit can go negative at the edges.
"""

import numpy as np
import pytest

from scispectrum.background import (
    ALSBackground,
    SNIPBackground,
    IterativePolyFit,
    IterativePolyFitWithMinimum,
    MinimaEnvelopeBackground,
)
from scispectrum.calibration import ResolutionCalibration
from scispectrum.identification.convolution import Convolution

# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(0)

N           = 1000
AXIS        = np.arange(N, dtype=float)
BG_LEVEL    = 50.0
PEAK_AMP    = 2000.0
PEAK_CENTER = 500
PEAK_SIGMA  = 15.0
PEAK_FWHM   = PEAK_SIGMA * 2 * np.sqrt(2 * np.log(2))

EDGE_SLICE = slice(0, 150)   # far from peak — only SNIP accuracy tested here

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _true_counts():
    signal = PEAK_AMP * np.exp(-((AXIS - PEAK_CENTER) ** 2) / (2 * PEAK_SIGMA ** 2))
    return signal + BG_LEVEL

def _noisy_counts():
    return RNG.poisson(_true_counts()).astype(float)

def _resolution(x):
    return np.full_like(np.asarray(x, dtype=float), PEAK_FWHM)

def _convolution():
    return Convolution(resolution=lambda x: PEAK_FWHM)

# ---------------------------------------------------------------------------
# Shared assertions
# ---------------------------------------------------------------------------

def _assert_shape(bg_est):
    assert bg_est.shape == (N,)

def _assert_below_peak(bg_est, counts):
    """Estimator must not track the Gaussian peak."""
    assert bg_est[PEAK_CENTER] < counts[PEAK_CENTER]

# ---------------------------------------------------------------------------
# ALSBackground
# ---------------------------------------------------------------------------

class TestALSBackground:

    def test_output_shape(self):
        bg = ALSBackground(lam=1e5, p=0.001).estimate(AXIS, _noisy_counts())
        _assert_shape(bg)

    def test_below_peak(self):
        counts = _noisy_counts()
        bg = ALSBackground(lam=1e5, p=0.001).estimate(AXIS, counts)
        _assert_below_peak(bg, counts)

    def test_non_negative(self):
        bg = ALSBackground(lam=1e5, p=0.001).estimate(AXIS, _noisy_counts())
        assert np.all(bg >= 0)

# ---------------------------------------------------------------------------
# SNIPBackground
# ---------------------------------------------------------------------------

class TestSNIPBackground:

    def test_output_shape(self):
        bg = SNIPBackground(iterations=20, resolution=_resolution).estimate(AXIS, _noisy_counts())
        _assert_shape(bg)

    def test_below_peak(self):
        counts = _noisy_counts()
        bg = SNIPBackground(iterations=20, resolution=_resolution).estimate(AXIS, counts)
        _assert_below_peak(bg, counts)

    def test_edge_accuracy(self):
        # SNIP is designed to recover the flat continuum — test edge channels
        counts = _noisy_counts()
        bg = SNIPBackground(iterations=20, resolution=_resolution).estimate(AXIS, counts)
        edge_mean = bg[EDGE_SLICE].mean()
        assert abs(edge_mean - BG_LEVEL) < 20.0, (
            f"SNIP edge estimate: {edge_mean:.1f}, expected ≈ {BG_LEVEL}"
        )

    def test_no_smoothing(self):
        bg = SNIPBackground(iterations=10, smooth=False).estimate(AXIS, _noisy_counts())
        _assert_shape(bg)

# ---------------------------------------------------------------------------
# IterativePolyFit
# ---------------------------------------------------------------------------

class TestIterativePolyFit:

    def test_output_shape(self):
        bg = IterativePolyFit(degree=5).estimate(AXIS, _noisy_counts())
        _assert_shape(bg)

    def test_below_peak(self):
        counts = _noisy_counts()
        bg = IterativePolyFit(degree=5).estimate(AXIS, counts)
        _assert_below_peak(bg, counts)

# ---------------------------------------------------------------------------
# IterativePolyFitWithMinimum
# ---------------------------------------------------------------------------

class TestIterativePolyFitWithMinimum:

    @pytest.fixture(scope="class")
    def estimator(self):
        return IterativePolyFitWithMinimum(
            resolution=_resolution,
            conv=_convolution(),
            degree=5,
            max_iter=50,
        )

    def test_output_shape(self, estimator):
        bg = estimator.estimate(AXIS, _noisy_counts())
        _assert_shape(bg)

    def test_below_peak(self, estimator):
        counts = _noisy_counts()
        bg = estimator.estimate(AXIS, counts)
        _assert_below_peak(bg, counts)

# ---------------------------------------------------------------------------
# MinimaEnvelopeBackground
# ---------------------------------------------------------------------------

class TestMinimaEnvelopeBackground:

    @pytest.fixture(scope="class")
    def estimator(self):
        return MinimaEnvelopeBackground(
            resolution_calib=ResolutionCalibration(_resolution),
            conv=_convolution(),
            iterations=20,
        )

    def test_output_shape(self, estimator):
        bg = estimator.estimate(AXIS, _noisy_counts())
        _assert_shape(bg)

    def test_below_peak(self, estimator):
        counts = _noisy_counts()
        bg = estimator.estimate(AXIS, counts)
        _assert_below_peak(bg, counts)
