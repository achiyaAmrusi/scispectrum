"""Tests for Convolution."""

import numpy as np
import pytest

from scispectrum.identification.convolution import Convolution

FWHM = 10.0
N    = 300
AXIS = np.arange(N, dtype=float)


@pytest.fixture
def conv():
    return Convolution(resolution=lambda x: FWHM)


class TestConvolutionOutputs:

    def test_returns_three_arrays(self, conv):
        result = conv.apply(AXIS, np.ones(N) * 100.0)
        assert len(result) == 3

    def test_output_shapes(self, conv):
        c, sigma, n_sigma = conv.apply(AXIS, np.ones(N) * 100.0)
        assert c.shape == (N,)
        assert sigma.shape == (N,)
        assert n_sigma.shape == (N,)

    def test_sigma_non_negative(self, conv):
        _, sigma, _ = conv.apply(AXIS, np.ones(N) * 50.0)
        assert np.all(sigma >= 0)

    def test_n_sigma_is_ratio(self, conv):
        counts = np.ones(N) * 100.0
        c, sigma, n_sigma = conv.apply(AXIS, counts)
        mask = sigma > 0
        np.testing.assert_allclose(n_sigma[mask], c[mask] / sigma[mask])


class TestConvolutionSignal:

    def test_flat_spectrum_near_zero_response(self, conv):
        # Zero-area kernel on a flat spectrum → convolution ≈ 0
        counts = np.ones(N) * 200.0
        c, _, _ = conv.apply(AXIS, counts)
        # Interior points (away from edges) should be ≈ 0
        assert np.abs(c[50:-50]).mean() < 1.0

    def test_peak_gives_positive_response(self):
        # A peak at center should produce a strong positive response
        center = N // 2
        counts = np.zeros(N)
        counts[center - 5: center + 5] = 1000.0
        c = Convolution(resolution=lambda x: FWHM).apply(AXIS, counts)[0]
        assert c[center] > 0

    def test_poisson_variance_propagation(self, conv):
        # Poisson counts → sigma should scale with sqrt(counts)
        counts_low  = np.ones(N) * 10.0
        counts_high = np.ones(N) * 1000.0
        _, sigma_low,  _ = conv.apply(AXIS, counts_low)
        _, sigma_high, _ = conv.apply(AXIS, counts_high)
        # Higher counts → larger absolute sigma (more Poisson noise to propagate)
        assert sigma_high.mean() > sigma_low.mean()
