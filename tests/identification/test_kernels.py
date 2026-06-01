"""Tests for identification kernels."""

import numpy as np
import pytest

from pyspectrum.identification.kernels.mexican_hat import gaussian_2_dev


class TestGaussian2Dev:

    def test_output_shape(self):
        x = np.linspace(-10, 10, 200)
        k = gaussian_2_dev(x, mean=0.0, fwhm=2.0)
        assert k.shape == (200,)

    def test_zero_area(self):
        # Zero-area (Mexican-hat) kernel: integral should be ≈ 0
        x = np.linspace(-30, 30, 3000)
        k = gaussian_2_dev(x, mean=0.0, fwhm=4.0)
        assert abs(np.trapezoid(k, x)) < 1e-6

    def test_symmetric(self):
        x = np.linspace(-10, 10, 1001)
        k = gaussian_2_dev(x, mean=0.0, fwhm=3.0)
        np.testing.assert_allclose(k, k[::-1], atol=1e-12)

    def test_peak_at_center(self):
        x = np.linspace(-10, 10, 1001)
        k = gaussian_2_dev(x, mean=0.0, fwhm=3.0)
        # Maximum should be at center
        assert np.argmax(k) == len(x) // 2

    def test_shifts_with_mean(self):
        x = np.linspace(0, 20, 1001)
        k = gaussian_2_dev(x, mean=10.0, fwhm=2.0)
        center_idx = np.argmax(k)
        assert abs(x[center_idx] - 10.0) < 0.1
