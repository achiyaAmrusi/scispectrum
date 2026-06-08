"""Tests for identification kernels."""

import numpy as np
import pytest

from scispectrum.identification.kernels.mexican_hat import gaussian_2_dev
from scispectrum.identification.kernels.base import Kernel1D


# ---------------------------------------------------------------------------
# Concrete subclass used only for testing the ABC helpers
# ---------------------------------------------------------------------------

class _GaussianKernel(Kernel1D):
    """Minimal concrete kernel: unnormalised Gaussian."""
    def values(self):
        return np.exp(-0.5 * (self.x / self.sigma) ** 2)


class TestKernel1D:

    def test_half_width(self):
        k = _GaussianKernel(sigma=2.0, support_sigma=3.0)
        assert k.half_width == int(np.ceil(3.0 * 2.0))

    def test_x_grid_length(self):
        k = _GaussianKernel(sigma=2.0, support_sigma=3.0)
        assert len(k.x) == 2 * k.half_width + 1

    def test_x_grid_symmetric(self):
        k = _GaussianKernel(sigma=2.0, support_sigma=3.0)
        np.testing.assert_array_equal(k.x, -k.x[::-1])

    def test_normalized_l1_norm(self):
        k = _GaussianKernel(sigma=3.0)
        v = k.normalized()
        assert abs(np.sum(np.abs(v)) - 1.0) < 1e-10

    def test_normalized_zero_kernel_unchanged(self):
        class _ZeroKernel(Kernel1D):
            def values(self):
                return np.zeros_like(self.x, dtype=float)
        k = _ZeroKernel(sigma=2.0)
        np.testing.assert_array_equal(k.normalized(), np.zeros_like(k.x, dtype=float))


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
