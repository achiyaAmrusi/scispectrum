import numpy as np
from typing import Callable
from pyspectrum.background.base import BackgroundEstimator


class IterativePolyFit(BackgroundEstimator):
    """
    Iterative polynomial baseline correction (Gan et al. 2006).

    Fits a polynomial to the spectrum, clips peaks above the fit,
    and repeats until convergence. The final polynomial is the background.
    Note: the background can be negative.

    Parameters
    ----------
    degree : int
        Polynomial degree. Default is 5.
    max_iter : int
        Maximum number of iterations. Default is 100.
    tolerance : float
        Convergence threshold (relative change in fit norm). Default is 0.001.
    """

    def __init__(self, degree=5, max_iter=100, tolerance=0.001):
        self.degree = degree
        self.max_iter = max_iter
        self.tolerance = tolerance

    def estimate(self, axis: np.ndarray, counts: np.ndarray) -> np.ndarray:
        y_work = counts.copy()
        last_fit = np.zeros_like(counts)

        for i in range(self.max_iter):
            coeffs = np.polyfit(axis, y_work, self.degree)
            current_fit = np.polyval(coeffs, axis)

            if i > 0:
                numerator = np.linalg.norm(current_fit - last_fit)
                denominator = np.linalg.norm(last_fit)
                if denominator > 0 and (numerator / denominator) < self.tolerance:
                    break

            last_fit = current_fit
            y_work = np.minimum(y_work, current_fit)

        return current_fit


class IterativePolyFitWithMinimum(BackgroundEstimator):
    """
    Iterative polynomial baseline (Gan et al. 2006) with SNR-gated minimum envelope.

    In low-SNR regions, the working signal is floored by the local minimum within
    a resolution-scaled window. High-SNR regions (peaks) are filled from their
    neighbours to avoid collapsing the baseline.

    Parameters
    ----------
    resolution : callable
        Maps axis values to FWHM in the same units.
    conv : Convolution
        Convolution object used to compute the local SNR map.
    degree : int
        Polynomial degree. Default is 5.
    max_iter : int
        Maximum iterations. Default is 100.
    tolerance : float
        Convergence threshold. Default is 1e-3.
    window_scale : float
        Half-window size as a multiple of sigma. Default is 5.0.
    snr_threshold : float
        SNR above which a channel is treated as a peak. Default is 4.0.
    """

    def __init__(
        self,
        resolution: Callable[[float], float],
        conv,
        degree=5,
        max_iter=100,
        tolerance=1e-3,
        window_scale=5.0,
        snr_threshold=4.0,
    ):
        self.resolution = resolution
        self.conv = conv
        self.degree = degree
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.window_scale = window_scale
        self.snr_threshold = snr_threshold

    def _fill_with_side_min(self, bg: np.ndarray) -> np.ndarray:
        n = len(bg)

        left_vals = np.full(n, np.nan)
        last_val = np.nan
        for i in range(n):
            if bg[i] > 0:
                last_val = bg[i]
            left_vals[i] = last_val

        right_vals = np.full(n, np.nan)
        last_val = np.nan
        for i in range(n - 1, -1, -1):
            if bg[i] > 0:
                last_val = bg[i]
            right_vals[i] = last_val

        filled = bg.copy()
        for i in range(n):
            if bg[i] > 0:
                continue
            l, r = left_vals[i], right_vals[i]
            if np.isfinite(l) and np.isfinite(r):
                filled[i] = min(l, r)
            elif np.isfinite(l):
                filled[i] = l
            elif np.isfinite(r):
                filled[i] = r
            else:
                filled[i] = 0.0

        return filled

    def _estimate_minimum_vector(self, axis: np.ndarray, counts: np.ndarray) -> np.ndarray:
        dx = axis[1] - axis[0]
        _, _, n_sigma = self.conv.apply(axis, counts)

        fwhm = np.array([self.resolution(xi) for xi in axis])
        sigma_pts = np.maximum((fwhm / 2.355) / dx, 1.0)

        bg_min = np.zeros_like(counts)
        for i in range(len(axis)):
            if np.abs(n_sigma[i]) < self.snr_threshold:
                sigma = sigma_pts[i]
                radius = int(self.window_scale * sigma)
                i_start = max(0, i - radius)
                i_end = min(len(counts), i + radius + 1)
                bg_min[i] = np.min(counts[i_start:i_end])

        return self._fill_with_side_min(bg_min)

    def estimate(self, axis: np.ndarray, counts: np.ndarray) -> np.ndarray:
        counts = counts.astype(float)
        y_min = self._estimate_minimum_vector(axis, counts)

        y_work = counts.copy()
        last_fit = np.zeros_like(counts)

        for i in range(self.max_iter):
            coeffs = np.polyfit(axis, y_work, self.degree)
            current_fit = np.polyval(coeffs, axis)

            if i > 0:
                num = np.linalg.norm(current_fit - last_fit)
                den = np.linalg.norm(last_fit)
                if den > 0 and (num / den) < self.tolerance:
                    break

            last_fit = current_fit
            y_work = np.minimum(y_work, current_fit)
            y_work = np.maximum(y_work, y_min)

        return current_fit
