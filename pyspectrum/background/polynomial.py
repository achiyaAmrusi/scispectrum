import numpy as np
import xarray as xr
from pyspectrum.background.base import BackgroundEstimator

class IterativePolyFit(BackgroundEstimator):
    """
    Implementation of the "Baseline correction by improved iterative polynomial fitter with
automatic threshold"
    method by Gan, Ruan, and Mo (2006).
    This method use a polynomial to fit the backgroumd.
    Note that the bg can be negative
    """
    def __init__(self, degree=5, max_iter=100, tolerance=0.001):
        self.degree = degree
        self.max_iter = max_iter
        self.tolerance = tolerance

    def estimate(self, x, y):        
        # Step 0: Initial signal y_k starts as original signal
        y_work = y.copy()
        last_fit = np.zeros_like(y)
        
        for i in range(self.max_iter):
            # Step 1 & 2: Calculate the polynomial fitter result (b_k)
            # This is the 'calculate the b' part
            coeffs = np.polyfit(x, y_work, self.degree)
            current_fit = np.polyval(coeffs, x)
            
            # Step 4: Check Convergence Criterion (p < 0.001)
            # p = ||b_k - b_{k-1}|| / ||b_{k-1}||
            if i > 0:
                numerator = np.linalg.norm(current_fit - last_fit)
                denominator = np.linalg.norm(last_fit)
                if denominator > 0 and (numerator / denominator) < self.tolerance:
                    break
            
            last_fit = current_fit
            
            # Step 3: Compare fit with signal (Automatic Thresholding)
            # If signal > fit, set signal = fit (cutting out the peaks)
            y_work = np.minimum(y_work, current_fit)

        # The paper defines the final 'current_fit' as the estimated baseline
        return current_fit

class IterativePolyFitWithMinimum(BackgroundEstimator):
    """
    Improved Iterative Polynomial Fitting (Gan et al. 2006)
    + local minimum value for fitter.
    The minimum is taken differently in high and low SNR regions.
    Low SNR - minimal count.
    High SNR - filled using the minimum of surrounding values.
    """

    def __init__(
            self,
            degree=5,
            max_iter=100,
            tolerance=1e-3,
            window_scale=5.0,
            snr_threshold=4.0,
    ):
        self.degree = degree
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.window_scale = window_scale
        self.snr_threshold = snr_threshold

    # -------------------------------------------------
    # Fill high-SNR gaps using local minimum propagation
    # -------------------------------------------------
    def _fill_with_side_min(self, bg, x, resolution):
        """
        Fill zero regions using the minimum of the nearest valid
        values from left and right sides.

        This preserves the local envelope instead of collapsing
        to deep minima inside large windows.
        """
        n = len(bg)

        # Precompute nearest valid values from left
        left_vals = np.full(n, np.nan)
        last_val = np.nan

        for i in range(n):
            if bg[i] > 0:
                last_val = bg[i]
            left_vals[i] = last_val

        # Precompute nearest valid values from right
        right_vals = np.full(n, np.nan)
        last_val = np.nan

        for i in range(n - 1, -1, -1):
            if bg[i] > 0:
                last_val = bg[i]
            right_vals[i] = last_val

        # Combine
        filled = bg.copy()

        for i in range(n):
            if bg[i] > 0:
                continue

            l = left_vals[i]
            r = right_vals[i]

            if np.isfinite(l) and np.isfinite(r):
                filled[i] = min(l, r)
            elif np.isfinite(l):
                filled[i] = l
            elif np.isfinite(r):
                filled[i] = r
            else:
                filled[i] = 0.0  # fallback

        return filled

    # -------------------------------------------------
    # Minimum envelope estimation
    # -------------------------------------------------
    def _estimate_minimum_vector(self, x, y, resolution, conv):
        dx = x[1] - x[0]

        _, _, n_sigma = conv.apply(x, y)

        fwhm = np.array([resolution(xi) for xi in x])
        sigma_pts = np.maximum((fwhm / 2.355) / dx, 1.0)

        bg_min = np.zeros_like(y)

        for i in range(len(x)):
            # LOW SNR -> local minimum in window
            if np.abs(n_sigma[i]) < self.snr_threshold:
                sigma = sigma_pts[i]
                radius = int(self.window_scale * sigma)

                i_start = max(0, i - radius)
                i_end = min(len(y), i + radius + 1)

                bg_min[i] = np.min(y[i_start:i_end])

            # HIGH SNR → leave zero for now (will fill later)

        # Fill high-SNR gaps using sides
        bg_min = self._fill_with_side_min(bg_min, x, resolution)

        return bg_min

    # -------------------------------------------------
    # Main estimator
    # -------------------------------------------------
    def estimate(self, x, y, resolution, conv):
        y = y.astype(float)

        # Minimum envelope
        y_min = self._estimate_minimum_vector(x, y, resolution, conv)

        # Iterative polynomial
        y_work = y.copy()
        last_fit = np.zeros_like(y)

        for i in range(self.max_iter):
            coeffs = np.polyfit(x, y_work, self.degree)
            current_fit = np.polyval(coeffs, x)

            # convergence
            if i > 0:
                num = np.linalg.norm(current_fit - last_fit)
                den = np.linalg.norm(last_fit)
                if den > 0 and (num / den) < self.tolerance:
                    break

            last_fit = current_fit

            # clip peaks
            y_work = np.minimum(y_work, current_fit)

            # enforce minimum envelope
            y_work = np.maximum(y_work, y_min)

        return current_fit