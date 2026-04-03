import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter1d
from pyspectrum.background.base import BackgroundEstimator

class MinimaEnvelopeBackground(BackgroundEstimator):
    """
    Background estimation using:
    - SNR masking
    - local Gaussian averaging
    - minimum propagation
    - final smoothing
    it estimate the signal in each iteration by:
     - if the data snr < threshold -> the mean value in an envelope proportional to the resolutin
     - if the data snr > threshold -> the values from the sides
     in each iteration the peaks means out.
     The method works well for gamma spectroscopy and i didn't tried for other purposes.
    This is a non-parametric background estimator and gives a good initial background estimation.
    """

    def __init__(self, window_scale=5.0, snr_threshold=4.0):
        self.window_scale = window_scale
        self.snr_threshold = snr_threshold

    # -------------------------------------------------
    # Fill missing regions using local minimum
    # -------------------------------------------------
    def _fill_with_local_min(self, initial_bg, x, resolution):
        dx = x[1] - x[0]
        n = len(initial_bg)

        fwhm = np.array([resolution(xi) for xi in x])
        radius_pts = np.maximum((self.window_scale * fwhm / dx).astype(int), 1)

        # Left -> right
        left_fill = initial_bg.copy()
        for i in range(n):
            if left_fill[i] > 0:
                continue
            window = left_fill[max(0, i - radius_pts[i]):i]
            valid = window[window > 0]
            if len(valid):
                left_fill[i] = np.min(valid)

        # Right -> left
        right_fill = initial_bg.copy()
        for i in range(n - 1, -1, -1):
            if right_fill[i] > 0:
                continue
            window = right_fill[i + 1:min(n, i + radius_pts[i] + 1)]
            valid = window[window > 0]
            if len(valid):
                right_fill[i] = np.min(valid)

        # Combine
        filled = np.minimum(
            np.where(left_fill > 0, left_fill, np.inf),
            np.where(right_fill > 0, right_fill, np.inf),
        )

        # Fallback
        valid_vals = initial_bg[initial_bg > 0]
        fallback = np.min(valid_vals) if len(valid_vals) else 0.0
        filled[~np.isfinite(filled)] = fallback

        return filled

    # -------------------------------------------------
    # Main estimator
    # -------------------------------------------------

    def estimate(self, axis, counts, resolution_calib, conv, iterations):

        dx = axis[1] - axis[0]
        fwhm = np.array([resolution_calib(xi) for xi in axis])
        sigma_pts = np.maximum((fwhm / 2.355) / dx, 1.0)

        new_bg_estimation = counts.copy()

        for i in range(iterations):
            previous_bg_estimation = new_bg_estimation.copy()
            new_bg_estimation = np.zeros_like(counts)

            _, _, n_sigma = conv.apply(axis, previous_bg_estimation)

            # --- Local Gaussian estimate ---
            for i in range(len(axis)):
                if n_sigma[i] >= self.snr_threshold:
                    continue

                sigma = sigma_pts[i]
                radius = int(self.window_scale * sigma)

                i_start = max(0, i - radius)
                i_end = min(len(counts), i + radius + 1)

                idx = np.arange(i_start, i_end)

                w = np.exp(-0.5 * ((idx - i) / sigma) ** 2)
                w /= w.sum()

                new_bg_estimation[i] = np.sum(w * previous_bg_estimation[i_start:i_end])

            # --- Fill ---
            new_bg_estimation = self._fill_with_local_min(
                new_bg_estimation, axis, resolution_calib
            )

            # --- Smooth ---
            sigma_global = np.mean(sigma_pts)
            new_bg_estimation = gaussian_filter1d(new_bg_estimation, sigma=sigma_global)

        return new_bg_estimation