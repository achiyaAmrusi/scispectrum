import numpy as np
from scipy.ndimage import gaussian_filter1d
from pyspectrum.background.base import BackgroundEstimator
from pyspectrum.calibration import ResolutionCalibration


class MinimaEnvelopeBackground(BackgroundEstimator):
    """
    Non-parametric SNR-gated minimum envelope background estimator.

    In each iteration, low-SNR channels are replaced by a resolution-weighted
    local Gaussian average; high-SNR channels (peaks) are filled from their
    neighbours. Converges to a smooth background beneath all peaks.
    Works well for gamma spectroscopy.

    Parameters
    ----------
    resolution_calib : ResolutionCalibration
        Maps axis values to FWHM in the same units.
    conv : Convolution
        Convolution object used to compute the local SNR map.
    iterations : int
        Number of estimation iterations. Default is 20.
    window_scale : float
        Half-window size as a multiple of sigma. Default is 5.0.
    snr_threshold : float
        SNR above which a channel is treated as a peak. Default is 4.0.
    """

    def __init__(
        self,
        resolution_calib: ResolutionCalibration,
        conv,
        iterations: int = 20,
        window_scale: float = 5.0,
        snr_threshold: float = 4.0,
    ):
        self.resolution_calib = resolution_calib
        self.conv = conv
        self.iterations = iterations
        self.window_scale = window_scale
        self.snr_threshold = snr_threshold

    def _fill_with_local_min(self, initial_bg: np.ndarray, axis: np.ndarray) -> np.ndarray:
        dx = axis[1] - axis[0]
        n = len(initial_bg)

        fwhm = np.array([self.resolution_calib(xi) for xi in axis])
        radius_pts = np.maximum((self.window_scale * fwhm / dx).astype(int), 1)

        left_fill = initial_bg.copy()
        for i in range(n):
            if left_fill[i] > 0:
                continue
            window = left_fill[max(0, i - radius_pts[i]):i]
            valid = window[window > 0]
            if len(valid):
                left_fill[i] = np.min(valid)

        right_fill = initial_bg.copy()
        for i in range(n - 1, -1, -1):
            if right_fill[i] > 0:
                continue
            window = right_fill[i + 1:min(n, i + radius_pts[i] + 1)]
            valid = window[window > 0]
            if len(valid):
                right_fill[i] = np.min(valid)

        filled = np.minimum(
            np.where(left_fill > 0, left_fill, np.inf),
            np.where(right_fill > 0, right_fill, np.inf),
        )

        valid_vals = initial_bg[initial_bg > 0]
        fallback = np.min(valid_vals) if len(valid_vals) else 0.0
        filled[~np.isfinite(filled)] = fallback

        return filled

    def estimate(self, axis: np.ndarray, counts: np.ndarray) -> np.ndarray:
        """
        Estimate background using the minimum envelope method.

        Parameters
        ----------
        axis : np.ndarray
            Axis values.
        counts : np.ndarray
            Spectrum counts.

        Returns
        -------
        np.ndarray
            Estimated background, same shape as counts.
        """
        dx = axis[1] - axis[0]
        fwhm = np.array([self.resolution_calib(xi) for xi in axis])
        sigma_pts = np.maximum((fwhm / 2.355) / dx, 1.0)

        new_bg = counts.copy()

        for _ in range(self.iterations):
            prev_bg = new_bg.copy()
            new_bg = np.zeros_like(counts)

            _, _, n_sigma = self.conv.apply(axis, prev_bg)

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
                new_bg[i] = np.sum(w * prev_bg[i_start:i_end])

            new_bg = self._fill_with_local_min(new_bg, axis)

            sigma_global = np.mean(sigma_pts)
            new_bg = gaussian_filter1d(new_bg, sigma=sigma_global)

        return new_bg
