import numpy as np
from pyspectrum.identification.kernels.mexican_hat import gaussian_2_dev
from pyspectrum.utils.gaussian import GAUSSIAN_FWHM_TO_SIGMA

class Convolution:
    """
    Convolution with a zero-area kernel of variable width.

    This class performs a localized convolution suitable for
    peak/domain detection in Poisson-limited spectra.

    Parameters
    ----------
    width : Callable
        Function width(x) -> FWHM at domain value x
    kernel : Callable
        Zero-area kernel function k(x, center, fwhm)
    window_fwhm : float
        How many FWHMs to include around the center for the kernel window (default=3)
    """

    def __init__(self, width, kernel=gaussian_2_dev, window_fwhm=3.0):
        self.width = width
        self.kernel = kernel
        self.window_fwhm = window_fwhm

    def _kernel(self, domain, center_idx):
        """
        Compute kernel values on a local window.
        """
        x0 = domain[center_idx]
        fwhm = self.width(x0)

        dx = domain[1] - domain[0]
        integration_radius = int(self.window_fwhm * fwhm / dx)

        low = max(0, center_idx - integration_radius)
        high = min(len(domain), center_idx + integration_radius + 1)

        x = domain[low:high]
        k = self.kernel(x, x0, fwhm)

        return low, high, k

    def apply(self, domain, counts):
        """
        Apply convolution and compute significance.

        Parameters
        ----------
        domain : np.ndarray
            Domain axis (channels or energy)
        counts : np.ndarray
            Spectrum counts (Poisson statistics assumed)

        Returns
        -------
        conv : np.ndarray
            Convolution response
        sigma : np.ndarray
            Standard deviation of convolution
        n_sigma : np.ndarray
            |conv| / sigma
        """
        n = len(domain)
        conv = np.zeros(n, dtype=float)
        var = np.zeros(n, dtype=float)

        for i in range(n):
            low, high, k = self._kernel(domain, i)

            y = counts[low:high]
            conv[i] = np.dot(k, y)

            # Poisson variance propagation
            var[i] = np.dot(k * k, y)

        sigma = np.sqrt(var)
        n_sigma = np.zeros_like(conv)
        mask = sigma > 0
        n_sigma[mask] = np.abs(conv[mask] / sigma[mask])

        return conv, sigma, n_sigma
