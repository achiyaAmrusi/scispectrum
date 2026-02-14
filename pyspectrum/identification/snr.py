# indentification/snr.py
import numpy as np
from pyspectrum.identification.base import DomainFinder
from pyspectrum.core.domain import Domain


class SNRFinder(DomainFinder):
    """
    Find signal-containing spectral domains using an SNR-based convolution method.

    This class identifies contiguous regions of a spectrum that contain
    statistically significant signal above noise. It does not assume that
    the signal corresponds to a single peak or any specific physical process.

    The algorithm proceeds as follows:
    1. Compute a local signal-to-noise ratio (SNR) using a zero-area convolution.
    2. Detect candidate locations where the SNR exceeds a threshold.
    3. Expand each detection into a full domain using relaxed SNR criteria
       and detector resolution (FWHM).

    The resulting domains are suitable for further analysis such as peak
    fitting, background modeling, or PAS Doppler integration.
    """

    def __init__(self, convolution, n_sigma_threshold=4.0, extension_factor=0.5):
        """
        Parameters
        ----------
        convolution : Convolution
            Convolution object used to compute the SNR spectrum.
        n_sigma_threshold : float, optional
            SNR threshold used to detect candidate signal locations.
        extension_factor : float, optional
            Factor applied to the detection threshold when extending
            domain boundaries (default is 0.5).
        """
        self.convolution = convolution
        self.n_sigma_threshold = n_sigma_threshold
        self.extension_threshold = extension_factor * n_sigma_threshold

    def find(self, spectrum):
        """
        Identify all signal-containing domains in a spectrum.

        Parameters
        ----------
        spectrum : Spectrum
            Input spectrum with calibrated axis, counts, and
            FWHM calibration.

        Returns
        -------
        list[Domain]
            List of detected signal domains.
        """
        _, _, n_sigma = self.convolution.apply(
            spectrum.axis,
            spectrum.counts
        )

        domains = []
        idx = 0
        n = len(n_sigma)

        while idx < n:
            if n_sigma[idx] >= self.n_sigma_threshold:
                lo, hi = self._expand_from_index(spectrum, n_sigma, idx)
                domains.append(Domain(spectrum, lo, hi))
                idx = hi + 1  # skip processed region
            else:
                idx += 1

        return domains

    def _expand_from_index(self, spectrum, n_sigma, center):
        """
        Expand a detected SNR crossing into a full domain.

        Starting from a detection index, the domain is extended to lower
        and higher channels until:
        - the SNR drops below a relaxed threshold, and
        - at least one detector FWHM is covered on each side.

        Parameters
        ----------
        spectrum : Spectrum
            Spectrum providing axis calibration and detector resolution.
        n_sigma : ndarray
            SNR array from convolution.
        center : int
            Index at which the detection threshold was crossed.

        Returns
        -------
        tuple[int, int]
            Inclusive (lo, hi) indices defining the domain.
        """
        n = len(n_sigma)
        daxis = spectrum.axis[1] - spectrum.axis[0]
        # Convert FWHM to approximate number of channels
        axis_center = spectrum.axis[center]
        ch_fwhm = (
            spectrum.resolution_calib(axis_center)
            / daxis
        )
        ch_fwhm = max(int(round(ch_fwhm)), 1)

        # --- Extend left ---
        lo = center
        while lo > 0:
            if (
                n_sigma[lo] < self.extension_threshold
                and (center - lo) >= ch_fwhm
            ):
                break
            lo -= 1

        # --- Extend right ---
        hi = center
        while hi < n - 1:
            if (
                n_sigma[hi] < self.extension_threshold
                and (hi - center) >= ch_fwhm
            ):
                break
            hi += 1

        return lo, hi
