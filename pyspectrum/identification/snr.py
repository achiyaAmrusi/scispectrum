# indentification/snr.py
import numpy as np
from pyspectrum.identification.base import DomainFinder
from pyspectrum.core.domain import Domain


class SNRFinder(DomainFinder):
    """
    Identify signal-containing domains in a spectrum using an SNR-based method.

    The algorithm detects statistically significant signal regions by computing
    a signal-to-noise ratio (SNR) spectrum using a zero-area convolution. Any
    location where the SNR exceeds a threshold is considered a candidate signal
    point. Each detection is then expanded into a contiguous domain that likely
    contains the full peak or signal structure.

    Domain expansion continues until the SNR remains below the detection
    threshold over a characteristic window determined by the detector
    resolution (FWHM).

    This approach is robust against small oscillations in the SNR spectrum,
    which may occur due to statistical fluctuations or convolution ringing.

    The resulting domains represent contiguous spectral regions that contain
    statistically significant signal and can be used for further analysis
    (e.g. peak fitter, background modeling, or Doppler integration).
    """

    def __init__(self, convolution, n_sigma_signal_threshold=4.0, n_sigma_bg_threshold=2.0 , persistence_factor=0.5):
        """
        Parameters
        ----------
        convolution : Convolution
            Convolution object used to compute the SNR spectrum.

        n_sigma_signal_threshold : float, optional
            SNR threshold used to detect candidate signal points.

        n_sigma_bg_threshold : float, optional
            SNR threshold used to determine the peak domain.
        persistence_factor : float, optional
            Fraction of the detector FWHM used as a persistence window when
            determining the domain edges.

            When expanding a detected peak, the algorithm requires that the
            SNR remains below `n_sigma_signal_threshold` for a continuous region of
            width:

                persistence_factor × FWHM

            before terminating the domain. This prevents premature stopping
            caused by small oscillations in the SNR spectrum.
            Note that the edge is not sensitive to padding_factor if there is no other close peak.

            Default is 0.5.
        """

        self.convolution = convolution
        self.n_sigma_signal_threshold = n_sigma_signal_threshold
        self.n_sigma_bg_threshold = n_sigma_bg_threshold
        self.persistence_factor = persistence_factor

        self._cached_spectrum = None
        self._cached_n_sigma = None

    def _get_n_sigma(self, spectrum):

        if spectrum is not self._cached_spectrum:
            _, _, n_sigma = self.convolution.apply(
                spectrum.axis,
                spectrum.counts
            )

            self._cached_spectrum = spectrum
            self._cached_n_sigma = n_sigma

        return self._cached_n_sigma

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
        n_sigma = self._get_n_sigma(spectrum)
        daxis = spectrum.axis[1] - spectrum.axis[0]

        domains = []
        idx = 0
        n = len(n_sigma)

        while idx < n:
            if n_sigma[idx] >= self.n_sigma_signal_threshold:
                lo, hi = self._expand_from_index(spectrum, n_sigma, idx)
                ch_fwhm = spectrum.resolution_calib(spectrum.axis[idx]) / daxis
                ch_fwhm = np.max([int(round(ch_fwhm)), 1])

                if (hi-lo)>=ch_fwhm:
                    domains.append(Domain(spectrum, lo, hi))
                idx = hi + 1  # skip processed region
            else:
                idx += 1

        return domains

    def _expand_from_index(self, spectrum, n_sigma, index):
        """
        Expand a detected SNR crossing into a full domain.

        Starting from a detection index, the domain is extended to lower
        and higher channels until:
        the SNR drops below a relaxed threshold for at least the pressistence distance which is
        persistence_factor × FWHM

        Parameters
        ----------
        spectrum : Spectrum
            Spectrum providing axis calibration and detector resolution.
        n_sigma : ndarray
            SNR array from convolution.
        index : int
            Index at which the detection threshold was crossed.

        Returns
        -------
        tuple[int, int]
            Inclusive (lo, hi) indices defining the domain.
        """

        if n_sigma[index]<self.n_sigma_signal_threshold:
            raise ValueError("value at index dont cross threshold")

        n = n_sigma.shape[0]
        daxis = spectrum.axis[1] - spectrum.axis[0]
        # Convert FWHM to approximate number of channels
        axis_center = spectrum.axis[index]
        ch_fwhm = spectrum.resolution_calib(axis_center)/ daxis
        ch_fwhm = np.max([int(round(ch_fwhm)), 1])
        persistence = np.max([int(self.persistence_factor * ch_fwhm), 1])

        # --- Extend left ---
        lo = index
        while lo > 0:
            persistence_left = np.max([0, lo-persistence])
            if np.all(np.abs(n_sigma[persistence_left:lo]) < self.n_sigma_bg_threshold):
                break
            lo -= 1

        # --- Extend right ---
        hi = index
        while hi < n - 1:
            persistence_right = np.min([n-1,hi+persistence])
            if np.all(np.abs(n_sigma[hi:persistence_right]) < self.n_sigma_bg_threshold):
                hi -= 1 # to aligen hi
                break
            hi += 1
        return lo, hi

    def domain(self, spectrum, axis_value):
        """
        Return the signal domain around a given axis value.

        Parameters
        ----------
        spectrum : Spectrum
            Spectrum to analyze.
        axis_value : float
            Axis value (e.g. energy) around which to find the peak.

        Returns
        -------
        Domain
            Domain containing the peak.
        """

        n_sigma = self._get_n_sigma(spectrum)

        # convert axis value to index
        center = np.argmin(np.abs(spectrum.axis - axis_value))

        lo, hi = self._expand_from_index(spectrum, n_sigma, center)

        return Domain(spectrum, lo, hi)