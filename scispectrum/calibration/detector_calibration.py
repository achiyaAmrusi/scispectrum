import numpy as np
from typing import List, Tuple

from scispectrum.core.spectrum import Spectrum
from scispectrum.core.domain import Domain
from scispectrum.domain_analysis.single_peak import center_estimator, fwhm_estimator
from scispectrum.calibration.axis import AxisCalibration
from scispectrum.calibration.resolution import ResolutionCalibration
from scispectrum.calibration.models.energy_poly import PolynomialEnergyModel
from scispectrum.calibration.models.hpge_fwhm_model import StandardHPGeFWHMModel

class DetectorCalibration:
    """
    High-level helper for HPGe detector energy and resolution calibration.

    This class provides a structured workflow for calibrating a detector
    using a measured spectrum and a set of known reference features
    (typically gamma-ray lines or well-defined spectral peaks).

    The calibration proceeds in three conceptual stages:

    1. Peak characterization
       Each provided peak domain is analyzed to estimate its centroid
       (in channel units) and its full width at half maximum (FWHM).

    2. Axis (energy) calibration
       A parametric axis calibration model is fitted to map detector
       channels to physical axis values (e.g. energy).

    3. Resolution calibration
       A resolution model is fitted to describe the detector FWHM
       as a function of the calibrated axis.

    The result is a pair of calibration objects that can be attached
    to a Spectrum instance for downstream analysis.
    """

    def __init__(
        self,
        spectrum: Spectrum,
        known_axis_values: List[float],
        peak_domains: List[Tuple[int, int]],
        *,
        energy_model=None,
        fwhm_model=None,
    ):
        """
        Parameters
        ----------
        spectrum : Spectrum
            Measured spectrum containing the calibration peaks.
            `generate()` always characterizes peaks in channel units
            internally, regardless of any axis calibration already
            attached to this spectrum. `estimate_peaks()`, on the other
            hand, reports in whatever axis units the spectrum currently
            has — see its docstring.

        known_axis_values : list of float
            Known physical axis values corresponding to each peak domain.
            Typical examples include gamma-ray energies in keV.
            The ordering must match `peak_domains`.

        peak_domains : list of (int, int)
            Inclusive (lo, hi) channel index ranges defining regions
            that contain isolated calibration peaks.

        energy_model : BaseCalibrationModel, optional
            Parametric model used for axis calibration.
            Defaults to a second-order polynomial energy model.

        fwhm_model : BaseCalibrationModel, optional
            Parametric model used for detector resolution calibration.
            Defaults to a standard HPGe FWHM model.
        """
        self.spectrum = spectrum
        self.known_axis = np.asarray(known_axis_values)
        self.peak_domains = peak_domains

        self.energy_model = energy_model or PolynomialEnergyModel(degree=2)
        self.fwhm_model = fwhm_model or StandardHPGeFWHMModel()

    def estimate_peaks(self):
        """
        Estimate peak centroids and widths from the provided peak domains.

        Each peak domain is sliced from `self.spectrum` and analyzed using
        a model-independent estimator to obtain the peak center and FWHM,
        reported in whatever axis units `self.spectrum` currently has
        (channels, if no calibration has been set on it yet; the calibrated
        axis otherwise — since `self.spectrum` is held by reference, calling
        this again after `spectrum.set_axis_calibration(...)` reports in the
        newly calibrated units). This lets the same peak domains be
        re-measured directly in calibrated units for verification, without
        going through the fitted calibration function.

        Use this to verify a calibration already applied to the spectrum;
        `generate()` (below) always characterizes peaks in raw channel units
        internally, so it isn't affected by the spectrum's calibration state.

        Returns
        -------
        centers : ndarray
            Estimated peak centroids, in the spectrum's current axis units.

        fwhms : ndarray
            Estimated peak full widths at half maximum, in the same units.
        """
        centers = []
        fwhms = []

        for lo, hi in self.peak_domains:
            domain = Domain(self.spectrum, start=lo, stop=hi)
            c = center_estimator(domain)
            f = fwhm_estimator(domain)
            centers.append(float(c))
            fwhms.append(float(f))

        return np.asarray(centers), np.asarray(fwhms)

    def _estimate_peaks_channels(self):
        """Like `estimate_peaks`, but always in raw channel units, regardless
        of any axis calibration already attached to `self.spectrum`. Used by
        `generate()` so axis-calibration fitting can never be contaminated by
        a spectrum that was calibrated before (or during) this object's
        lifetime."""
        channel_spectrum = Spectrum(counts=self.spectrum.counts, counts_err=self.spectrum.counts_err)

        centers = []
        fwhms = []

        for lo, hi in self.peak_domains:
            domain = Domain(channel_spectrum, start=lo, stop=hi)
            c = center_estimator(domain)
            f = fwhm_estimator(domain)
            centers.append(float(c))
            fwhms.append(float(f))

        return np.asarray(centers), np.asarray(fwhms)

    def generate(self, energy_p0=None, fwhm_p0=None):
        """
        Generate axis and resolution calibration objects.

        This method performs the full calibration pipeline:
        - estimate peak parameters
        - fit axis calibration model
        - convert FWHM values to calibrated axis units
        - fit resolution calibration model

        Parameters
        ----------
        energy_p0 : sequence of float, optional
            Initial parameter guess for the energy calibration fit.

        fwhm_p0 : sequence of float, optional
            Initial parameter guess for the resolution calibration fit.

        Returns
        -------
        axis_calibration : AxisCalibration
            Callable calibration mapping detector channels to
            physical axis units (e.g. energy).

        resolution_calibration : ResolutionCalibration
            Callable calibration describing detector resolution
            (FWHM) as a function of the calibrated axis.
        """
        centers_ch, fwhm_ch = self._estimate_peaks_channels()

        # --- Energy calibration ---
        e_params, _ = self.energy_model.fit(
            x=centers_ch,
            y=self.known_axis,
            p0=energy_p0)
        energy_func = self.energy_model.generator(e_params)
        axis_calib = AxisCalibration(energy_func, name="energy")

        # --- Convert FWHM from channels to axis units ---
        axis = axis_calib.apply(centers_ch)
        fwhm_axis = (
            axis_calib.apply(centers_ch + fwhm_ch / 2)
            - axis_calib.apply(centers_ch - fwhm_ch / 2)
        )

        # --- Resolution calibration ---
        r_params, _ = self.fwhm_model.fit(axis, fwhm_axis, p0=fwhm_p0)
        res_func = self.fwhm_model.generator(r_params)
        resolution_calib = ResolutionCalibration(res_func)

        return axis_calib, resolution_calib

