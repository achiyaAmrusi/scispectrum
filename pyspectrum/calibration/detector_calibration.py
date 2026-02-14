import numpy as np
import xarray as xr
from typing import List, Tuple

from pyspectrum.core.spectrum import Spectrum
from pyspectrum.core.peak import Peak
from pyspectrum.calibration.axis import AxisCalibration
from pyspectrum.calibration.resolution import ResolutionCalibration
from pyspectrum.calibration.models.energy_poly import PolynomialEnergyModel
from pyspectrum.calibration.models.hpge_fwhm_model import StandardHPGeFWHMModel

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
            The spectrum is assumed to be indexed in detector channels.

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

        Each peak domain is sliced from the spectrum and analyzed using
        a model-independent estimator to obtain:
        - the peak center (in channel units)
        - the peak FWHM (in channel units)

        Returns
        -------
        centers : ndarray
            Estimated peak centroids in detector channels.

        fwhms : ndarray
            Estimated peak full widths at half maximum in channels.
        """
        centers = []
        fwhms = []

        for lo, hi in self.peak_domains:
            peak = xr.DataArray(self.spectrum.counts[lo:hi],
                                coords={'channel':np.arange(lo,hi)})
            c, f = Peak.center_fwhm_estimator(peak)
            centers.append(c)
            fwhms.append(f)

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
        centers_ch, fwhm_ch = self.estimate_peaks()

        # --- Energy calibration ---
        e_params, _ = self.energy_model.fit(
            centers_ch,
            self.known_axis,
            p0=energy_p0,
        )
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

