import numpy as np
import xarray as xr
from pyspectrum.peak import Peak
from scipy.optimize import curve_fit
from .calbration_functions import germanium_fwhm, standard_fwhm_generator


class Calibration:
    """
    This class is a tool to calibrate detector spectra fwhm and energy.
    Function get a spectrum with known peaks, the known energies and estimated domain for the peaks in those energies.
    Using these the function can fit the energy and fwhm calibrations.

    """

    def __init__(self, spectrum: xr.DataArray, photopeak_energies: list, photopeak_estimated_domain: list,
                 detector_fwhm_function=germanium_fwhm, fwhm_generator=standard_fwhm_generator):
        self.spectrum = spectrum.rename({spectrum.dims[0]: 'channel'})
        self.photopeak_domains = photopeak_estimated_domain
        self.photopeak_energies = photopeak_energies
        self.detector_fwhm_function = detector_fwhm_function
        self.fwhm_generator = fwhm_generator

    def estimate_fwhm_and_center(self):
        channel_centers = np.zeros_like(self.photopeak_energies)
        channel_fwhm = np.zeros_like(self.photopeak_energies)
        for i, domain in enumerate(self.photopeak_domains):
            peak = self.spectrum.sel(channel=slice(domain[0], domain[1]))
            channel_centers[i], channel_fwhm[i] = Peak.center_fwhm_estimator(peak)
        return channel_centers, channel_fwhm

    def generate_calibration(self, degree_of_poly: int, p0=None):
        """
        function optimize the energy calibration polynomial and proximate fwhm function.
        todo: change the estimation of the center and fwhm into fitting method
        """
        if p0 is None:
            p0 = [0, 0.02, 0]

        # estimate centers and fwhm
        peaks_center_ch, peaks_fwhm_ch = self.estimate_fwhm_and_center()
        # calibrate energy
        energy_calib = np.poly1d(np.polyfit(peaks_center_ch, self.photopeak_energies, deg=degree_of_poly))
       # express fwhm in energy
        fwhm_val = (energy_calib(peaks_center_ch+peaks_fwhm_ch/2) - energy_calib(peaks_center_ch-peaks_fwhm_ch/2))
        # calibrate fwhm
        fwhm_calib = self._fit_fwhm(self.photopeak_energies, fwhm_val, p0=p0)

        return energy_calib, fwhm_calib

    def _fit_fwhm(self, energies, estimated_fwhm, p0):
        """
        optimizing approximated fwhm function
        """
        parm = curve_fit(f=self.detector_fwhm_function, xdata=energies, ydata=estimated_fwhm, p0=p0)
        return self.fwhm_generator(parm[0])

