import numpy as np
import xarray as xr
from scipy.optimize import curve_fit
import warnings
from pyspectrum.core.domain import Domain
from pyspectrum.domain_analysis.morphology import  find_domain_peaks
from pyspectrum.utils.gaussian import fwhm_to_sigma
from pyspectrum.domain_fitting.abstract_fitting_class import PeakFit

class SumOfGaussians(PeakFit):

    def __init__(self):
        """No initialization needed for now."""

    @staticmethod
    def _flat_evaluate(axis: np.ndarray, *params) -> np.ndarray:
        """
        Wrapper around ``evaluate`` that accepts a flat parameter vector,
        as required by ``scipy.optimize.curve_fit``.

        Parameter order (repeating for each peak k):
            amplitude_0, fwhm_0, x0_0, amplitude_1, fwhm_1, x0_1, ...
        """

        params = np.asarray(params)
        a = params[0::3][:, np.newaxis]
        s = fwhm_to_sigma(params[1::3])[:, np.newaxis]
        m = params[2::3][:, np.newaxis]

        return (a * np.exp(-((axis - m) ** 2) / (2 * s ** 2))).sum(axis=0)

    @classmethod
    def evaluate(cls, axis: np.ndarray, fit_result: xr.Dataset) -> np.ndarray:
        """
        Evaluate a sum of Gaussians from a fit result Dataset.

        Parameters
        ----------
        axis : np.ndarray, shape (N,)
        fit_result : xr.Dataset
            As returned by ``fit``. Must contain ``amplitude``, ``fwhm``, ``center``.

        Returns
        -------
        np.ndarray, shape (N,)
        """


        p0 = np.empty(3 * len(fit_result["amplitude"].values))
        p0[0::3] = fit_result["amplitude"].values
        p0[1::3] = fit_result["fwhm"].values
        p0[2::3] = fit_result["center"].values

        return cls._flat_evaluate(axis, *p0)

    @staticmethod
    def _initial_guess(
        domain: Domain,
        *,
        smooth: bool = True,
        smoothing_fwhm_scale: float = 0.3,
        prominence: float | None = None,
    ) -> np.ndarray:
        """
        Build a flat p0 vector from detected peaks for use with curve_fit.

        Returns
        -------
        np.ndarray
            Interleaved as: amplitude_0, fwhm_0, center_0, ...
            Empty array if no peaks are found.
        """
        peak_positions, peak_properties = find_domain_peaks(
            domain=domain,
            smooth=smooth,
            smoothing_fwhm_scale=smoothing_fwhm_scale,
            prominence=prominence,
        )
        n = len(peak_positions)
        p0 = np.empty(3 * n)
        p0[0::3] = peak_properties["prominences"]
        p0[1::3] = peak_properties["fwhm"]
        p0[2::3] = peak_positions
        return p0

    @staticmethod
    def _create_bounds(domain: Domain, peaks_number: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Build flat lower/upper bound arrays for curve_fit.

        Parameters
        ----------
        domain : Domain
        peaks_number : int

        Returns
        -------
        lower, upper : np.ndarray
            Each of shape (3 * peaks_number,), interleaved as amplitude, fwhm, center.
        """
        lower = np.empty(3 * peaks_number)
        upper = np.empty(3 * peaks_number)

        max_val = domain.max().item()
        axis_start = domain.spectrum.axis[domain.start].item()
        axis_stop = domain.spectrum.axis[domain.stop - 1].item()
        axis_extent = abs(axis_stop - axis_start)

        lower[0::3] = 0;          upper[0::3] = 2 * max_val
        lower[1::3] = 0;          upper[1::3] = axis_extent
        lower[2::3] = axis_start; upper[2::3] = axis_stop

        return lower, upper

    @classmethod
    def fit(
        cls,
        domain: Domain,
        *,
        smooth: bool = True,
        smoothing_fwhm_scale: float = 0.3,
        prominence: float | None = None,
        maxiter: int = 100,
    ) -> xr.Dataset | bool:
        """
        Fit a sum of Gaussians to a spectral domain.

        Parameters
        ----------
        domain : Domain
        smooth : bool
            Smooth before peak detection.
        smoothing_fwhm_scale : float
            Smoothing scale relative to detector resolution. Keep below 0.5.
        prominence : float, optional
            Minimum peak prominence for detection.
        maxiter : int
            Max optimizer iterations per parameter.

        Returns
        -------
        xr.Dataset
            Variables ``amplitude``, ``fwhm``, ``center`` (coordinate ``i``)
            and ``covariance`` (shape 3K × 3K). Returns ``False`` if fit fails.
        """
        p0 = cls._initial_guess(domain, smooth=smooth,
                                 smoothing_fwhm_scale=smoothing_fwhm_scale,
                                 prominence=prominence)

        if len(p0) == 0:
            warnings.warn("No peaks detected in domain; skipping fit.")
            return False

        n_peaks = len(p0) // 3
        lower, upper = cls._create_bounds(domain, n_peaks)

        domain_axis = domain.spectrum.axis[domain.start:domain.stop]
        domain_values = domain.values

        try:
            popt, pcov = curve_fit(f=cls._flat_evaluate,
                                   xdata=domain_axis,
                                   ydata=domain_values,
                                   p0=p0,
                                   bounds=(lower, upper),
                                   max_nfev=maxiter * len(p0),
                                   method="trf",)
        except (ValueError, RuntimeError) as e:
            warnings.warn(f"Fit of domain [{domain_axis[0]}, {domain_axis[-1]}] failed: {e}")
            return False

        return xr.Dataset(
            data_vars={
                "amplitude":  ("i", popt[0::3]),
                "fwhm":       ("i", popt[1::3]),
                "center":     ("i", popt[2::3]),
                "covariance": (["param", "param_"], pcov),
            },
            coords={"i": np.arange(n_peaks)},
        )