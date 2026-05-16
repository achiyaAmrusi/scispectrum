import numpy as np
import xarray as xr
from scipy.optimize import curve_fit
from scipy.special import erf
import warnings

from pyspectrum.core.domain import Domain
from pyspectrum.domain_analysis.morphology import find_domain_peaks, domain_bases
from pyspectrum.utils.gaussian import fwhm_to_sigma
from pyspectrum.domain_fitting.abstract_fitting_class import PeakFit


class SumOfGaussiansErf(PeakFit):

    def __init__(self):
        """No initialization needed for now."""

    @staticmethod
    def _flat_evaluate(axis: np.ndarray, *params) -> np.ndarray:
        """
        Sum of Gaussians plus prominence-weighted erf background, from a flat parameter vector.

        Parameter order:
            height_diff, peak_baseline, amplitude_0, fwhm_0, center_0, ...
        """
        params        = np.asarray(params)
        height_diff   = params[0]
        peak_baseline = params[1]
        peak_params   = params[2:]

        a = peak_params[0::3][:, np.newaxis]          # (K, 1)
        s = fwhm_to_sigma(peak_params[1::3])[:, np.newaxis]  # (K, 1)
        m = peak_params[2::3][:, np.newaxis]          # (K, 1)

        weights        = a / a.sum()
        normalized_erf = (weights * (erf(-(axis - m) / s) + 1) / 2).sum(axis=0)
        background     = height_diff * normalized_erf + peak_baseline

        return (a * np.exp(-((axis - m) ** 2) / (2 * s ** 2))).sum(axis=0) + background

    @classmethod
    def evaluate(cls, axis: np.ndarray, fit_result: xr.Dataset) -> np.ndarray:
        """
        Evaluate the fitted model from a fit result Dataset.

        Parameters
        ----------
        axis : np.ndarray, shape (N,)
        fit_result : xr.Dataset
            As returned by ``fit``.

        Returns
        -------
        np.ndarray, shape (N,)
        """
        n = len(fit_result["amplitude"].values)
        p = np.empty(2 + 3 * n)
        p[0]        = fit_result["height_diff"].item()
        p[1]        = fit_result["peak_baseline"].item()
        p[2 + 0::3] = fit_result["amplitude"].values
        p[2 + 1::3] = fit_result["fwhm"].values
        p[2 + 2::3] = fit_result["center"].values
        return cls._flat_evaluate(axis, *p)

    @staticmethod
    def _initial_guess(domain: Domain, *, smooth: bool = True,
                       smoothing_fwhm_scale: float = 0.3,
                       prominence: float | None = None) -> np.ndarray:
        """
        Flat p0 vector: [height_diff, peak_baseline, amplitude_0, fwhm_0, center_0, ...]
        Background initialised from domain edges; erf center from prominence-weighted peak positions.
        """
        peak_positions, peak_properties = find_domain_peaks(
            domain=domain, smooth=smooth,
            smoothing_fwhm_scale=smoothing_fwhm_scale,
            prominence=prominence,
        )
        height_left, height_right = domain_bases(domain)
        prominences = peak_properties["prominences"]

        n  = len(peak_positions)
        p0 = np.empty(2 + 3 * n)
        p0[0]        = height_left - height_right
        p0[1]        = min(height_left, height_right)
        p0[2 + 0::3] = prominences
        p0[2 + 1::3] = peak_properties["fwhm"]
        p0[2 + 2::3] = peak_positions
        return p0

    @staticmethod
    def _create_bounds(domain: Domain, peaks_number: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Flat lower/upper bounds: [height_diff, peak_baseline, amplitude_0, fwhm_0, center_0, ...]
        """
        lower = np.empty(2 + 3 * peaks_number)
        upper = np.empty(2 + 3 * peaks_number)

        max_val     = domain.max().item()
        axis_start  = domain.spectrum.axis[domain.start].item()
        axis_stop   = domain.spectrum.axis[domain.stop - 1].item()
        axis_extent = abs(axis_stop - axis_start)

        lower[0] = -2 * max_val;  upper[0] = 2 * max_val    # height_diff
        lower[1] = 2*np.minimum(domain.min().item(), -domain.min().item());             upper[1] = 2 * max_val    # peak_baseline
        lower[2 + 0::3] = 0;           upper[2 + 0::3] = 2 * max_val  # amplitude
        lower[2 + 1::3] = 0;           upper[2 + 1::3] = axis_extent  # fwhm
        lower[2 + 2::3] = axis_start;  upper[2 + 2::3] = axis_stop    # center

        return lower, upper

    @classmethod
    def fit(cls, domain: Domain, *, smooth: bool = True,
            smoothing_fwhm_scale: float = 0.3,
            prominence: float | None = None,
            maxiter: int = 100) -> xr.Dataset | bool:
        """
        Fit a sum of Gaussians with erf background to a spectral domain.

        Parameters
        ----------
        domain : Domain
        smooth : bool
        smoothing_fwhm_scale : float
            Keep below 0.5.
        prominence : float, optional
        maxiter : int

        Returns
        -------
        xr.Dataset
            Variables ``height_diff``, ``peak_baseline``, ``amplitude``, ``fwhm``,
            ``center`` (coordinate ``i``), and ``covariance`` (3K+2 x 3K+2).
            Returns ``False`` if fit fails.
        """
        p0 = cls._initial_guess(domain, smooth=smooth,
                                 smoothing_fwhm_scale=smoothing_fwhm_scale,
                                 prominence=prominence)
        if len(p0) == 2:
            warnings.warn("No peaks detected in domain; skipping fit.")
            return False

        n_peaks      = (len(p0) - 2) // 3
        lower, upper = cls._create_bounds(domain, n_peaks)
        domain_axis  = domain.spectrum.axis[domain.indices]
        domain_values = domain.values

        try:
            popt, pcov = curve_fit(
                cls._flat_evaluate,
                domain_axis,
                domain_values,
                p0=p0,
                bounds=(lower, upper),
                max_nfev=maxiter * len(p0),
                method="trf",
            )
        except (ValueError, RuntimeError) as e:
            warnings.warn(f"Fit of domain [{domain_axis[0]}, {domain_axis[-1]}] failed: {e}")
            return False

        return xr.Dataset(
            data_vars={
                "height_diff":   popt[0],
                "peak_baseline": popt[1],
                "amplitude":  ("i", popt[2 + 0::3]),
                "fwhm":       ("i", popt[2 + 1::3]),
                "center":     ("i", popt[2 + 2::3]),
                "covariance": (["param", "param_"], pcov),
            },
            coords={"i": np.arange(n_peaks)},
        )