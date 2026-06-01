import numpy as np
import xarray as xr
from scipy.optimize import curve_fit
from scipy.special import erf as _scipy_erf
import warnings

from pyspectrum.core.domain import Domain
from pyspectrum.domain_analysis.morphology import find_domain_peaks, domain_bases
from pyspectrum.utils.gaussian import fwhm_to_sigma
from pyspectrum.domain_fitting.abstract_fitting_class import MultiPeakFitter


class MultiGaussianFitter(MultiPeakFitter):
    """
    Fit multiple Gaussian peaks (with optional erf step background) to a Domain.

    Parameters
    ----------
    background : {'none', 'erf'}
        ``'none'`` — plain sum of Gaussians. Use on background-subtracted domains.
        ``'erf'`` — adds a prominence-weighted erf step background that models the
        Compton continuum. Use on raw domains; requires the parent Spectrum to have
        a resolution calibration set.

    Usage
    -----
    fitter = MultiGaussianFitter()
    result = fitter.fit(domain)

    fitter_erf = MultiGaussianFitter(background='erf')
    result     = fitter_erf.fit(domain)
    curves     = fitter_erf.sample_curves(axis, result, size=1000)

    Fit result Dataset
    ------------------
    ``params`` : DataArray, shape ``(n_peaks, 3)``, coords ``i`` × ``quantity``
        Fitted peak parameters. ``quantity`` is ``["amplitude", "fwhm", "center"]``.
        Access individual parameters with ``result["params"].sel(quantity="center")``.

    ``background`` : DataArray, shape ``(2,)``, coord ``bg_quantity``  *(erf mode only)*
        ``["height_diff", "peak_baseline"]`` — the two erf background parameters.

    ``covariance`` : DataArray, shape ``(n_params, n_params)``, coords ``param`` × ``param_``
        Full parameter covariance matrix from the fit, with named coordinates.
        Use with ``numpy.random.multivariate_normal`` via ``sample()``.
    """

    def __init__(self, background: str = "none"):
        if background not in ("none", "erf"):
            raise ValueError(f"background must be 'none' or 'erf', got {background!r}")
        self.background = background

    # ------------------------------------------------------------------
    # Model functions  (passed directly to curve_fit)
    # ------------------------------------------------------------------

    @staticmethod
    def _gaussians(axis: np.ndarray, *params) -> np.ndarray:
        """Sum of Gaussians. Params: amplitude_0, fwhm_0, center_0, ..."""
        params = np.asarray(params)
        a = params[0::3][:, np.newaxis]
        s = fwhm_to_sigma(params[1::3])[:, np.newaxis]
        m = params[2::3][:, np.newaxis]
        return (a * np.exp(-((axis - m) ** 2) / (2 * s ** 2))).sum(axis=0)

    @staticmethod
    def _gaussians_erf(axis: np.ndarray, *params) -> np.ndarray:
        """Sum of Gaussians + prominence-weighted erf step background.
        Params: height_diff, peak_baseline, amplitude_0, fwhm_0, center_0, ..."""
        params        = np.asarray(params)
        height_diff   = params[0]
        peak_baseline = params[1]
        peak_params   = params[2:]

        a = peak_params[0::3][:, np.newaxis]
        s = fwhm_to_sigma(peak_params[1::3])[:, np.newaxis]
        m = peak_params[2::3][:, np.newaxis]

        weights        = a / a.sum()
        normalized_erf = (weights * (_scipy_erf(-(axis - m) / s) + 1) / 2).sum(axis=0)

        return (
            (a * np.exp(-((axis - m) ** 2) / (2 * s ** 2))).sum(axis=0)
            + height_diff * normalized_erf
            + peak_baseline
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _initial_guess(domain, background, *, smooth, smoothing_fwhm_scale, prominence):
        # prominence must be non-None so scipy computes and returns the prominences key.
        # Default of 10 % of the domain maximum suppresses noise peaks in typical
        # Poisson-noisy spectra while keeping all real Gaussian peaks.
        _prom = prominence if prominence is not None else domain.max().item() * 0.1

        if background == "erf":
            height_left, height_right = domain_bases(domain)
            peak_positions, props = find_domain_peaks(
                domain=domain, smooth=smooth,
                smoothing_fwhm_scale=smoothing_fwhm_scale, prominence=_prom,
            )
            n = len(peak_positions)
            if n == 0:
                return np.empty(2)
            p0 = np.empty(2 + 3 * n)
            p0[0]        = height_left - height_right
            p0[1]        = min(height_left, height_right)
            p0[2 + 0::3] = props["prominences"]
            p0[2 + 1::3] = props["fwhm"]
            p0[2 + 2::3] = peak_positions
        else:
            peak_positions, props = find_domain_peaks(
                domain=domain, smooth=smooth,
                smoothing_fwhm_scale=smoothing_fwhm_scale, prominence=_prom,
            )
            n = len(peak_positions)
            if n == 0:
                return np.empty(0)
            p0 = np.empty(3 * n)
            p0[0::3] = props["prominences"]
            p0[1::3] = props["fwhm"]
            p0[2::3] = peak_positions

        return p0

    @staticmethod
    def _build_bounds(domain, n_peaks, background):
        axis       = domain.axis
        max_val    = domain.max().item()
        axis_start = axis[0]
        axis_stop  = axis[-1]
        extent     = abs(axis_stop - axis_start)

        if background == "erf":
            lower = np.empty(2 + 3 * n_peaks)
            upper = np.empty(2 + 3 * n_peaks)
            min_val = domain.min().item()
            lower[0] = -2 * max_val;       upper[0] = 2 * max_val
            lower[1] = -2 * abs(min_val);  upper[1] = 2 * max_val
            lower[2 + 0::3] = 0;           upper[2 + 0::3] = 2 * max_val
            lower[2 + 1::3] = 0;           upper[2 + 1::3] = extent
            lower[2 + 2::3] = axis_start;  upper[2 + 2::3] = axis_stop
        else:
            lower = np.empty(3 * n_peaks)
            upper = np.empty(3 * n_peaks)
            lower[0::3] = 0;           upper[0::3] = 2 * max_val
            lower[1::3] = 0;           upper[1::3] = extent
            lower[2::3] = axis_start;  upper[2::3] = axis_stop

        return lower, upper

    @staticmethod
    def _build_dataset(popt, pcov, n_peaks, background):
        peak_param_names = [
            f"{name}_{i}"
            for i in range(n_peaks)
            for name in ("amplitude", "fwhm", "center")
        ]

        if background == "erf":
            param_names = ["height_diff", "peak_baseline"] + peak_param_names
            return xr.Dataset(
                data_vars={
                    "params":     (["i", "quantity"], popt[2:].reshape(n_peaks, 3)),
                    "background": (["bg_quantity"],   popt[:2]),
                    "covariance": (["param", "param_"], pcov),
                },
                coords={
                    "i":           np.arange(n_peaks),
                    "quantity":    ["amplitude", "fwhm", "center"],
                    "bg_quantity": ["height_diff", "peak_baseline"],
                    "param":       param_names,
                    "param_":      param_names,
                },
                attrs={"background": "erf"},
            )

        return xr.Dataset(
            data_vars={
                "params":     (["i", "quantity"], popt.reshape(n_peaks, 3)),
                "covariance": (["param", "param_"], pcov),
            },
            coords={
                "i":       np.arange(n_peaks),
                "quantity": ["amplitude", "fwhm", "center"],
                "param":   peak_param_names,
                "param_":  peak_param_names,
            },
            attrs={"background": "none"},
        )

    def _to_flat(self, dataset: xr.Dataset):
        """Return (mean, cov) compatible with numpy.random.multivariate_normal."""
        params_flat = dataset["params"].values.ravel()
        cov = dataset["covariance"].values
        if self.background == "erf":
            mean = np.concatenate([dataset["background"].values, params_flat])
        else:
            mean = params_flat
        return mean, cov

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(
        self,
        domain: Domain,
        *,
        smooth: bool = True,
        smoothing_fwhm_scale: float = 0.3,
        prominence: float | None = None,
        maxiter: int = 100,
    ) -> "xr.Dataset | bool":
        """Fit the model to a spectral domain.

        Parameters
        ----------
        domain : Domain
        smooth : bool
            Apply Gaussian smoothing before peak detection. Requires the parent
            Spectrum to have a resolution calibration set.
        smoothing_fwhm_scale : float
            Smoothing scale relative to detector FWHM. Keep below 0.5.
        prominence : float, optional
            Minimum peak prominence for detection.
            Defaults to 10 % of the domain maximum, which suppresses noise
            peaks in typical Poisson-noisy spectra while retaining real
            Gaussian peaks. Pass an explicit value to override.
        maxiter : int
            Max optimizer iterations per parameter.

        Returns
        -------
        xr.Dataset or False
            Dataset on success (see class docstring for structure).
            ``False`` if no peaks are detected or the fit fails to converge.
        """
        p0 = self._initial_guess(
            domain, self.background,
            smooth=smooth, smoothing_fwhm_scale=smoothing_fwhm_scale,
            prominence=prominence,
        )

        no_peaks = (len(p0) == 0) if self.background == "none" else (len(p0) == 2)
        if no_peaks:
            warnings.warn("No peaks detected in domain; skipping fit.")
            return False

        n_peaks  = len(p0) // 3 if self.background == "none" else (len(p0) - 2) // 3
        lower, upper = self._build_bounds(domain, n_peaks, self.background)
        model_fn = self._gaussians_erf if self.background == "erf" else self._gaussians

        domain_axis   = domain.axis
        domain_values = domain.values

        try:
            popt, pcov = curve_fit(
                f=model_fn,
                xdata=domain_axis,
                ydata=domain_values,
                p0=p0,
                bounds=(lower, upper),
                max_nfev=maxiter * len(p0),
                method="trf",
            )
        except (ValueError, RuntimeError) as e:
            warnings.warn(
                f"Fit of domain [{domain_axis[0]:.3g}, {domain_axis[-1]:.3g}] failed: {e}"
            )
            return False

        return self._build_dataset(popt, pcov, n_peaks, self.background)

    def evaluate(self, axis: np.ndarray, dataset: xr.Dataset) -> np.ndarray:
        """Evaluate the model on *axis* using parameters from *dataset*.

        *dataset* can be a fit result (from ``fit()``) or a single sample
        (from ``sample().isel(sample=k)``).

        Parameters
        ----------
        axis : np.ndarray
        dataset : xr.Dataset
            Must contain ``params`` (and ``background`` for erf mode).

        Returns
        -------
        np.ndarray, shape (len(axis),)
        """
        params = dataset["params"].values  # (n_peaks, 3)
        if self.background == "erf":
            flat = np.concatenate([dataset["background"].values, params.ravel()])
            return self._gaussians_erf(axis, *flat)
        return self._gaussians(axis, *params.ravel())

    def sample(
        self,
        fit_result: xr.Dataset,
        size: int = 1000,
        *,
        rng=None,
    ) -> xr.Dataset:
        """Draw parameter samples from the fitted multivariate normal distribution.

        The returned Dataset has the same structure as the fit result but with
        an extra ``sample`` dimension, so a single draw can be passed directly
        to ``evaluate``::

            samples = fitter.sample(result, size=500)
            curve   = fitter.evaluate(axis, samples.isel(sample=0))

        Parameters
        ----------
        fit_result : xr.Dataset
            Output of ``fit()``.
        size : int
            Number of samples.
        rng : int, numpy.random.Generator, or None
            Seed or generator for reproducibility.

        Returns
        -------
        xr.Dataset
            ``params`` : ``(sample, i, quantity)``
            ``background`` : ``(sample, bg_quantity)``  *(erf mode only)*
        """
        mean, cov = self._to_flat(fit_result)
        raw = np.random.default_rng(rng).multivariate_normal(mean, cov, size=size)
        n_peaks = len(fit_result.coords["i"])

        if self.background == "erf":
            return xr.Dataset(
                data_vars={
                    "params":     (["sample", "i", "quantity"], raw[:, 2:].reshape(size, n_peaks, 3)),
                    "background": (["sample", "bg_quantity"],   raw[:, :2]),
                },
                coords={
                    "sample":      np.arange(size),
                    "i":           fit_result.coords["i"],
                    "quantity":    fit_result.coords["quantity"],
                    "bg_quantity": fit_result.coords["bg_quantity"],
                },
                attrs={"background": "erf"},
            )

        return xr.Dataset(
            data_vars={
                "params": (["sample", "i", "quantity"], raw.reshape(size, n_peaks, 3)),
            },
            coords={
                "sample":   np.arange(size),
                "i":        fit_result.coords["i"],
                "quantity": fit_result.coords["quantity"],
            },
            attrs={"background": "none"},
        )

    def sample_curves(
        self,
        axis: np.ndarray,
        fit_result: xr.Dataset,
        size: int = 1000,
        *,
        rng=None,
    ) -> np.ndarray:
        """Sample the distribution of fitted curves.

        Draws *size* parameter vectors from the multivariate normal defined by
        the fit result and evaluates the model at each, giving a pointwise
        uncertainty band over the fitted curve.

        Parameters
        ----------
        axis : np.ndarray
        fit_result : xr.Dataset
        size : int
        rng : int, numpy.random.Generator, or None

        Returns
        -------
        np.ndarray, shape ``(size, len(axis))``
            ``curves.mean(axis=0)`` → mean curve
            ``curves.std(axis=0)``  → pointwise 1-sigma band
        """
        samples = self.sample(fit_result, size=size, rng=rng)
        return np.stack([
            self.evaluate(axis, samples.isel(sample=k)) for k in range(size)
        ])
