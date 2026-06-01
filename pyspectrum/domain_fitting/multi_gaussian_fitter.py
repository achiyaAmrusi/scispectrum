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
    Fit multiple Gaussian peaks (with optional background) to a Domain.

    Parameters
    ----------
    background : {'none', 'erf', 'linear'}
        ``'none'``   — plain sum of Gaussians; use on background-subtracted domains.
        ``'erf'``    — adds a prominence-weighted erf step background (Compton
                       continuum model); use on raw domains.
        ``'linear'`` — adds a linear background interpolated between the two domain
                       edges; parameterised as ``(bg_left, bg_right)`` — the
                       background level at the left and right edge of the domain.
                       More numerically stable than slope/intercept for large axis
                       values (e.g. energy in keV).

        Both ``'erf'`` and ``'linear'`` require the parent Spectrum to have a
        resolution calibration set (used by ``domain_bases`` to estimate edge heights).

    Usage
    -----
    fitter = MultiGaussianFitter()
    result = fitter.fit(domain)

    fitter_lin = MultiGaussianFitter(background='linear')
    result     = fitter_lin.fit(domain)
    curves     = fitter_lin.sample_curves(axis, result, size=1000)

    Fit result Dataset
    ------------------
    ``params`` : DataArray, shape ``(n_peaks, 3)``, coords ``i`` × ``quantity``
        Fitted peak parameters. ``quantity`` is ``["amplitude", "fwhm", "center"]``.

    ``background`` : DataArray, shape ``(2,)``, coord ``bg_quantity``
        Background parameters (absent for ``background='none'``).
        - erf mode:    ``["height_diff", "peak_baseline"]``
        - linear mode: ``["bg_left", "bg_right"]``

    ``covariance`` : DataArray, shape ``(n_params, n_params)``, coords ``param`` × ``param_``
        Full parameter covariance matrix from the fit.
        Use with ``sample()`` or ``numpy.random.multivariate_normal`` directly.
    """

    # Maps background mode → names of the prepended background parameters.
    # This dict is the single source of truth that drives all background-mode
    # branching throughout the class; add a new mode here and supply a model
    # function + _bg_initial_values/_bg_bounds entries to extend.
    _BG_PARAM_NAMES: dict[str, list[str]] = {
        "none":   [],
        "erf":    ["height_diff", "peak_baseline"],
        "linear": ["bg_left", "bg_right"],
    }

    def __init__(self, background: str = "none"):
        if background not in self._BG_PARAM_NAMES:
            raise ValueError(
                f"background must be one of {list(self._BG_PARAM_NAMES)}, "
                f"got {background!r}"
            )
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

    @staticmethod
    def _gaussians_linear(axis: np.ndarray, *params) -> np.ndarray:
        """Sum of Gaussians + linear background parameterised by edge values.
        Params: bg_left, bg_right, amplitude_0, fwhm_0, center_0, ..."""
        params      = np.asarray(params)
        bg_left     = params[0]
        bg_right    = params[1]
        peak_params = params[2:]

        a = peak_params[0::3][:, np.newaxis]
        s = fwhm_to_sigma(peak_params[1::3])[:, np.newaxis]
        m = peak_params[2::3][:, np.newaxis]

        t      = (axis - axis[0]) / (axis[-1] - axis[0])
        linear = bg_left + (bg_right - bg_left) * t

        return (
            (a * np.exp(-((axis - m) ** 2) / (2 * s ** 2))).sum(axis=0)
            + linear
        )

    def _model_fn(self):
        return {
            "none":   self._gaussians,
            "erf":    self._gaussians_erf,
            "linear": self._gaussians_linear,
        }[self.background]

    # ------------------------------------------------------------------
    # Background-specific helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _bg_initial_values(background: str, domain, height_left: float, height_right: float) -> np.ndarray:
        """Return initial parameter values for the prepended background params."""
        if background == "erf":
            return np.array([height_left - height_right, min(height_left, height_right)])
        if background == "linear":
            return np.array([height_left, height_right])
        return np.empty(0)

    @staticmethod
    def _bg_bounds(background: str, domain, lower: np.ndarray, upper: np.ndarray):
        """Fill the first n_bg elements of lower/upper with background param bounds."""
        max_val = domain.max().item()
        min_val = domain.min().item()
        if background == "erf":
            lower[0] = -2 * max_val;      upper[0] = 2 * max_val
            lower[1] = -2 * abs(min_val); upper[1] = 2 * max_val
        elif background == "linear":
            lower[0] = -2 * abs(min_val); upper[0] = 2 * max_val
            lower[1] = -2 * abs(min_val); upper[1] = 2 * max_val

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _initial_guess(self, domain, *, smooth, smoothing_fwhm_scale, prominence):
        _prom  = prominence if prominence is not None else domain.max().item() * 0.1
        n_bg   = len(self._BG_PARAM_NAMES[self.background])

        if n_bg > 0:
            height_left, height_right = domain_bases(domain)
            # Subtract linear trend before detection so peaks stand out clearly
            # regardless of whether the background is erf or linear.
            linear_bg  = np.linspace(height_left, height_right, domain.stop - domain.start)
            flat_domain = domain.subtract_background(linear_bg)
            peak_positions, props = find_domain_peaks(
                domain=flat_domain, smooth=smooth,
                smoothing_fwhm_scale=smoothing_fwhm_scale, prominence=_prom,
            )
            n = len(peak_positions)
            if n == 0:
                return np.empty(n_bg)
            p0          = np.empty(n_bg + 3 * n)
            p0[:n_bg]   = self._bg_initial_values(self.background, domain, height_left, height_right)
            p0[n_bg + 0::3] = props["prominences"]
            p0[n_bg + 1::3] = props["fwhm"]
            p0[n_bg + 2::3] = peak_positions
        else:
            peak_positions, props = find_domain_peaks(
                domain=domain, smooth=smooth,
                smoothing_fwhm_scale=smoothing_fwhm_scale, prominence=_prom,
            )
            n = len(peak_positions)
            if n == 0:
                return np.empty(0)
            p0        = np.empty(3 * n)
            p0[0::3]  = props["prominences"]
            p0[1::3]  = props["fwhm"]
            p0[2::3]  = peak_positions

        return p0

    def _build_bounds(self, domain, n_peaks):
        n_bg       = len(self._BG_PARAM_NAMES[self.background])
        axis       = domain.axis
        max_val    = domain.max().item()
        axis_start = axis[0]
        axis_stop  = axis[-1]
        extent     = abs(axis_stop - axis_start)

        lower = np.empty(n_bg + 3 * n_peaks)
        upper = np.empty(n_bg + 3 * n_peaks)

        if n_bg > 0:
            self._bg_bounds(self.background, domain, lower, upper)

        lower[n_bg + 0::3] = 0;           upper[n_bg + 0::3] = 2 * max_val
        lower[n_bg + 1::3] = 0;           upper[n_bg + 1::3] = extent
        lower[n_bg + 2::3] = axis_start;  upper[n_bg + 2::3] = axis_stop

        return lower, upper

    def _build_dataset(self, popt, pcov, n_peaks):
        n_bg          = len(self._BG_PARAM_NAMES[self.background])
        bg_names      = self._BG_PARAM_NAMES[self.background]
        peak_p_names  = [
            f"{name}_{i}"
            for i in range(n_peaks)
            for name in ("amplitude", "fwhm", "center")
        ]
        param_names = bg_names + peak_p_names

        data_vars = {
            "params":     (["i", "quantity"], popt[n_bg:].reshape(n_peaks, 3)),
            "covariance": (["param", "param_"], pcov),
        }
        coords = {
            "i":       np.arange(n_peaks),
            "quantity": ["amplitude", "fwhm", "center"],
            "param":   param_names,
            "param_":  param_names,
        }
        if n_bg > 0:
            data_vars["background"] = (["bg_quantity"], popt[:n_bg])
            coords["bg_quantity"]   = bg_names

        return xr.Dataset(data_vars=data_vars, coords=coords,
                          attrs={"background": self.background})

    def _to_flat(self, dataset: xr.Dataset):
        """Return (mean, cov) compatible with numpy.random.multivariate_normal."""
        params_flat = dataset["params"].values.ravel()
        cov         = dataset["covariance"].values
        n_bg        = len(self._BG_PARAM_NAMES[self.background])
        if n_bg > 0:
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
        p0   = self._initial_guess(domain, smooth=smooth,
                                    smoothing_fwhm_scale=smoothing_fwhm_scale,
                                    prominence=prominence)
        n_bg = len(self._BG_PARAM_NAMES[self.background])

        if len(p0) == n_bg:
            warnings.warn("No peaks detected in domain; skipping fit.")
            return False

        n_peaks      = (len(p0) - n_bg) // 3
        lower, upper = self._build_bounds(domain, n_peaks)
        domain_axis  = domain.axis
        domain_values = domain.values

        try:
            popt, pcov = curve_fit(
                f=self._model_fn(),
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

        return self._build_dataset(popt, pcov, n_peaks)

    def evaluate(self, axis: np.ndarray, dataset: xr.Dataset) -> np.ndarray:
        """Evaluate the model on *axis* using parameters from *dataset*.

        *dataset* can be a fit result (from ``fit()``) or a single sample
        (from ``sample().isel(sample=k)``).

        Returns
        -------
        np.ndarray, shape (len(axis),)
        """
        params = dataset["params"].values
        n_bg   = len(self._BG_PARAM_NAMES[self.background])
        if n_bg > 0:
            flat = np.concatenate([dataset["background"].values, params.ravel()])
        else:
            flat = params.ravel()
        return self._model_fn()(axis, *flat)

    def sample(
        self,
        fit_result: xr.Dataset,
        size: int = 1000,
        *,
        rng=None,
    ) -> xr.Dataset:
        """Draw parameter samples from the fitted multivariate normal distribution.

        The returned Dataset mirrors the fit result structure with an extra
        ``sample`` dimension — so ``samples.isel(sample=k)`` is directly
        compatible with ``evaluate``::

            samples = fitter.sample(result, size=500)
            curve   = fitter.evaluate(axis, samples.isel(sample=0))

        Returns
        -------
        xr.Dataset
            ``params``     : ``(sample, i, quantity)``
            ``background`` : ``(sample, bg_quantity)``  *(if background != 'none')*
        """
        mean, cov = self._to_flat(fit_result)
        raw       = np.random.default_rng(rng).multivariate_normal(mean, cov, size=size)
        n_peaks   = len(fit_result.coords["i"])
        n_bg      = len(self._BG_PARAM_NAMES[self.background])

        data_vars = {
            "params": (["sample", "i", "quantity"],
                       raw[:, n_bg:].reshape(size, n_peaks, 3)),
        }
        coords = {
            "sample":   np.arange(size),
            "i":        fit_result.coords["i"],
            "quantity": fit_result.coords["quantity"],
        }
        if n_bg > 0:
            data_vars["background"] = (["sample", "bg_quantity"], raw[:, :n_bg])
            coords["bg_quantity"]   = fit_result.coords["bg_quantity"]

        return xr.Dataset(data_vars=data_vars, coords=coords,
                          attrs={"background": self.background})

    def sample_curves(
        self,
        axis: np.ndarray,
        fit_result: xr.Dataset,
        size: int = 1000,
        *,
        rng=None,
    ) -> np.ndarray:
        """Sample the distribution of fitted curves.

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
