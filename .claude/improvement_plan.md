# pySpectrum — Agent Improvement Plan

Read CLAUDE.md first for project context before starting any item here.
Items marked ✅ are done. Open items are ordered by dependency.

---

## Session summary (2026-06-01 → 2026-06-02)

Starting from 38 tests at ~83% coverage, this session completed ITEMs 2, 2b, 3, and 4:

- **Merged fitting classes** — `SumOfGaussians` + `SumOfGaussiansErf` replaced by
  `MultiGaussianFitter` with `background='none'|'erf'|'linear'`. `PeakFit` ABC
  renamed `MultiPeakFitter`. New sampling API: `fit()` → `evaluate()` → `sample()`
  → `sample_curves()` with consistent Dataset schema across all modes.
- **Filename typos fixed** — two misspelled filenames corrected via `git mv`.
- **Test coverage** — 38 → 157 tests; 83% → 91% coverage. New suites for background
  estimators, identification (convolution, kernels, SNRFinder), domain_analysis
  (find_peaks, morphology, moment, background functions), and io (TimeChannelParser).
- **Deleted** `asymmetrical_rect_zero_area` stub (see MAYBE for redesign note).
- **Linear background** — `MultiGaussianFitter(background='linear')` added with
  extensible `_BG_PARAM_NAMES` architecture.

Open: ITEM 1 (estimator-level uncertainty), MAYBE items (asymmetrical kernel,
serialization). CLAUDE.md still references stale class names — needs updating.

---

## ✅ COMPLETED

### Deleted Peak class (`core/peak.py`)
`Peak` was the legacy data container for a spectral slice. `Domain` is its replacement
and fully covers its role. All usages were migrated before deletion.

### Deleted GaussianWithBGFitting (`domain_fitting/std_gaussian_fitting.py`)
Was Peak's internal fitting backend (Gaussian + erf via xarray.curvefit). Removed
together with Peak. The canonical fitter is now `SumOfGaussians`.

### Deleted SkewGaussianFitting stub (`domain_fitting/skew_gauss.py`)
Empty 23-line stub that was never implemented. Removed from the public package.

### Migrated DetectorCalibration off Peak
`DetectorCalibration.estimate_peaks()` now uses `Domain` +
`domain_analysis.single_peak.center_estimator` / `fwhm_estimator` instead of
`Peak.center_fwhm_estimator`.

### Fixed `_get_data` bug in `domain_analysis/single_peak.py`
Was calling `spectrum.data_with_errors` as a boolean check, which raises `ValueError`
when `counts_err is None`. Fixed to check `spectrum.counts_err is not None`.

### Fixed `Spectrum._apply_operation` stale kwarg
Was passing removed `channels=` parameter to `Spectrum()` constructor. Removed.

### Added DetectorCalibration test suite (`tests/calibration/test_detector_calibration.py`)
9 tests covering `estimate_peaks()` (count, order, centre accuracy ≤1% or 2 ch, FWHM
accuracy ±15%) and `generate()` (types, energy accuracy ±1 keV, resolution accuracy
±20%, attachability). Uses a fully synthetic spectrum — no real data dependency.

### Domain-level background uncertainty propagation (ITEM 1, partial)
`Domain` now stores `background` and `background_err` as separate plain arrays,
mirroring the `Spectrum` pattern of `counts` / `counts_err`. `subtract_background()`
accepts an optional `background_err` argument. `data_with_errors` wraps the background
in `uarray(background, background_err)` when errors are present, combining them in
quadrature with spectrum Poisson errors. `Spectrum.domain()` also accepts `background_err`.

### Unified BackgroundEstimator interface
All five background estimators now share the honest common interface:
`estimate(self, axis: np.ndarray, counts: np.ndarray) -> np.ndarray`

- `BackgroundEstimator` ABC: corrected signature and return type (`np.ndarray`, not
  `xr.DataArray`); removed unused `xr` and `Spectrum` imports; docstring states that
  auxiliary inputs must go in `__init__`, not `estimate`.
- `ALSBackground`: removed `**kwargs` from `estimate`.
- `SNIPBackground`: `resolution` and `smooth` moved to `__init__`; `estimate` is now
  `(self, axis, counts)`; renamed public `sinp` → private `_sinp`; fixed silent bug
  where `smoothing` kwarg was accepted by `estimate` but never forwarded.
- `IterativePolyFit`: renamed params `x, y` → `axis, counts`.
- `IterativePolyFitWithMinimum`: `resolution` and `conv` moved to `__init__`; `estimate`
  is now `(self, axis, counts)`; fixed shadowed loop variable `i`.
- `MinimaEnvelopeBackground`: `resolution_calib`, `conv`, and `iterations` moved to
  `__init__`; `estimate` is now `(self, axis, counts)`.
- `ALSBackground` added to `background/__init__.py` (was missing from public exports).

---

## OPEN ITEMS

### ITEM 1 — Estimator-level background uncertainty (remaining)

**Priority: Medium — Domain propagation is done; this is the last piece**

Each estimator still returns a plain `np.ndarray`. The caller is responsible for
constructing a `background_err` and passing it to `subtract_background()`. What each
estimator would need to return an uncertainty estimate:
- **ALSBackground**: RMSE of final-iteration residuals as uniform std_dev
- **SNIPBackground**: run on `counts ± sqrt(counts)`, half-difference as std_dev
- **IterativePolyFit**: residual std of the final polynomial fit
- **MinimaEnvelopeBackground**: spread of local minima within each window

The BackgroundEstimator ABC question is now resolved: auxiliary params go in `__init__`,
and `estimate(axis, counts) -> np.ndarray` is the enforced interface. Uncertainty can
be added as a second return value (tuple) or a separate method without breaking the ABC.

---

### ✅ ITEM 2 — Consolidate the fitting classes

`SumOfGaussians` and `SumOfGaussiansErf` merged into `MultiGaussianFitter`
(`domain_fitting/multi_gaussian_fitter.py`). `background='none'|'erf'` selects
the model. The ABC `PeakFit` was renamed `MultiPeakFitter`. Dataset schema:
`params (i × quantity)`, `covariance (param × param_)`, `background (bg_quantity)`
for erf mode. Sampling API: `fit()` → `evaluate()` → `sample()` → `sample_curves()`.

---

### ✅ ITEM 2b — Add linear background option to MultiGaussianFitter

`background='linear'` added to `MultiGaussianFitter`. Parameterised as
`(bg_left, bg_right)` — the background level at each domain edge — rather than
slope/intercept, which is numerically unstable for large axis values (e.g. keV).

Alongside this, the class was refactored for extensibility: `_BG_PARAM_NAMES` is
now the single source of truth for all background modes. Every method reads
`n_bg = len(_BG_PARAM_NAMES[background])` and branches only on `n_bg > 0`, so
adding a future mode requires only a new dict entry, a model function, and entries
in `_bg_initial_values` / `_bg_bounds`.

Both `'erf'` and `'linear'` modes now subtract a linear trend before peak detection
so peaks stand out cleanly regardless of background shape.

---

### ✅ ITEM 3 — Fix filename typos

- `pyspectrum/background/minimum_in_envolope.py` → `minimum_in_envelope.py`
- `tests/identification/test_snr_idendification.py` → `test_snr_identification.py`
- Updated `background/__init__.py` import accordingly.

---

### ✅ ITEM 4 — Fill test coverage

144 tests total, all passing. Overall coverage: 83% → 91%.

New test files:
- `tests/background/test_background_estimators.py` — all 5 estimators: shape, below-peak, SNIP edge accuracy
- `tests/identification/test_convolution.py` — output shapes, Poisson variance, flat-spectrum near-zero response
- `tests/identification/test_kernels.py` — gaussian_2_dev (zero-area, symmetry) + Kernel1D ABC helpers
- `tests/identification/test_snr_identification.py` — detects both peaks, domain bounds, caching
- `tests/domain_analysis/test_domain_analysis.py` — two fixtures: single_domain (center=500) and
  two_peak_domain (peaks at 300 & 370, ~3 FWHM apart); covers find_domain_peaks, morphology,
  single_peak estimators, moment functions, and domain background functions
- `tests/io/test_time_channel.py` — filter(), from_dataframe() with flag/negative filtering

Deleted: `identification/kernels/asymmetrical.py` (unfinished stub, see MAYBE section).

Known limitation: `domain_peaks_fwhm` uses peak_widths on unsmoothed data with smoothed
peak indices — can return 0 or unreliable values. Use `find_domain_peaks(domain)["fwhm"]`
for accurate FWHM estimates.

---

## MAYBE (not prioritised)

### Asymmetrical peak detection kernel
The deleted `asymmetrical_rect_zero_area` was a placeholder for detecting
asymmetric peaks (e.g. Doppler-broadened or pile-up distorted lines).
Needs a proper design: likely a skewed Gaussian or an asymmetric zero-area
kernel that inherits from `Kernel1D`, with a `Convolution`-compatible
interface and tests against a synthetic asymmetric peak.

---

### Spectrum serialization
Add `Spectrum.save(path)` / `Spectrum.load(path)` using xarray's NetCDF backend.
Calibration functions cannot be serialised directly — would need to store parameters
in `metadata` and reconstruct on load. Deferred until there is a clear use case.

---

## Dependency order

```
ITEM 1 (estimator-level uncertainty) — independent; ABC and Domain side already done

ITEM 2b (linear background) — independent; builds on the MultiGaussianFitter pattern
```
