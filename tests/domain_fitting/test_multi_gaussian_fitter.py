"""
Tests for MultiGaussianFitter.

Uses fully synthetic spectra — no real data dependency.

Ground truth
------------
Single Gaussian peak:
    center   = TRUE_CENTER = 50.0 axis units
    fwhm     = TRUE_FWHM   = 5.0  axis units
    amplitude= TRUE_AMP    = 3000 counts (high SNR, tight Poisson noise)

For erf-background mode, data also includes a Compton-step background
generated from the same erf formula the model uses, so the fit has a
clean convergence path.
"""

import numpy as np
import pytest
import xarray as xr
from scipy.special import erf as scipy_erf

from pyspectrum.core import Spectrum
from pyspectrum.calibration.axis import AxisCalibration
from pyspectrum.calibration.resolution import ResolutionCalibration
from pyspectrum.domain_fitting import MultiGaussianFitter

# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)

N           = 512
AXIS        = np.linspace(0.0, 100.0, N)
TRUE_CENTER = 50.0
TRUE_FWHM   = 5.0
TRUE_AMP    = 3000.0
TRUE_HEIGHT_DIFF = 80.0
TRUE_BASELINE    = 15.0

_SIGMA = TRUE_FWHM / (2 * np.sqrt(2 * np.log(2)))

DOMAIN_START = TRUE_CENTER - 3 * TRUE_FWHM   # 35.0
DOMAIN_STOP  = TRUE_CENTER + 3 * TRUE_FWHM   # 65.0

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gaussian(axis):
    return TRUE_AMP * np.exp(-((axis - TRUE_CENTER) ** 2) / (2 * _SIGMA ** 2))

def _erf_bg(axis):
    # Matches the single-peak erf model (weights=1)
    return TRUE_HEIGHT_DIFF * (scipy_erf(-(axis - TRUE_CENTER) / _SIGMA) + 1) / 2 + TRUE_BASELINE

def _res_calib():
    return ResolutionCalibration(
        lambda x: np.full_like(np.asarray(x, dtype=float), TRUE_FWHM)
    )

def _axis_calib():
    return AxisCalibration.from_array(AXIS, name="energy")

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def spectrum_plain():
    counts = RNG.poisson(np.maximum(_gaussian(AXIS), 0)).astype(float)
    s = Spectrum(counts=counts, axis_calib=_axis_calib())
    s.set_resolution_calibration(_res_calib())
    return s

@pytest.fixture(scope="module")
def spectrum_erf():
    counts = RNG.poisson(np.maximum(_gaussian(AXIS) + _erf_bg(AXIS), 0)).astype(float)
    s = Spectrum(counts=counts, axis_calib=_axis_calib())
    s.set_resolution_calibration(_res_calib())
    return s

@pytest.fixture(scope="module")
def domain_plain(spectrum_plain):
    return spectrum_plain.domain(DOMAIN_START, DOMAIN_STOP)

@pytest.fixture(scope="module")
def domain_erf(spectrum_erf):
    return spectrum_erf.domain(DOMAIN_START, DOMAIN_STOP)

@pytest.fixture(scope="module")
def fitter():
    return MultiGaussianFitter()

@pytest.fixture(scope="module")
def fitter_erf():
    return MultiGaussianFitter(background="erf")

@pytest.fixture(scope="module")
def result(fitter, domain_plain):
    r = fitter.fit(domain_plain)
    assert r is not False, "Fit should succeed on synthetic single-peak spectrum"
    return r

@pytest.fixture(scope="module")
def result_erf(fitter_erf, domain_erf):
    r = fitter_erf.fit(domain_erf)
    assert r is not False, "Fit should succeed on synthetic erf-background spectrum"
    return r

# ---------------------------------------------------------------------------
# Dataset structure
# ---------------------------------------------------------------------------

class TestFitResultStructure:

    def test_returns_dataset(self, result):
        assert isinstance(result, xr.Dataset)

    def test_has_params_with_correct_dims(self, result):
        assert "params" in result
        assert result["params"].dims == ("i", "quantity")

    def test_quantity_coordinate(self, result):
        assert list(result["quantity"].values) == ["amplitude", "fwhm", "center"]

    def test_has_covariance(self, result):
        n_params = 3 * len(result["i"])
        assert result["covariance"].shape == (n_params, n_params)

    def test_covariance_has_named_param_coords(self, result):
        coords = list(result.coords["param"].values)
        assert "amplitude_0" in coords
        assert "fwhm_0" in coords
        assert "center_0" in coords

    def test_background_attr_none(self, result):
        assert result.attrs.get("background") == "none"

    def test_erf_has_background_variable(self, result_erf):
        assert "background" in result_erf
        assert list(result_erf["bg_quantity"].values) == ["height_diff", "peak_baseline"]

    def test_erf_covariance_size(self, result_erf):
        n_peaks  = len(result_erf["i"])
        n_params = 2 + 3 * n_peaks
        assert result_erf["covariance"].shape == (n_params, n_params)

    def test_erf_background_attr(self, result_erf):
        assert result_erf.attrs.get("background") == "erf"

    def test_erf_param_coord_includes_bg(self, result_erf):
        coords = list(result_erf.coords["param"].values)
        assert coords[0] == "height_diff"
        assert coords[1] == "peak_baseline"

# ---------------------------------------------------------------------------
# Fit accuracy
# ---------------------------------------------------------------------------

class TestFitAccuracy:

    def test_single_peak_detected(self, result):
        assert len(result["i"]) == 1

    def test_center_accuracy(self, result):
        center = result["params"].sel(quantity="center").values[0]
        assert abs(center - TRUE_CENTER) < 0.5 * TRUE_FWHM

    def test_fwhm_accuracy(self, result):
        fwhm = result["params"].sel(quantity="fwhm").values[0]
        assert abs(fwhm - TRUE_FWHM) / TRUE_FWHM < 0.2

    def test_amplitude_accuracy(self, result):
        amp = result["params"].sel(quantity="amplitude").values[0]
        assert abs(amp - TRUE_AMP) / TRUE_AMP < 0.2

    def test_erf_center_accuracy(self, result_erf):
        center = result_erf["params"].sel(quantity="center").values[0]
        assert abs(center - TRUE_CENTER) < 0.5 * TRUE_FWHM

    def test_erf_fwhm_accuracy(self, result_erf):
        fwhm = result_erf["params"].sel(quantity="fwhm").values[0]
        assert abs(fwhm - TRUE_FWHM) / TRUE_FWHM < 0.2

# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------

class TestEvaluate:

    def test_output_shape(self, fitter, result):
        axis = np.linspace(DOMAIN_START, DOMAIN_STOP, 80)
        assert fitter.evaluate(axis, result).shape == (80,)

    def test_peak_value_near_amplitude(self, fitter, result):
        axis = np.array([TRUE_CENTER])
        amp  = result["params"].sel(quantity="amplitude").values[0]
        val  = fitter.evaluate(axis, result)[0]
        assert abs(val - amp) / amp < 0.01

    def test_erf_output_shape(self, fitter_erf, result_erf):
        axis = np.linspace(DOMAIN_START, DOMAIN_STOP, 80)
        assert fitter_erf.evaluate(axis, result_erf).shape == (80,)

# ---------------------------------------------------------------------------
# sample  — structure and consistency with evaluate
# ---------------------------------------------------------------------------

class TestSample:

    def test_returns_dataset(self, fitter, result):
        samples = fitter.sample(result, size=10, rng=0)
        assert isinstance(samples, xr.Dataset)

    def test_params_dims(self, fitter, result):
        samples = fitter.sample(result, size=10, rng=0)
        assert samples["params"].dims == ("sample", "i", "quantity")

    def test_sample_size(self, fitter, result):
        samples = fitter.sample(result, size=50, rng=0)
        assert samples.sizes["sample"] == 50

    def test_coords_consistent_with_fit_result(self, fitter, result):
        samples = fitter.sample(result, size=10, rng=0)
        assert list(samples["quantity"].values) == list(result["quantity"].values)
        assert list(samples["i"].values)        == list(result["i"].values)

    def test_erf_has_background_dim(self, fitter_erf, result_erf):
        samples = fitter_erf.sample(result_erf, size=10, rng=0)
        assert "background" in samples
        assert samples["background"].dims == ("sample", "bg_quantity")

    def test_single_sample_compatible_with_evaluate(self, fitter, result):
        samples = fitter.sample(result, size=5, rng=0)
        axis    = np.linspace(DOMAIN_START, DOMAIN_STOP, 40)
        curve   = fitter.evaluate(axis, samples.isel(sample=0))
        assert curve.shape == (40,)

    def test_erf_single_sample_compatible_with_evaluate(self, fitter_erf, result_erf):
        samples = fitter_erf.sample(result_erf, size=5, rng=0)
        axis    = np.linspace(DOMAIN_START, DOMAIN_STOP, 40)
        curve   = fitter_erf.evaluate(axis, samples.isel(sample=0))
        assert curve.shape == (40,)

    def test_rng_reproducibility(self, fitter, result):
        s1 = fitter.sample(result, size=20, rng=7)
        s2 = fitter.sample(result, size=20, rng=7)
        np.testing.assert_array_equal(s1["params"].values, s2["params"].values)

# ---------------------------------------------------------------------------
# sample_curves
# ---------------------------------------------------------------------------

class TestSampleCurves:

    def test_output_shape(self, fitter, result):
        axis   = np.linspace(DOMAIN_START, DOMAIN_STOP, 60)
        curves = fitter.sample_curves(axis, result, size=100, rng=0)
        assert curves.shape == (100, 60)

    def test_mean_close_to_best_fit(self, fitter, result):
        axis = np.linspace(DOMAIN_START, DOMAIN_STOP, 60)
        best = fitter.evaluate(axis, result)
        curves = fitter.sample_curves(axis, result, size=2000, rng=0)
        np.testing.assert_allclose(curves.mean(axis=0), best, atol=0.05 * TRUE_AMP)

    def test_erf_output_shape(self, fitter_erf, result_erf):
        axis   = np.linspace(DOMAIN_START, DOMAIN_STOP, 60)
        curves = fitter_erf.sample_curves(axis, result_erf, size=100, rng=0)
        assert curves.shape == (100, 60)

# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_invalid_background_raises(self):
        with pytest.raises(ValueError, match="background must be"):
            MultiGaussianFitter(background="linear")

    def test_returns_false_for_empty_domain(self, fitter, spectrum_plain):
        # Far from the peak — counts are essentially 0, no peaks detected
        domain = spectrum_plain.domain(2.0, 15.0)
        assert fitter.fit(domain) is False
