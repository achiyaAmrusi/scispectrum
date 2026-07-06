"""
Tests for DetectorCalibration.

Uses a fully synthetic spectrum — Gaussian peaks at known channels on a flat
background — so there is no dependency on real data files and the ground truth
is exact.

Ground truth
------------
- Energy calibration: linear,  energy = ENERGY_SLOPE * channel + ENERGY_OFFSET  (keV)
- FWHM model:         HPGe-like, fwhm_energy = sqrt(energy) keV
  → in channels: fwhm_ch = sqrt(energy) / ENERGY_SLOPE

This FWHM shape was chosen because it is exactly what StandardHPGeFWHMModel
can fit (dominant sqrt(E) term), so generate() has a clean convergence path.
"""

import numpy as np
import pytest

from scispectrum.core import Spectrum
from scispectrum.calibration.axis import AxisCalibration
from scispectrum.calibration.resolution import ResolutionCalibration
from scispectrum.calibration.detector_calibration import DetectorCalibration
from scispectrum.calibration.models.energy_poly import PolynomialEnergyModel
from scispectrum.calibration.models.hpge_fwhm_model import StandardHPGeFWHMModel

# ---------------------------------------------------------------------------
# Ground truth constants
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)

N_CHANNELS = 2001
CHANNELS = np.arange(N_CHANNELS, dtype=float)

ENERGY_SLOPE  = 0.3    # keV / channel
ENERGY_OFFSET = 10.0   # keV

# Five well-separated peaks spanning the dynamic range
TRUE_CENTERS_CH = np.array([200.0, 500.0, 800.0, 1200.0, 1700.0])
TRUE_ENERGIES   = ENERGY_SLOPE * TRUE_CENTERS_CH + ENERGY_OFFSET

# FWHM grows like sqrt(energy) — matches StandardHPGeFWHMModel shape
TRUE_FWHM_ENERGY_KEV = np.sqrt(TRUE_ENERGIES)                        # keV
TRUE_FWHM_CH         = TRUE_FWHM_ENERGY_KEV / ENERGY_SLOPE           # channels
TRUE_SIGMA_CH        = TRUE_FWHM_CH / (2 * np.sqrt(2 * np.log(2)))

PEAK_AMPLITUDE = 3000.0   # counts at peak centre — high to keep Poisson noise small
BACKGROUND     = 20.0     # flat background counts

# Domain windows: ±3 sigma around each peak
PEAK_DOMAINS = [
    (max(0, int(c - 3 * s)), min(N_CHANNELS - 1, int(c + 3 * s)))
    for c, s in zip(TRUE_CENTERS_CH, TRUE_SIGMA_CH)
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_counts(rng):
    counts = np.full(N_CHANNELS, BACKGROUND, dtype=float)
    for c, s in zip(TRUE_CENTERS_CH, TRUE_SIGMA_CH):
        counts += PEAK_AMPLITUDE * np.exp(-(CHANNELS - c) ** 2 / (2 * s ** 2))
    return rng.poisson(counts).astype(float)


# ---------------------------------------------------------------------------
# Fixtures  (module-scoped so the spectrum is built once per test session)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synthetic_spectrum():
    return Spectrum(counts=_build_counts(RNG))


@pytest.fixture(scope="module")
def detector_cal(synthetic_spectrum):
    return DetectorCalibration(
        spectrum=synthetic_spectrum,
        known_axis_values=TRUE_ENERGIES.tolist(),
        peak_domains=PEAK_DOMAINS,
        energy_model=PolynomialEnergyModel(degree=1),
        fwhm_model=StandardHPGeFWHMModel(),
    )


@pytest.fixture(scope="module")
def generated(detector_cal):
    """Run generate() once and share the result across all generate() tests."""
    return detector_cal.generate()


# ---------------------------------------------------------------------------
# estimate_peaks() tests
# ---------------------------------------------------------------------------

def test_estimate_peaks_returns_correct_count(detector_cal):
    centers, fwhms = detector_cal.estimate_peaks()
    assert len(centers) == len(TRUE_CENTERS_CH)
    assert len(fwhms)   == len(TRUE_CENTERS_CH)


def test_estimate_peaks_centers_are_ordered(detector_cal):
    """Centres must come back in the same order as the supplied domains."""
    centers, _ = detector_cal.estimate_peaks()
    assert np.all(np.diff(centers) > 0)


def test_estimate_peaks_center_accuracy(detector_cal):
    """Each estimated centre must be within 1 precent true Gaussian centre."""
    centers, fwhms = detector_cal.estimate_peaks()
    assert np.all(np.isclose(centers, TRUE_CENTERS_CH, rtol=0, atol=np.maximum(fwhms*1e-2,2)))


def test_estimate_peaks_fwhm_accuracy(detector_cal):
    """Each estimated FWHM must be within 15 % of the true value."""
    _, fwhms = detector_cal.estimate_peaks()
    relative_errors = np.abs(fwhms - TRUE_FWHM_CH) / TRUE_FWHM_CH
    assert np.all(relative_errors < 0.15), (
        f"FWHM relative errors: {relative_errors.round(3)} — expected all < 0.15"
    )


def test_estimate_peaks_tracks_spectrums_current_axis_units(synthetic_spectrum, generated):
    """estimate_peaks() intentionally reports in whatever axis units
    self.spectrum currently has: channels before calibration, and the
    calibrated axis afterwards (since self.spectrum is held by reference).
    This lets a second call after set_axis_calibration() re-measure peak
    positions directly in calibrated units, independent of the fitted
    calibration function — exactly how the calibration.ipynb example verifies
    its fit."""
    # Use a fresh spectrum so this test doesn't mutate the module-scoped fixture
    fresh = Spectrum(counts=synthetic_spectrum.counts.copy())
    detector_cal = DetectorCalibration(
        spectrum=fresh,
        known_axis_values=TRUE_ENERGIES.tolist(),
        peak_domains=PEAK_DOMAINS,
        energy_model=PolynomialEnergyModel(degree=1),
        fwhm_model=StandardHPGeFWHMModel(),
    )
    centers_ch, _ = detector_cal.estimate_peaks()
    np.testing.assert_allclose(centers_ch, TRUE_CENTERS_CH, atol=2.0)

    axis_calib, res_calib = generated
    fresh.set_axis_calibration(axis_calib)
    fresh.set_resolution_calibration(res_calib)

    centers_energy, _ = detector_cal.estimate_peaks()
    np.testing.assert_allclose(centers_energy, TRUE_ENERGIES, atol=1.0)


def test_generate_unaffected_by_preexisting_spectrum_calibration(synthetic_spectrum):
    """generate() must fit the same channel->energy calibration whether or
    not the spectrum it's given already has some (unrelated) axis calibration
    attached — this is what broke domain_fitting.ipynb: building the Spectrum
    with a calibration already applied before DetectorCalibration.generate()
    silently fit calibrated-axis values against themselves instead of against
    raw channels."""
    precalibrated = Spectrum(
        counts=synthetic_spectrum.counts.copy(),
        axis_calib=AxisCalibration(lambda ch: 2.0 * ch + 5.0, name="bogus"),
    )
    detector_cal = DetectorCalibration(
        spectrum=precalibrated,
        known_axis_values=TRUE_ENERGIES.tolist(),
        peak_domains=PEAK_DOMAINS,
        energy_model=PolynomialEnergyModel(degree=1),
        fwhm_model=StandardHPGeFWHMModel(),
    )
    axis_calib, _ = detector_cal.generate()
    recovered = axis_calib.apply(TRUE_CENTERS_CH)
    assert np.all(np.abs(recovered - TRUE_ENERGIES) < 1.0), (
        f"Energy errors (keV): {np.abs(recovered - TRUE_ENERGIES).round(3)} — expected all < 1.0 keV"
    )


# ---------------------------------------------------------------------------
# generate() tests
# ---------------------------------------------------------------------------

def test_generate_returns_correct_types(generated):
    axis_calib, res_calib = generated
    assert isinstance(axis_calib, AxisCalibration)
    assert isinstance(res_calib,  ResolutionCalibration)


def test_generate_energy_calibration_accuracy(generated):
    """Calibrated energies at the known peak channels must be within 1 keV of truth."""
    axis_calib, _ = generated
    recovered = axis_calib.apply(TRUE_CENTERS_CH)
    errors = np.abs(recovered - TRUE_ENERGIES)
    assert np.all(errors < 1.0), (
        f"Energy errors (keV): {errors.round(3)} — expected all < 1.0 keV"
    )


def test_generate_resolution_calibration_positive(generated):
    """Resolution calibration must return positive FWHM values across the energy range."""
    _, res_calib = generated
    fwhm_values = res_calib.apply(TRUE_ENERGIES)
    assert np.all(fwhm_values > 0), (
        f"Got non-positive FWHM values: {fwhm_values}"
    )


def test_generate_resolution_calibration_accuracy(generated):
    """FWHM from resolution calibration must be within 20 % of the true FWHM in keV."""
    _, res_calib = generated
    recovered = res_calib.apply(TRUE_ENERGIES)
    relative_errors = np.abs(recovered - TRUE_FWHM_ENERGY_KEV) / TRUE_FWHM_ENERGY_KEV
    assert np.all(relative_errors < 0.20), (
        f"FWHM relative errors: {relative_errors.round(3)} — expected all < 0.20"
    )


def test_generate_calibrations_attachable_to_spectrum(synthetic_spectrum, generated):
    """Calibrations returned by generate() must be accepted by set_*_calibration()."""
    axis_calib, res_calib = generated
    # Use a fresh spectrum so this test doesn't mutate the module-scoped fixture
    fresh = Spectrum(counts=synthetic_spectrum.counts.copy())
    fresh.set_axis_calibration(axis_calib)
    fresh.set_resolution_calibration(res_calib)
    assert fresh.axis_calib     is axis_calib
    assert fresh.resolution_calib is res_calib
