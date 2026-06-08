import numpy as np
import pytest
from scispectrum.calibration import ResolutionCalibration, AxisCalibration

AXIS_FACTOR = 2


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def resolution_calibration():

    res_calib = ResolutionCalibration(lambda x: np.sqrt(x))

    return res_calib
@pytest.fixture
def axis_calibration():
    channels = np.linspace(0, 1000, 1001)
    poly = np.poly1d([2.0, 0 ])
    axis_calib = AxisCalibration(func=poly, name="channel")

    return axis_calib
# -----------------------------------------------------------------------------
# Test calibrations apply
# -----------------------------------------------------------------------------

def test_calibration_resolution_application(resolution_calibration:ResolutionCalibration):
    channels = np.linspace(0, 1000, 1001)

    assert np.all(resolution_calibration.apply(channels) == channels ** 0.5)

def test_calibration_axis_application(axis_calibration: AxisCalibration):
    channels = np.linspace(0, 1000, 1001)

    assert np.all(axis_calibration.apply(channels) == AXIS_FACTOR * channels)
