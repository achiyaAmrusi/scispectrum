import numpy as np
from .base import BaseCalibrationModel


class StandardHPGeFWHMModel(BaseCalibrationModel):
    """
    FWHM(E) = a + b * sqrt(E + c * E^2)
    """
    def function(self, energy: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
        return a + b * np.sqrt(np.abs(energy) + np.abs(c) * energy ** 2)
