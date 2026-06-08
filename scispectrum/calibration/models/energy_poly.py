import numpy as np
from .base import BaseCalibrationModel

class PolynomialEnergyModel(BaseCalibrationModel):
    """
    Polynomial axis calibration model.

    Represents:
        axis(ch) = a_n ch^n + ... + a_1 ch + a_0
    """

    def __init__(self, degree: int = 2):
        self.degree = int(degree)

    def function(self, ch: np.ndarray, *coeffs) -> np.ndarray:
        if len(coeffs) != self.degree + 1:
            raise ValueError(
                f"Expected {self.degree + 1} coefficients, got {len(coeffs)}"
            )
        return np.polyval(coeffs, ch)

    def fit(self, x: np.ndarray, y: np.ndarray, p0=None):
        if p0 is None:
            p0 = np.ones(self.degree + 1)
        return super().fit(x, y, p0=p0)
