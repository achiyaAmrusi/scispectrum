from abc import ABC, abstractmethod
from typing import Callable, Sequence
from scipy.optimize import curve_fit
import numpy as np


class BaseCalibrationModel(ABC):
    """
    Parametric calibration model.

    Defines:
    - a mathematical form
    - how to generate a callable from parameters
    - how to fit parameters from data
    """

    @abstractmethod
    def function(self, x: np.ndarray, *params) -> np.ndarray:
        """Parametric calibration function."""
        pass

    def generator(self, params: Sequence[float]) -> Callable[[float], float]:
        """Freeze parameters and return a callable."""
        def f(x):
            return self.function(np.asarray(x), *params)
        return f

    def fit(self, x: np.ndarray, y: np.ndarray, p0=None):
        """Fit parameters to data."""
        params, cov = curve_fit(self.function, x, y, p0=p0)
        return params, cov
