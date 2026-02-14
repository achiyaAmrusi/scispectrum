from typing import Callable
import numpy as np

class AxisCalibration:
    """
    Generic calibration for a spectrum axis.

    Parameters
    ----------
    func : Callable[[np.ndarray], np.ndarray]
        Function mapping raw channels -> physical values.
    name : str, optional
        Name of the axis (e.g., "energy", "time", "wavelength").
    """
    def __init__(self, func: Callable[[np.ndarray], np.ndarray], name: str = "axis"):
        if not callable(func):
            raise TypeError("mapping must be callable")
        self.func = func
        self.name = name

    def __call__(self, channels)-> np.ndarray:
        """Return the axis values at a given channels."""
        return self.func(channels)

    def apply(self, channels) -> np.ndarray:
        return self.func(channels)
