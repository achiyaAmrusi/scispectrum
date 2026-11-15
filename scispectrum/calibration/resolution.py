from typing import Union
import numpy as np

class ResolutionCalibration:
    """
    Detector resolution calibration.

    Parameters
    ----------
    func : Callable[[float], float]
        Function returning resolution for a given axis value.
    """
    def __init__(self, func):
        self.func = func

    def __call__(self, axis_value: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Return the resolution at a given axis value."""
        return self.func(axis_value)

    def apply(self, axis_value: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return self.func(axis_value)

    # ---------------------------
    # copy properties
    # ---------------------------

    def __copy__(self):
        return ResolutionCalibration(self.func)

    def __deepcopy__(self, memo):
        new_instance = ResolutionCalibration(self.func)
        memo[id(self)] = new_instance
        return new_instance
