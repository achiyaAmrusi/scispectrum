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

    @classmethod
    def from_array(cls, axis: np.ndarray, name: str = "axis") -> "AxisCalibration":
        """
        Create an AxisCalibration from a pre-computed physical axis array.

        The calibration is built using linear interpolation over the array,
        so channels are assumed to be integer indices 0, 1, 2, ...

        Parameters
        ----------
        axis : np.ndarray
            Pre-computed physical axis values (e.g. energy in keV).
            Must have the same length as the spectrum counts array.
        name : str, optional
            Name of the axis. Default is "axis".

        Returns
        -------
        AxisCalibration

        Examples
        --------
        >>> axis = np.array([1460.0, 1461.5, 1463.0])
        >>> calib = AxisCalibration.from_array(axis, name="energy_keV")
        >>> calib.apply(np.array([0, 1, 2]))
        array([1460. , 1461.5, 1463. ])
        """
        if not isinstance(axis, np.ndarray) or axis.ndim != 1:
            raise TypeError("axis must be a 1D numpy array")

        channels = np.arange(len(axis))
        func = lambda ch: np.interp(ch, channels, axis)
        return cls(func=func, name=name)