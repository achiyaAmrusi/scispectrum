from abc import ABC, abstractmethod

import numpy as np


class BackgroundEstimator(ABC):
    """
    Abstract base class for background estimation algorithms.

    All auxiliary inputs (resolution calibration, convolution objects, etc.)
    must be passed at construction time, not to estimate().
    """

    @abstractmethod
    def estimate(self, axis: np.ndarray, counts: np.ndarray) -> np.ndarray:
        """
        Estimate background for a 1D spectrum.

        Parameters
        ----------
        axis : np.ndarray
            Axis values (e.g. energy in keV).
        counts : np.ndarray
            Spectrum counts.

        Returns
        -------
        np.ndarray
            Estimated background, same shape as counts.
        """
        pass
