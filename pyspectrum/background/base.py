from abc import ABC, abstractmethod

import numpy as np
import xarray as xr
from pyspectrum.core.spectrum import Spectrum

class BackgroundEstimator(ABC):
    """
    Abstract base class for background estimation algorithms.
    """

    @abstractmethod
    def estimate(self, axis: np.ndarray, counts: np.ndarray, **kwargs) -> xr.DataArray:
        """
        Estimate background for a 1D spectrum.

        Parameters
        ----------
        axis : np.ndarray
        The axis along which to estimate the background.
        counts : np.ndarray
        The counts of the spectrum along which to estimate the background.

        Returns
        -------
        background : DataArray
            Estimated background.
        """
        pass
