from abc import ABC, abstractmethod
import xarray as xr
from pyspectrum.core.spectrum import Spectrum

class BackgroundEstimator(ABC):
    """
    Abstract base class for background estimation algorithms.
    """

    @abstractmethod
    def estimate(self, spectrum: Spectrum, *kwargs) -> xr.DataArray:
        """
        Estimate background for a 1D spectrum.

        Parameters
        ----------
        spectrum : Spectrum
            The spectrum
        Returns
        -------
        background : DataArray
            Estimated background.
        """
        pass
