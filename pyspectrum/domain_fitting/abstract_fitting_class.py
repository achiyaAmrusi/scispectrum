import xarray as xr
import numpy as np
from abc import ABC, abstractmethod


class MultiPeakFitter(ABC):
    """
    Abstract base class for multi-peak fitting on a spectral Domain.

    Subclasses configure the model in ``__init__`` and implement
    ``fit`` and ``evaluate``.
    """

    @abstractmethod
    def fit(self, domain, **kwargs) -> "xr.Dataset | bool":
        """Fit the model to a spectral domain. Returns xr.Dataset or False."""

    @abstractmethod
    def evaluate(self, axis: np.ndarray, dataset: xr.Dataset) -> np.ndarray:
        """Evaluate the model on *axis* using parameters from *dataset*."""
