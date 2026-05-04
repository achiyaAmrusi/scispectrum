import xarray as xr
from abc import ABC, abstractmethod


class PeakFit(ABC):
    """
    Abstract base class for peak fitting methods.
    Any peak fitting class must implement the `fit` and `plot_fit` methods.
    Must be implemented by subclasses.
    Methods
    ----------
    fit: callable
    given spectrum slice and fitting data (initial data which is required from the fitting methods) the function
    returns the fit
    evaluate: callable
    evaluate the fit from the results of fitting methods
    """

    @staticmethod
    @abstractmethod
    def fit(counts, axis, **kwargs):
        """
        Fit the given spectrum slice and return the fit properties.
        """
        pass

    @staticmethod
    @abstractmethod
    def evaluate(domain, fit_properties):
        """
        evaluate the result int the domain.
        """
        pass

