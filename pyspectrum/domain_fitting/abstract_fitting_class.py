import xarray as xr
from abc import ABC, abstractmethod


class PeakFit(ABC):
    """
    Abstract base class for peak domain_fitting methods.
    Any peak domain_fitting class must implement the `fit` and `plot_fit` methods.
    Must be implemented by subclasses.
    Methods
    ----------
    fit: callable
    given spectrum slice and domain_fitting data (initial data which is required from the domain_fitting methods) the function
    returns the fit
    evaluate: callable
    evaluate the fit from the results of domain_fitting methods
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

