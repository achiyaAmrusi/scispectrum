from abc import ABC, abstractmethod
import numpy as np


class Kernel1D(ABC):
    """
    Abstract base class for 1D convolution kernels.
    """

    def __init__(self, sigma, support_sigma=4.0):
        """
        Parameters
        ----------
        sigma : float
            Standard deviation in channels.
        support_sigma : float
            Kernel half-width in units of sigma.
        """
        self.sigma = float(sigma)
        self.support_sigma = float(support_sigma)

    @property
    def half_width(self):
        """Half-width of kernel in channels."""
        return int(np.ceil(self.support_sigma * self.sigma))

    @property
    def x(self):
        """Discrete coordinate grid."""
        hw = self.half_width
        return np.arange(-hw, hw + 1)

    @abstractmethod
    def values(self):
        """Return kernel values as numpy array."""
        pass

    def normalized(self):
        """Return kernel normalized to unit L1 norm."""
        v = self.values()
        s = np.sum(np.abs(v))
        return v / s if s != 0 else v
