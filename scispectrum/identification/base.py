# domain_finding/base.py

from abc import ABC, abstractmethod

class DomainFinder(ABC):

    @abstractmethod
    def find(self, spectrum):
        """
        Parameters
        ----------
        spectrum : Spectrum

        Returns
        -------
        list[Domain]
        """
        pass
