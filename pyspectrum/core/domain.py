from typing import Optional, Dict, Any
import numpy as np
import xarray as xr

from pyspectrum.core.spectrum import Spectrum


class Domain:
    """
    Domain represents a region of a Spectrum.

    A Domain is:
    - contiguous in index space
    - interpretation-light
    - tied to a parent Spectrum

    It exposes its data exclusively as an xarray.DataArray.
    """

    def __init__(
        self,
        spectrum: Spectrum,
        start: int,
        stop: int,
    ):
        if start < 0 or stop <= start:
            raise ValueError("Invalid domain bounds")

        self.spectrum = spectrum
        self.start = int(start)
        self.stop = int(stop)

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------


    @property
    def data(self) -> xr.DataArray:
        da = self.spectrum.xr_spectrum().isel(
            **{self.spectrum.axis_name: slice(self.start, self.stop)}
        )

        attrs = dict(da.attrs)
        attrs.update({
            "domain_start": self.start,
            "domain_stop": self.stop,
        })

        return da.assign_attrs(attrs)

    @property
    def indices(self) -> np.ndarray:
        return np.arange(self.start, self.stop)

    @property
    def local_resolution(self):
        axis = self.data.coords[self.spectrum.axis_name].values
        center = axis.mean()
        return self.spectrum.resolution_calib(center)
    # ------------------------------------------------------------------
    # Peak conversion
    # ------------------------------------------------------------------

    def to_peak(self, fitter=None):
        from pyspectrum.core.peak import Peak
        return Peak.from_domain(self)

    # ---------------------------
    # Array-like access
    # ---------------------------

    def __getattr__(self, name):
        """Delegate attribute access to xarray DataArray."""
        return getattr(self.data, name)

    def __getitem__(self, key):
        """Allow indexing like spectrum[key]."""
        return self.data[key]

    def __repr__(self):
        return repr(self.data)