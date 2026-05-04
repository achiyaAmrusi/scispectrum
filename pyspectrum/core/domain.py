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
        background: np.ndarray = None
    ):
        if start < 0 or stop <= start:
            raise ValueError("Invalid domain bounds")

        self.spectrum = spectrum
        self.start = int(start)
        self.stop = int(stop)
        if background is not None:
            if not len(background) ==  (self.stop - self.start):
                raise ValueError("Background length needs to match domain length")
            else:
                self._background = background
        else:
            self._background = None

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------


    @property
    def data(self) -> xr.DataArray:

        if self.background is None:
            da = self.spectrum.xr_spectrum().isel(
                **{self.spectrum.axis_name: slice(self.start, self.stop)}
            )
        else:
            da = self.spectrum.xr_spectrum().isel(
                **{self.spectrum.axis_name: slice(self.start, self.stop)}
            ) - self.background

        attrs = dict(da.attrs)
        attrs.update({
            "domain_start": self.start,
            "domain_stop": self.stop,
        })

        return da.assign_attrs(attrs)

    @property
    def background(self):
        return self._background

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

    def set_background(self, background: np.ndarray | None):
        if (background is not None) and (not len(background) == (self.stop - self.start)):
            raise ValueError("Background length needs to match domain length")
        else:
            self._background = background
        return self.data

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