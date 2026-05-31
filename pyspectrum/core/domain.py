from typing import Optional, Dict, Any
import numpy as np
import xarray as xr

from pyspectrum.core.spectrum import Spectrum


class Domain:
    """
    A contiguous region of a Spectrum, defined by index bounds.

    A Domain represents a slice of a parent Spectrum for localized
    analysis — peak fitting, background subtraction, parameter
    calculation, or any operation that applies to a sub-range of
    the spectrum axis.

    Parameters
    ----------
    spectrum : Spectrum
        The parent spectrum this domain belongs to.
    start : int
        Start index into the spectrum (inclusive).
    stop : int
        Stop index into the spectrum (exclusive).
    background : np.ndarray, optional
        Background array to subtract from the domain counts.
        Must match the domain length (stop - start).

    Attributes
    ----------
    data : xr.DataArray
        Domain counts as an xarray DataArray, with background
        subtracted if provided.
    data_with_errors : xr.DataArray
        Domain counts with uncertainty via the uncertainties library.
    background : np.ndarray or None
        The background array, if set.
    indices : np.ndarray
        Array of spectrum indices covered by this domain.

    Methods
    -------
    subtract_background(background)
        Return a new Domain with a background array attached.
        The original Domain is not modified.
    local_resolution()
        Estimate the detector resolution at the domain center,
        requires resolution calibration on the parent Spectrum.
    from_axis_values(spectrum, start_val, stop_val)
        Construct a Domain from physical axis values rather than
        channel indices (e.g. energy in keV).
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
            da = self.spectrum.data.isel(
                **{self.spectrum.axis_name: slice(self.start, self.stop)}
            )
        else:
            da = self.spectrum.data.isel(
                **{self.spectrum.axis_name: slice(self.start, self.stop)}
            ) - self.background

        attrs = dict(da.attrs)
        attrs.update({
            "domain_start": self.start,
            "domain_stop": self.stop,
        })

        return da.assign_attrs(attrs)

    @property
    def data_with_errors(self) -> xr.DataArray:
        if self.background is None:
            da = self.spectrum.data_with_errors.isel(
                **{self.spectrum.axis_name: slice(self.start, self.stop)}
            )
        else:
            da = self.spectrum.data_with_errors.isel(
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
    def axis(self) -> np.ndarray:
        return self.spectrum.axis[self.indices]

    @property
    def local_resolution(self):
        if self.spectrum.resolution_calib is None:
            raise ValueError("Spectrum must have resolution calibration")
        axis = self.data.coords[self.spectrum.axis_name].values
        center = axis.mean()
        return self.spectrum.resolution_calib(center)

    def subtract_background(self, background: np.ndarray | None):

        """
        Return a new Domain with a background attached.

        The background is subtracted lazily when accessing `.data`.
        The original Domain is not modified.
        """
        if (background is not None) and (not len(background) == (self.stop - self.start)):
            raise ValueError("Background length needs to match domain length")

        return Domain(
            spectrum=self.spectrum,
            start=self.start,
            stop=self.stop,
            background=background)

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