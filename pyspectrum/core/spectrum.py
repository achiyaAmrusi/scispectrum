import numpy as np
import xarray as xr
from typing import Optional
from pyspectrum.calibration import AxisCalibration, ResolutionCalibration

class Spectrum:
    """
    Generic 1D spectrum.

    Parameters
    ----------
    counts : np.ndarray
        1D array of measured counts.
    channels : np.ndarray, optional
        1D array of raw channel indices. Defaults to np.arange(len(counts)).
    axis_calib : AxisCalibration, optional
        Axis calibration mapping channels -> physical values.
    resolution_calib : ResolutionCalibration, optional
        Optional resolution calibration.
    metadata : dict, optional
        Additional spectrum metadata (detector info, live time, etc.).

    Attributes
    ----------
    counts : np.ndarray
        Array of counts.
    channels : np.ndarray
        Array of channels.
    axis : np.ndarray
        Calibrated axis (same as channels if no calibration provided).
    axis_name : str
        Name of the axis (default "channel").
    _xr : xr.DataArray
        Xarray representation of the spectrum for slicing and plotting.
    metadata : dict
        Spectrum metadata.
    """
    def __init__(
        self,
        counts: np.ndarray,
        channels: Optional[np.ndarray] = None,
        *,
        axis_calib: Optional[AxisCalibration] = None,
        resolution_calib: Optional[ResolutionCalibration] = None,
        metadata: Optional[dict] = None,
    ):
        if not (isinstance(counts, np.ndarray) and counts.ndim == 1):
            raise TypeError("counts must be a 1D numpy array")
        self.counts = counts

        self.channels = channels if channels is not None else np.arange(len(counts))
        if not (isinstance(self.channels, np.ndarray) and self.channels.ndim == 1):
            raise TypeError("channels must be a 1D numpy array")

        self.axis_calib = axis_calib
        self.resolution_calib = resolution_calib
        self.metadata = metadata or {}

        # Build axis
        if self.axis_calib:
            self.axis = self.axis_calib.apply(self.channels)
            self.axis_name = self.axis_calib.name or "axis"
        else:
            self.axis = self.channels
            self.axis_name = "channel"

        # Xarray representation
        self._xr = xr.DataArray(self.counts, coords={self.axis_name: self.axis}, dims=[self.axis_name])

    # ---------------------------
    # Public methods
    # ---------------------------

    def xr_spectrum(self) -> xr.DataArray:
        """Return xarray representation of the spectrum."""
        return self._xr

    def set_axis_calibration(self, axis_calib: AxisCalibration):
        """Set or update axis calibration."""
        if not isinstance(axis_calib, AxisCalibration):
            raise TypeError("calibration must be a Calibration object")
        self.axis_calib = axis_calib
        self.axis = axis_calib.apply(self.channels)
        self.axis_name = axis_calib.name or "axis"
        self._xr = xr.DataArray(self.counts, coords={self.axis_name: self.axis}, dims=[self.axis_name])

    def set_resolution_calibration(self, resolution_calib: ResolutionCalibration):
        """Set or update resolution calibration."""
        if not isinstance(resolution_calib, ResolutionCalibration):
            raise TypeError("resolution must be a ResolutionCalibration object")
        self.resolution_calib = resolution_calib

    @classmethod
    def from_dataframe(cls, df, channel_col="channel", counts_col="counts",
                       *,
                       axis_calib: Optional[AxisCalibration] = None,
                       resolution_calib: Optional[ResolutionCalibration] = None,
                       metadata: Optional[dict] = None):
        """Create Spectrum from pandas DataFrame."""
        return cls(
            counts=df[counts_col].to_numpy(),
            channels=df[channel_col].to_numpy(),
            axis_calib=axis_calib,
            resolution_calib=resolution_calib,
            metadata=metadata
        )

    # ---------------------------
    # Array-like access
    # ---------------------------

    def __getattr__(self, name):
        """Delegate attribute access to xarray DataArray."""
        return getattr(self._xr, name)

    def __getitem__(self, key):
        """Allow indexing like spectrum[key]."""
        return self._xr[key]

    def __repr__(self):
        return repr(self._xr)

    # ---------------------------
    # Arithmetic operations
    # ---------------------------

    def _apply_operation(self, other, op):
        other_values = other._xr.values if isinstance(other, Spectrum) else other
        result_counts = op(self._xr.values, other_values)
        return Spectrum(
            counts=result_counts,
            channels=self.channels,
            axis_calib=self.axis_calib,
            resolution_calib=self.resolution_calib,
            metadata=self.metadata.copy()
        )

    def __add__(self, other): return self._apply_operation(other, lambda x, y: x + y)
    def __sub__(self, other): return self._apply_operation(other, lambda x, y: x - y)
    def __mul__(self, other): return self._apply_operation(other, lambda x, y: x * y)
    def __truediv__(self, other): return self._apply_operation(other, lambda x, y: x / y)
    def __pow__(self, other): return self._apply_operation(other, lambda x, y: x ** y)