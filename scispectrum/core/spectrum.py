import warnings
import numpy as np
import xarray as xr
from typing import Optional
from uncertainties.unumpy import uarray, nominal_values, std_devs
from scispectrum.calibration import AxisCalibration, ResolutionCalibration

class Spectrum:
    """
    Generic 1D spectrum.

    Parameters
    ----------
    counts : np.ndarray
        1D array of measured counts.
    counts_err : np.ndarray, optional
        Per-channel uncertainty on "counts", same shape. Zero entries are
        replaced by 1 with a warning.
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
        *,
        counts_err: np.ndarray = None,
        axis_calib: Optional[AxisCalibration] = None,
        resolution_calib: Optional[ResolutionCalibration] = None,
        metadata: Optional[dict] = None,
    ):
        if not (isinstance(counts, np.ndarray) and counts.ndim == 1):
            raise TypeError("counts must be a 1D numpy array")
        self.counts = counts

        self.channels = np.arange(len(counts))

        if counts_err is not None:
            if not (isinstance(counts_err, np.ndarray) and counts_err.shape == counts.shape):
                raise TypeError("counts error must have the same shape as counts")
            self.counts_err = self._replace_zero_errors(counts_err)
        else:
            self.counts_err = None

        self.axis_calib = axis_calib
        self.resolution_calib = resolution_calib
        self.metadata = metadata or {}

        # Build axis
        if self.axis_calib:
            self.axis = self.axis_calib.apply(self.channels)
            self.axis_name = self.axis_calib.name or "axis"
        else:
            self.axis_calib = AxisCalibration(lambda x: x, name="channel")
            self.axis = self.axis_calib.apply(self.channels)
            self.axis_name = "channel"

        # Xarray representation
        self._xr = xr.DataArray(self.counts, coords={self.axis_name: self.axis}, dims=[self.axis_name])

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    @staticmethod
    def _replace_zero_errors(counts_err: np.ndarray) -> np.ndarray:
        """Replace zero uncertainties by 1, warning that it was done."""
        zeros = counts_err == 0
        if not zeros.any():
            return counts_err

        warnings.warn(
            f"{zeros.sum()} of {zeros.size} entries in counts_err are zero. "
            f"They are replaced by 1."
        )
        counts_err = np.array(counts_err, dtype=float)
        counts_err[zeros] = 1.0
        return counts_err

    @property
    def data(self) -> xr.DataArray:
        return self._xr

    @property
    def data_with_errors(self) -> xr.DataArray:
        if self.counts_err is None:
            raise ValueError("The spectrum has no errors")
        return xr.DataArray(uarray(self.counts, self.counts_err), coords={self.axis_name: self.axis}, dims=[self.axis_name])

    # ---------------------------
    # Public methods
    # ---------------------------

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

    def domain(self, start_val: float, stop_val: float, background=None, background_err=None):
        """
        Create a Domain from physical axis values.

        Parameters
        ----------
        start_val : float
            Start value in axis units (e.g. keV).
        stop_val : float
            Stop value in axis units.
        background : np.ndarray, optional
            Background to subtract from the domain.
        background_err : np.ndarray, optional
            1-sigma uncertainty of the background.

        Returns
        -------
        Domain
        """
        from scispectrum.core.domain import Domain  # lazy import to avoid circular dependency

        axis = self.axis
        indices = np.where((axis >= start_val) & (axis <= stop_val))[0]

        if len(indices) == 0:
            raise ValueError(
                f"No channels found between {start_val} and {stop_val}. "
                f"Axis range is [{axis.min():.3f}, {axis.max():.3f}]"
            )

        return Domain(
            spectrum=self,
            start=int(indices[0]),
            stop=int(indices[-1] + 1),
            background=background,
            background_err=background_err,
        )

    @classmethod
    def from_dataframe(cls, df,counts_col="counts",
                       *,
                       counts_error_col=None,
                       channel_col=None,
                       axis_calib: Optional[AxisCalibration] = None,
                       resolution_calib: Optional[ResolutionCalibration] = None,
                       metadata: Optional[dict] = None):
        """
    Create a Spectrum from a pandas DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing spectrum data.
    counts_col : str, optional
        Name of the column containing counts. Defaults to "counts".
    counts_error_col : str, optional
        Name of the column containing counts uncertainties.
        If None, no errors are assigned to the spectrum.
    channel_col : str, optional
        Name of the column containing channel indices.
        If None, channels default to np.arange(len(counts)).
    axis_calib : AxisCalibration, optional
        Axis calibration mapping channels to physical values.
        If None, the axis is set to the channel indices.
    resolution_calib : ResolutionCalibration, optional
        Resolution calibration for the spectrum.
        If None, no resolution calibration is assigned.
    metadata : dict, optional
        Additional spectrum metadata (e.g. detector info, live time).

    Returns
    -------
    Spectrum
        A new Spectrum instance constructed from the DataFrame.

    Raises
    ------
    ValueError
        If any of the specified column names are not found in the DataFrame.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"channel": [0, 1, 2], "counts": [10, 20, 30], "err": [1, 2, 3]})
    >>> spectrum = Spectrum.from_dataframe(df, channel_col="channel", counts_error_col="err")
    """

        if counts_col not in df.columns:
            raise ValueError(
                f"counts column '{counts_col}' not found in DataFrame. Available columns: {list(df.columns)}")

        channels = None
        if channel_col is not None:
            if channel_col not in df.columns:
                raise ValueError(
                    f"channel column '{channel_col}' not found in DataFrame. Available columns: {list(df.columns)}")
            channels = df[channel_col].to_numpy()

        counts_err = None
        if counts_error_col is not None:
            if counts_error_col not in df.columns:
                raise ValueError(
                    f"counts error column '{counts_error_col}' not found in DataFrame. Available columns: {list(df.columns)}")
            counts_err = df[counts_error_col].to_numpy()

        return cls(
            counts=df[counts_col].to_numpy(),
            counts_err=counts_err,
            axis_calib=axis_calib,
            resolution_calib=resolution_calib,
            metadata=metadata
        )

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

    # ---------------------------
    # Arithmetic operations
    # ---------------------------

    def _apply_operation(self, other, op):

        # --- self values ---
        if self.counts_err is not None:
            self_values = uarray(self.counts, self.counts_err)
        else:
            self_values = self.counts

        # --- other values ---
        if isinstance(other, Spectrum):

            if other.counts_err is not None:
                other_values = uarray(other.counts, other.counts_err)
            else:
                other_values = other.counts

        else:
            # scalar or ndarray
            other_values = other

        # --- operation ---
        result = op(self_values, other_values)

        # --- extract values/errors ---
        if hasattr(result, "dtype") and result.dtype == object:
            result_counts = nominal_values(result)
            result_err = std_devs(result)
        else:
            result_counts = result
            result_err = None

        return Spectrum(
            counts=result_counts,
            counts_err=result_err,
            axis_calib=self.axis_calib,
            resolution_calib=self.resolution_calib,
            metadata=self.metadata.copy()
        )
    def __add__(self, other): return self._apply_operation(other, lambda x, y: x + y)
    def __sub__(self, other): return self._apply_operation(other, lambda x, y: x - y)
    def __mul__(self, other): return self._apply_operation(other, lambda x, y: x * y)
    def __truediv__(self, other): return self._apply_operation(other, lambda x, y: x / y)
    def __pow__(self, other): return self._apply_operation(other, lambda x, y: x ** y)