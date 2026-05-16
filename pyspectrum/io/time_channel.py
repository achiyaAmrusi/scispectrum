import pandas as pd
import numpy as np
from pathlib import Path
from pyspectrum.core import Spectrum
from typing import Optional
from pyspectrum.calibration import AxisCalibration, ResolutionCalibration

class TimeChannelParser:
    """
    Convert time–channel list-mode data into a Spectrum.

    The parser filters invalid events (negative channels or flagged
    pileup/overflow) and builds a histogram of counts per channel.

    Large files can be processed using chunked reading via `from_file`,
    which avoids loading the entire dataset into memory.

    Methods
    -------
    from_file(...)
        Parse a list-mode file in chunks and build a Spectrum.

    from_dataframe(...)
        Convert an in-memory dataframe into a Spectrum.

    filter(...)
        Identify valid events based on channel and optional flag.
    """
    def __init__(self):
        pass

    @staticmethod
    def filter(channel: np.ndarray, flag: np.ndarray =None):
        """
        Filter the time_channel data frame from alerted counts (pileup or overflow) and negative counts.
        Parameters
        ----------
        channel: np.ndarray
         unfiltered channel vector
        flag : np.ndarray, optional
        Array indicating pileup/overflow events. Non-zero values are rejected.
        Returns
        -------
        valid_indecies
            valid events indeceies
        """

        # 3 data columns for each row
        if flag is not None:
            valid_indices = (flag == 0) & (channel >= 0)
        else:
            valid_indices = (channel >= 0)
        return valid_indices

    @staticmethod
    def from_file(sourcefile,
                  axis_calib: Optional[AxisCalibration] = None,
                  resolution_calib: Optional[ResolutionCalibration] = None,
                  num_of_channels=2**14, chunk_size=100_000, **kwargs):
        """
        Parse a large list-mode file into a Spectrum.
        The file is read in chunks to avoid loading the entire dataset
        into memory. Each chunk is filtered and accumulated into the
        final histogram.

        Parameters
        ----------
        sourcefile : str or Path
            Path to the list-mode file.
        axis_calib : AxisCalibration, optional
            Axis calibration mapping channels -> physical values.
        resolution_calib : ResolutionCalibration, optional
            Optional resolution calibration.
        num_of_channels : int
            Number of channels in the spectrum.
        chunk_size : int
            Number of rows to read per chunk.
        **kwargs
            Additional arguments passed to `pd.read_csv`.

        Returns
        -------
        Spectrum
        """

        if not isinstance(sourcefile, (str, Path)):
            raise TypeError("sourcefile must be a path")

        counts = np.zeros(num_of_channels, dtype=np.int64)
        reader = pd.read_csv(sourcefile, chunksize=chunk_size, **kwargs)

        for chunk in reader:
            channel = chunk["channel"].to_numpy()

            if "flag" in chunk.columns:
                flag = chunk["flag"].to_numpy()
            else:
                flag = None

            valid = TimeChannelParser.filter(channel, flag)
            channel = channel[valid]

            counts += np.bincount(channel, minlength=num_of_channels)

        counts[-1] = 0

        spectrum_df = pd.DataFrame({
            "channel": np.arange(num_of_channels),
            "counts": counts,
            "counts_error": np.sqrt(counts)
        })

        return Spectrum.from_dataframe(
            spectrum_df,
            channel_col="channel",
            counts_col="counts",
            counts_error_col="counts_error",
            axis_calib=axis_calib,
            resolution_calib=resolution_calib
        )

    @staticmethod
    def from_dataframe(df,
                       axis_calib: Optional[AxisCalibration] = None,
                       resolution_calib: Optional[ResolutionCalibration] = None,
                       num_of_channels=2**14):
        """
        Convert an in-memory time-channel dataframe into a Spectrum.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing at least a 'channel' column.
            Optional columns: 'time', 'flag'.
        axis_calib : AxisCalibration, optional
            Axis calibration mapping channels -> physical values.
        resolution_calib : ResolutionCalibration, optional
            Optional resolution calibration.
        num_of_channels : int
            Number of detector channels.

        Returns
        -------
        Spectrum
        """
        if "channel" not in df.columns:
            raise ValueError(f"DataFrame must contain a 'channel' column. Found: {list(df.columns)}")

        channel = df["channel"].to_numpy()
        if "flag" in df.columns:
            flag = df["flag"].to_numpy()
        else:
            flag=None

        valid = TimeChannelParser.filter(channel, flag)
        channel = channel[valid]

        counts = np.bincount(channel, minlength=num_of_channels)
        counts[-1] = 0

        spectrum_df = pd.DataFrame({
            "channel": np.arange(num_of_channels),
            "counts": counts,
            "counts_error": np.sqrt(counts)
        })

        return Spectrum.from_dataframe(
            spectrum_df,
            channel_col="channel",
            counts_col="counts",
            counts_error_col="counts_error",
            axis_calib=axis_calib,
            resolution_calib=resolution_calib
        )

