import pandas as pd
import numpy as np
from pyspectrum.spectrum import Spectrum


class TimeChannelParser:
    """
    Parser class for list mode into Spectrum.
    The methods parse the data, filters negative counts and can process pileup or flag indicator (require alert_column == 0).

    Methods
    -------
    to_spectrum(time_channel_df: pd.DataFrame, energy_calibration_poly=np.poly1d([1, 0]), fwhm_calibration=None, num_of_channels=2 ** 14)
    Filter the time_channel file from pileup and negative counts
    then transform data to spectrum using the calibrations
    counts_in_time_into_spectrum(cls, time_channel_df: pd.DataFrame, num_of_channels=2 ** 14)
     Takes a data frame with time stemp, count and pileup and turn into spectrum in dataframe form
    """
    def __init__(self):
        pass

    @staticmethod
    def filter(time_channel_df: pd.DataFrame, flag = True):
        """
        Filter the time_channel data frame from alerted counts (pileup or overflow) and negative counts.
        Parameters
        ----------
        time_channel_df: pd.Dataframe
         unfiltered list mode table
        flag: bool (default True)
        is there a pileup or overflow indicator
        Returns
        -------
        pd.DataFrame
         filtered dataframe of time - channel
        """
        # 3 data columns for each row
        if flag:
            filtered_data = time_channel_df[
                (time_channel_df['flag'] == 0) & (time_channel_df['channel'] >= 0)
                ]
        else:
            filtered_data = time_channel_df[(time_channel_df['channel'] >= 0)]
        return filtered_data[['time', 'channel']]

    @staticmethod
    def to_spectrum(time_channel_df: pd.DataFrame,
                    energy_calibration_poly=np.poly1d([1, 0]), fwhm_calibration=None, num_of_channels=2 ** 14):
        """
        Filter the time_channel file from pileup and negative counts,
        then transform data to spectrum using the calibrations
        If an alert flag was used in the MCA ot should be 0 to be accounted as a valid count.

        Parameters
        ----------
        time_channel_df: pd.DataFrame
        a table of time - channel, also time - channel - alert-flag is optional
        energy_calibration_poly: numpy.poly1d([a, b])
         the energy calibration of the detector
        fwhm_calibration: Callable
        a function that given energy/channel(first raw in file) returns the fwhm
        num_of_channels: int
        the number of channels in the measurement
        Returns
        -------
        Spectrum
         the final spectrum
        """
        time_channel_df = TimeChannelParser.filter(time_channel_df)
        spectrum_df = TimeChannelParser.counts_in_time_into_spectrum(time_channel_df, num_of_channels)
        return Spectrum.from_dataframe(spectrum_df, energy_calibration_poly, fwhm_calibration)

    @classmethod
    def counts_in_time_into_spectrum(cls, time_channel_df: pd.DataFrame, num_of_channels=2 ** 14):
        """
        Takes a dataframe of time stamp and counts  and turn it into spectrum.

        Parameters
        ----------
        time_channel_df: pd.DataFrame
        a table of time - channel
        num_of_channels: int default = 2**14
        The number of channels in the detector
        Returns
        -------
        pd.DataFrame
         filtered dataframe of the time channel file
        """
        # Use NumPy histogram instead of Pandas' value_counts for speed
        channel_array = time_channel_df['channel'].to_numpy()
        counts = np.bincount(channel_array, minlength=num_of_channels)
        counts[-1] = 0
        return pd.DataFrame({'counts':counts,
                             'channel': np.arange(num_of_channels)})


