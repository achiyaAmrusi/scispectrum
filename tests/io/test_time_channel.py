"""Tests for TimeChannelParser."""

import numpy as np
import pandas as pd
import pytest

from pyspectrum.core import Spectrum
from pyspectrum.io import TimeChannelParser

N_CHANNELS = 256


# ---------------------------------------------------------------------------
# filter()
# ---------------------------------------------------------------------------

class TestFilter:

    def test_removes_negative_channels(self):
        channels = np.array([-1, 0, 5, 10, 100])
        valid = TimeChannelParser.filter(channels)
        assert valid.sum() == 4
        assert not valid[0]

    def test_keeps_all_non_negative(self):
        channels = np.array([0, 1, 50, 255])
        valid = TimeChannelParser.filter(channels)
        assert valid.all()

    def test_removes_flagged_events(self):
        channels = np.array([1, 2, 3, 4])
        flags    = np.array([0, 1, 0, 0])
        valid = TimeChannelParser.filter(channels, flags)
        assert valid.sum() == 3
        assert not valid[1]

    def test_removes_negative_and_flagged(self):
        channels = np.array([-1, 2, 3, 4])
        flags    = np.array([0,  0, 1, 0])
        valid = TimeChannelParser.filter(channels, flags)
        assert valid.sum() == 2

    def test_no_flag_column(self):
        channels = np.array([0, 10, 20])
        valid = TimeChannelParser.filter(channels, flag=None)
        assert valid.sum() == 3


# ---------------------------------------------------------------------------
# from_dataframe()
# ---------------------------------------------------------------------------

def _make_df(channels, flags=None):
    d = {"channel": channels}
    if flags is not None:
        d["flag"] = flags
    return pd.DataFrame(d)


class TestFromDataframe:

    def test_returns_spectrum(self):
        df = _make_df(np.array([0, 1, 5, 5, 10]))
        s = TimeChannelParser.from_dataframe(df, num_of_channels=N_CHANNELS)
        assert isinstance(s, Spectrum)

    def test_channel_counts_correct(self):
        # channels: 5 appears 3 times, 10 appears 2 times
        channels = np.array([5, 5, 5, 10, 10])
        df = _make_df(channels)
        s = TimeChannelParser.from_dataframe(df, num_of_channels=N_CHANNELS)
        assert s.counts[5]  == 3
        assert s.counts[10] == 2

    def test_negative_channels_filtered(self):
        channels = np.array([-1, 5, 5, 10])
        df = _make_df(channels)
        s = TimeChannelParser.from_dataframe(df, num_of_channels=N_CHANNELS)
        assert s.counts[5] == 2

    def test_flagged_events_filtered(self):
        channels = np.array([5, 5, 10])
        flags    = np.array([0, 1, 0])
        df = _make_df(channels, flags)
        s = TimeChannelParser.from_dataframe(df, num_of_channels=N_CHANNELS)
        assert s.counts[5]  == 1
        assert s.counts[10] == 1

    def test_spectrum_length(self):
        df = _make_df(np.array([0, 10, 20]))
        s = TimeChannelParser.from_dataframe(df, num_of_channels=N_CHANNELS)
        assert len(s.counts) == N_CHANNELS

    def test_missing_channel_column_raises(self):
        df = pd.DataFrame({"time": [1, 2, 3]})
        with pytest.raises(ValueError, match="channel"):
            TimeChannelParser.from_dataframe(df, num_of_channels=N_CHANNELS)

    def test_counts_err_set(self):
        df = _make_df(np.array([5, 5, 10]))
        s = TimeChannelParser.from_dataframe(df, num_of_channels=N_CHANNELS)
        assert s.counts_err is not None
        assert s.counts_err[5] == pytest.approx(np.sqrt(2))
