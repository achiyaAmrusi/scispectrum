from typing import Optional, Dict, Any
import numpy as np
import xarray as xr

from pyspectrum.core.spectrum import Spectrum


class Domain:
    """
    Domain represents a contiguous, statistically significant region
    of a Spectrum.

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
        *,
        metadata: Optional[Dict[str, Any]] = None,
        score: Optional[float] = None,
        method: Optional[str] = None,
    ):
        if start < 0 or stop <= start:
            raise ValueError("Invalid domain bounds")

        self.spectrum = spectrum
        self.start = int(start)
        self.stop = int(stop)

        self.metadata = metadata or {}
        self.score = score
        self.method = method

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    @property
    def data(self) -> xr.DataArray:
        if hasattr(self, "_background_override"):
            return self._background_override
        return self._base_data()

    def _base_data(self) -> xr.DataArray:
        da = self.spectrum.xr_spectrum().isel(
            channel=slice(self.start, self.stop)
        )

        attrs = dict(da.attrs)
        attrs.update({
            "domain_start": self.start,
            "domain_stop": self.stop,
            "domain_width": self.width,
            "domain_method": self.method,
            "domain_score": self.score,
            **self.metadata,
        })

        return da.assign_attrs(attrs)

    # ------------------------------------------------------------------
    # Basic properties
    # ------------------------------------------------------------------

    @property
    def width(self) -> int:
        return self.stop - self.start

    @property
    def indices(self) -> np.ndarray:
        return np.arange(self.start, self.stop)

    # ------------------------------------------------------------------
    # Background handling
    # ------------------------------------------------------------------

    def subtract_background(self, background) -> "Domain":
        da = self.data
        bg = background(self) if callable(background) else background

        new = Domain(
            spectrum=self.spectrum,
            start=self.start,
            stop=self.stop,
            metadata=self.metadata.copy(),
            score=self.score,
            method=self.method,
        )

        new._background_override = da - bg
        return new

    # ------------------------------------------------------------------
    # Fitting / peak conversion hooks
    # ------------------------------------------------------------------

    def fit(self, fitter):
        return fitter.fit(self)

    def to_peak(self, fitter=None):
        from pyspectrum.core.peak import Peak

        if fitter is None:
            return Peak.from_domain(self)
        return fitter.fit(self)

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __len__(self):
        return self.width

    def __repr__(self):
        return (
            f"Domain(start={self.start}, stop={self.stop}, "
            f"width={self.width}, method={self.method})"
        )
