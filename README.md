# PySpectrum

A Python package for 1D spectrum analysis, designed for physicists and scientists working with detector data such as gamma-ray, X-ray, or particle spectra.

PySpectrum provides a clean, extensible framework for the full spectrum analysis pipeline — from raw detector output to calibrated, fitted, background-subtracted results — with proper uncertainty propagation throughout.

---

## Features

- **Spectrum object** with axis calibration, resolution calibration, and error propagation using the uncertainties package
- **Domain slicing** by physical axis values (e.g. energy in keV) or channels
- **List-mode parser** for time-channel detector data, with chunked reading for large files
- **Domain fitting and analysis** — multiple fitting and analysis methods for a domain including sum of Gaussians fitting, find peaks, centers and fwhm
- **Background estimation** — multiple global background extimation methods including Asymmetric Least Squares (ALS)
- **Extensible base classes** — build your own fitting, background, or analysis procedures with a consistent interface
- **xarray throughout** — labeled, sliceable data with named coordinates
- **Uncertainty propagation** via the `uncertainties` library

---

## Installation

PySpectrum is not yet available on PyPI. Install directly from GitHub:

```bash
git clone https://github.com/achiyaAmrusi/pySpectrum
cd pySpectrum
pip install -e .
```
This installs the package in editable mode, so any changes you make to the source are reflected immediately.

---

## Quick Start

### Load a spectrum from a DataFrame

```python
import pandas as pd
from pyspectrum.core import Spectrum

df = pd.read_csv("my_spectrum.csv")
spectrum = Spectrum.from_dataframe(df, channel_col="channel", counts_col="counts")
```

### Apply an energy calibration

```python
from pyspectrum.calibration import AxisCalibration

# Linear calibration: energy = 0.5 * channel + 1.2
calib = AxisCalibration(lambda ch: 0.5 * ch + 1.2, name="energy_keV")
spectrum.set_axis_calibration(calib)
```

### Slice a domain by axis values

```python
# Select the region between 1460 and 1480 keV
domain = spectrum.domain(1460, 1480)
```

### Fit peaks

```python
from pyspectrum.domain_fitting import SumOfGaussians

result = SumOfGaussians.fit(domain)

print(result["center"].values)   # peak centers in keV
print(result["fwhm"].values)     # peak widths
print(result["amplitude"].values) # peak amplitudes
```

### Estimate and subtract background

```python
from pyspectrum.background import ALSBackground

bg_estimator = ALSBackground(lam=1e5, p=0.001, max_iter=50)
als_bg = bg_estimator.estimate(spectrum.axis, spectrum.counts)
domain_subtracted = domains.subtract_background(als_bg[domain.indices])
```

### Parse raw list-mode data

```python
from pyspectrum.parsers import TimeChannelParser

# From a large file — processed in chunks to save memory
spectrum = TimeChannelParser.from_file(
    "detector_run.csv",
    axis_calib=calib,
    num_of_channels=2**14
)

# From an in-memory DataFrame
spectrum = TimeChannelParser.from_dataframe(df, axis_calib=calib)
```

---

## Uncertainty Propagation

PySpectrum propagates measurement uncertainties through arithmetic operations using the `uncertainties` library. Poisson errors are assigned automatically when parsing list-mode data.

```python
# Arithmetic preserves errors
subtraction = spectrum_a - spectrum_b
subtraction.counts_err  # propagated uncertainties
```

---

## Extending PySpectrum

PySpectrum is designed to be extended. Each analysis category has an abstract base class that defines the interface:

### Custom fitting method

```python
from pyspectrum.domain_fitting.abstract_fitting_class import PeakFit

class MyFitter(PeakFit):
    @classmethod
    def fit(cls, domain, **kwargs):
        # your fitting logic here
        ...
```

### Custom background estimator

```python
from pyspectrum.background.base import BackgroundEstimator

class MyBackground(BackgroundEstimator):
    def estimate(self, x, y):
        # your background logic here
        ...
```

All custom classes integrate seamlessly with `Domain`, `Spectrum`, and the rest of the pipeline.

---

## Examples

Full worked examples are available in the [examples directory](https://github.com/achiyaAmrusi/pySpectrum/tree/master/examples):

**Core**
- [Loading a spectrum](./examples/Core/loading_spectrum.ipynb) — reading and constructing a `Spectrum` from data
- [Calibration](./examples/Core/calibration.ipynb) — applying axis and resolution calibrations
- [Domain slicing](./examples/Core/domain_example.ipynb) — creating and working with domains

**Background Estimation**
- [Background subtraction](./examples/Background/background_substraction.ipynb) — estimating and subtracting background from a domain

**Domain Analysis and Fitting**
- [Domain fitting](./examples/Domain_Analysis_and_Fitting/domain_fitting.ipynb) — fitting peaks in a single domain
- [Full spectrum fitting](./examples/Domain_Analysis_and_Fitting/full_sepctrum_fitting.ipynb) — fitting peaks across an entire spectrum

**SNR Identification**
- [Peak domain identification](./examples/SNR%20Identification/peak_domin_identification.ipynb) — identifying signal regions automatically
- [Complex spectrum domains](./examples/SNR%20Identification/complex_spectrum_domains.ipynb) — handling overlapping and complex peak structures

**Library**
Sample data files for running the examples are provided in the [Library directory](./examples/Library).

---

## Requirements

- numpy>=2.0.0,<3.0
- pandas>=2.3,<4.0
- scipy>=1.14.0
- xarray>=2024.6.0
- uncertainties>=3.1

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Author

Achiya Yosef Amrusi — [GitHub](https://github.com/achiyaAmrusi/pySpectrum)

Contributions and feedback are welcome.
