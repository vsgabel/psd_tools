# psd-tools

[![Tests](https://github.com/vsgabel/psd_tools/actions/workflows/tests.yml/badge.svg)](https://github.com/vsgabel/psd_tools/actions/workflows/tests.yml)

Python utilities for computing power spectral densities (PSDs) of signals and
fields sampled on regular grids, in 1D, 2D, and 3D. Built on `numpy.fft`,
with optional Hanning windowing and variance-preserving normalization so
that integrating the PSD recovers the (windowed) data variance, per
Parseval's theorem.

## Installation

```bash
pip install -e .
```

Requires Python >= 3.10 and `numpy`.

## Features

- **1D PSD** (`psd1d`) — one-sided spectrum via `np.fft.rfft`.
- **2D PSD** (`psd2d`) — full 2D spectrum via `np.fft.fft2`.
- **3D PSD** (`psd3d`) — full 3D (time, y, x) spectrum via `np.fft.fftn`.
- **Radial averaging** (`psd_kh`) — bin a 2D PSD onto an isotropic
  horizontal wavenumber axis.
- **Plotting helpers** (`shift`, `shift_positive`) — center/scale a 2D PSD
  for log-log spectral plots.
- Optional Hanning windowing with variance correction, and `verbose` output
  to sanity-check Parseval's theorem.

## Usage

All functions are re-exported from the top-level package.

### 1D

```python
import numpy as np
from psd_tools import psd1d

x = np.random.randn(256)
P, kx, dkx = psd1d(x, dx=1.0, verbose=True)
```

### 2D

```python
import numpy as np
from psd_tools import psd2d, psd_kh, shift

field = np.random.randn(128, 128)
P, ky, kx, dky, dkx = psd2d(field, dy=1.0, dx=1.0, verbose=True)

# Radially average onto an isotropic wavenumber axis
P_kh, kh, dkh = psd_kh(P, kx, ky, dkx, dky)

# Center and log-scale for plotting
P_plot, ky_plot, kx_plot = shift(P, kx, ky)
```

### 3D

```python
import numpy as np
from psd_tools import psd3d

field = np.random.randn(50, 64, 64)  # (nt, ny, nx)
P, f, ky, kx, df, dky, dkx = psd3d(field, dt=1.0, dy=1.0, dx=1.0, verbose=True)
```

## Testing

```bash
pip install -e ".[test]"
pytest
```

The test suite checks Parseval's theorem (variance conservation) on white
noise for `psd1d`/`psd2d`/`psd3d`, and exercises the 3D and radial-averaging
paths with a synthetic Gaussian eddy (`tests/helpers.py`) — a blob
translating through the domain over time — verifying that its energy stays
conserved and concentrated at large scales / low frequencies, as expected
for a smooth, coherent structure.

## Project layout

```
src/psd_tools/
├── psd_1d.py     # psd1d
├── psd_2d.py     # psd2d, psd_kh
├── psd_3d.py     # psd3d
└── plotting.py   # shift, shift_positive
```

## License

MIT License. See [LICENSE](LICENSE).

## Commercial use

Commercial use is permitted under the MIT License.

If you use psd_tools in a commercial product or service, I would appreciate
hearing about how the software is being used. This is a request, not a
condition of the license.

🇮🇱 **Developed in Israel**