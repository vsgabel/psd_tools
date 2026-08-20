import numpy as np
import pytest

from psd_tools import psd1d

@pytest.fixture
def rng():
    return np.random.default_rng(0)

@pytest.mark.parametrize("n", [256, 257])  # even and odd length
def test_shapes(rng, n):
    x = rng.standard_normal(n)
    P, kx, dkx = psd1d(x, dx=1.0)
    assert P.shape == kx.shape == (n // 2 + 1,)
    assert dkx > 0

@pytest.mark.parametrize("hanning", [True, False])
@pytest.mark.parametrize("n", [256, 257])
def test_parseval_white_noise(rng, n, hanning):
    x = rng.standard_normal(n)
    P, kx, dkx = psd1d(x, dx=1.0, hanning=hanning)

    w = np.hanning(n) if hanning else np.ones(n)
    q = x - x.mean()
    U = np.mean(w ** 2)
    var_windowed = np.mean((w * q) ** 2) / U

    var_psd = np.sum(P) * dkx
    assert var_psd == pytest.approx(var_windowed, rel=1e-8)

def test_no_window_matches_raw_data_variance(rng):
    x = rng.standard_normal(500)
    P, kx, dkx = psd1d(x, dx=1.0, hanning=False)
    assert np.sum(P) * dkx == pytest.approx(np.var(x), rel=1e-8)
