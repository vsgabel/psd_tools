import numpy as np
import pytest

from psd_tools import psd2d, psd_kh
from helpers import moving_gaussian_eddy

@pytest.fixture
def rng():
    return np.random.default_rng(1)

def test_shapes(rng):
    field = rng.standard_normal((48, 64))
    P, ky, kx, dky, dkx = psd2d(field, dy=1.0, dx=1.0)
    assert P.shape == (48, 64)
    assert ky.shape == (48,)
    assert kx.shape == (64,)

@pytest.mark.parametrize("hanning", [True, False])
def test_parseval_white_noise(rng, hanning):
    field = rng.standard_normal((50, 60))
    P, ky, kx, dky, dkx = psd2d(field, dy=1.0, dx=1.0, hanning=hanning)

    ny, nx = field.shape
    wy = np.hanning(ny) if hanning else np.ones(ny)
    wx = np.hanning(nx) if hanning else np.ones(nx)
    W = wy[:, None] * wx[None, :]
    q = field - field.mean()
    U = np.mean(W ** 2)
    var_windowed = np.mean((W * q) ** 2) / U

    var_psd = np.sum(P) * dky * dkx
    assert var_psd == pytest.approx(var_windowed, rel=1e-8)

def test_psd_kh_conserves_variance(rng):
    field = rng.standard_normal((50, 60))
    P, ky, kx, dky, dkx = psd2d(field, dy=1.0, dx=1.0)
    P_kh, kh, dkh = psd_kh(P, kx, ky, dkx, dky)

    var_2d = np.sum(P) * dky * dkx
    var_radial = np.sum(P_kh) * dkh
    assert var_radial == pytest.approx(var_2d, rel=0.05)

def test_gaussian_eddy_energy_is_concentrated_at_large_scales():
    # A smooth, spatially-compact eddy should carry most of its variance
    # at low wavenumbers (large scales), unlike white noise.
    field = moving_gaussian_eddy(nt=1, ny=64, nx=64, sigma=4.0)[0]
    P, ky, kx, dky, dkx = psd2d(field, dy=1.0, dx=1.0)
    P_kh, kh, dkh = psd_kh(P, kx, ky, dkx, dky)

    total_var = np.sum(P_kh) * dkh
    low_k_var = np.sum(P_kh[: len(kh) // 8]) * dkh
    assert low_k_var / total_var > 0.9
