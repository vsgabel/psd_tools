import numpy as np
import pytest

from psd_tools import psd3d
from helpers import moving_gaussian_eddy

@pytest.fixture
def rng():
    return np.random.default_rng(2)

def test_shapes(rng):
    cube = rng.standard_normal((12, 20, 24))
    P, f, ky, kx, df, dky, dkx = psd3d(cube, dt=1.0, dy=1.0, dx=1.0)
    assert P.shape == (12, 20, 24)
    assert f.shape == (12,)
    assert ky.shape == (20,)
    assert kx.shape == (24,)

@pytest.mark.parametrize("hanning", [True, False])
def test_parseval_white_noise(rng, hanning):
    cube = rng.standard_normal((10, 18, 22))
    P, f, ky, kx, df, dky, dkx = psd3d(cube, dt=1.0, dy=1.0, dx=1.0, hanning=hanning)

    nt, ny, nx = cube.shape
    wt = np.hanning(nt) if hanning else np.ones(nt)
    wy = np.hanning(ny) if hanning else np.ones(ny)
    wx = np.hanning(nx) if hanning else np.ones(nx)
    W = wt[:, None, None] * wy[None, :, None] * wx[None, None, :]
    q = cube - cube.mean()
    U = np.mean(W ** 2)
    var_windowed = np.mean((W * q) ** 2) / U

    var_psd = np.sum(P) * df * dky * dkx
    assert var_psd == pytest.approx(var_windowed, rel=1e-8)

def test_moving_eddy_variance_is_conserved():
    # A Gaussian blob advecting through the domain over time, as an
    # idealized eddy, should still satisfy Parseval's theorem regardless
    # of the structured (non-random) signal shape.
    field = moving_gaussian_eddy(nt=24, ny=48, nx=48, sigma=4.0, speed=(0.6, 0.3))
    P, f, ky, kx, df, dky, dkx = psd3d(field, dt=1.0, dy=1.0, dx=1.0, hanning=False)

    var_data = np.var(field - field.mean())
    var_psd = np.sum(P) * df * dky * dkx
    assert var_psd == pytest.approx(var_data, rel=1e-8)

def test_moving_eddy_energy_is_concentrated_at_low_frequency():
    # A slowly-translating, smooth eddy should be dominated by low
    # temporal frequencies, unlike white noise which is flat in frequency.
    field = moving_gaussian_eddy(nt=24, ny=48, nx=48, sigma=4.0, speed=(0.6, 0.3))
    P, f, ky, kx, df, dky, dkx = psd3d(field, dt=1.0, dy=1.0, dx=1.0)

    Pf = np.sum(P, axis=(1, 2)) * dky * dkx
    total_var = np.sum(Pf) * df

    order = np.argsort(np.abs(f))
    cum = np.cumsum(Pf[order]) * df
    low_freq_var = cum[len(f) // 4]
    assert low_freq_var / total_var > 0.9
