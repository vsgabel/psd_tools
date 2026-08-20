import numpy as np
import pytest

from psd_tools import psd2d, shift, shift_positive

@pytest.fixture
def spectrum():
    rng = np.random.default_rng(3)
    field = rng.standard_normal((40, 50))
    return psd2d(field, dy=1.0, dx=1.0)

def test_shift_centers_zero_wavenumber(spectrum):
    P, ky, kx, dky, dkx = spectrum
    P_prep, ky_plot, kx_plot = shift(P, kx, ky, log=False)

    assert P_prep.shape == P.shape
    assert ky_plot[np.argmin(np.abs(ky_plot))] == pytest.approx(0.0)
    assert kx_plot[np.argmin(np.abs(kx_plot))] == pytest.approx(0.0)

def test_shift_log_matches_unlogged(spectrum):
    P, ky, kx, dky, dkx = spectrum
    P_lin, _, _ = shift(P, kx, ky, log=False)
    P_log, _, _ = shift(P, kx, ky, log=True, thresh=1e-8)
    assert np.allclose(P_log, np.log10(P_lin + 1e-8))

def test_shift_positive_selects_positive_quadrant(spectrum):
    P, ky, kx, dky, dkx = spectrum
    P_prep, ky_pos, kx_pos = shift_positive(P, kx, ky, log=False)

    assert np.all(ky_pos > 0)
    assert np.all(kx_pos > 0)
    assert P_prep.shape == (ky_pos.size, kx_pos.size)
