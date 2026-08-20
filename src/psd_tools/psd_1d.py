import numpy as np

def psd1d(x, dx, hanning=True, verbose=False, unit="units"):
    """Compute the 1D power spectral density (PSD) of a signal on a regular grid.

    The mean is removed before transforming, an optional Hanning window is
    applied (with variance correction), and a one-sided spectrum is
    returned via ``np.fft.rfft`` with power doubled at all non-DC/Nyquist
    frequencies so that summing the PSD over all wavenumbers recovers the
    (windowed) data variance, per Parseval's theorem.

    Parameters
    ----------
    x : ndarray, shape (nx,)
        1D input signal.
    dx : float
        Sample spacing along x.
    hanning : bool, optional
        If True (default), apply a Hanning window before the FFT to
        reduce spectral leakage. If False, use a rectangular (all-ones)
        window.
    verbose : bool, optional
        If True, print the data variance, windowed data variance, PSD
        variance, and their ratios.
    unit : str, optional
        Unit label used only in the verbose printout.

    Returns
    -------
    P : ndarray
        One-sided power spectral density.
    kx : ndarray
        Non-negative frequency (wavenumber) coordinates, as returned by
        ``np.fft.rfftfreq``.
    dkx : float
        Wavenumber bin spacing.
    """
    xmean = np.mean(x)
    q = x - xmean

    nx = q.shape[0]

    wx = np.hanning(nx)
    if not hanning:
        wx=np.ones_like(wx)

    U = np.mean(wx**2)
    F = np.fft.rfft(wx*q)
    kx = np.fft.rfftfreq(nx, d=dx)

    dkx = np.abs(kx[1] - kx[0])

    P = ((dx)/(nx*U))*np.abs(F)**2

    if nx % 2 == 0:
        P[1:-1] *= 2
    else:
        P[1:] *= 2 

    var_psd = np.sum(P)*dkx
    var_data = np.var(q)
    var_windowed = np.mean((wx*q)**2) / U

    if verbose:
        print(f"Data variance: {var_data:.4f} {unit}²\nWindowed data variance: {var_windowed:.4f} {unit}²\nPSD variance: {var_psd:.4f} {unit}²")
        print(f"\nData/PSD ratio: {var_data/var_psd:.2f}\nWindowed data/PSD ratio: {var_windowed/var_psd:.2f}")
    
    return P, kx, dkx