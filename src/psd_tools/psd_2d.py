import numpy as np

def psd2d(x, dy, dx, hanning=True, verbose=False, unit="units"):
    """Compute the 2D power spectral density (PSD) of a field on a regular grid.

    The mean is removed before transforming, an optional 2D Hanning window
    is applied (with window-power correction), and the result is normalized so
    that integrating the PSD over all wavenumbers recovers the window-corrected
    mean-square power, consistent with Parseval's theorem.

    Parameters
    ----------
    x : ndarray, shape (ny, nx)
        2D input field.
    dy, dx : float
        Grid spacing along the y (rows) and x (columns) axes.
    hanning : bool, optional
        If True (default), apply a separable 2D Hanning window before the FFT
        to reduce spectral leakage. If False, use a rectangular (all-ones)
        window.
    verbose : bool, optional
        If True, print the data variance, window-corrected mean-square power,
        PSD-integrated variance, their ratios, and whether Parseval's theorem
        is satisfied within a 5% relative tolerance.
    unit : str, optional
        Unit label used only in the verbose printout.

    Returns
    -------
    P : ndarray, shape (ny, nx)
        Two-sided 2D power spectral density.
    ky, kx : ndarray
        Wavenumber coordinates along y and x, as returned by
        ``np.fft.fftfreq``.
    dky, dkx : float
        Wavenumber bin spacing along y and x.
    """
    xmean = np.mean(x)
    q = x - xmean

    ny, nx = q.shape

    wx = np.hanning(nx)
    wy = np.hanning(ny)

    W2=wy[:,None]*wx[None,:]
    if not hanning:
        W2=np.ones_like(W2)

    U = np.mean(W2**2)
    F = np.fft.fft2(W2*q)
    kx = np.fft.fftfreq(nx, d=dx)
    ky = np.fft.fftfreq(ny, d=dy)

    dkx = np.abs(kx[1] - kx[0])
    dky = np.abs(ky[1] - ky[0])

    P = ((dy*dx)/(ny*nx*U))*np.abs(F)**2

    var_psd = np.sum(P)*dky*dkx
    var_data = np.var(q)
    var_windowed = np.mean((W2*q)**2) / U

    if verbose:
        print(f"Data variance: {var_data:.4f} {unit}²\nWindowed data variance: {var_windowed:.4f} {unit}²\nPSD variance: {var_psd:.4f} {unit}²")
        print(f"\nData/PSD ratio: {var_data/var_psd:.2f}\nWindowed data/PSD ratio: {var_windowed/var_psd:.2f}")
        print("\nParseval OK" if np.isclose(var_windowed, var_psd, rtol=0.05) else "\nParseval FAILED")

    return P, ky, kx, dky, dkx

def psd_kh(P, kx, ky, dkx, dky, verbose=False, unit="units"):
    """Azimuthally integrate a 2D PSD onto an isotropic horizontal wavenumber axis.

    Bins the 2D power spectral density ``P`` by horizontal wavenumber
    ``kh = sqrt(kx**2 + ky**2)`` using annular bins of width
    ``min(dkx, dky)``, producing a variance-preserving 1D isotropic PSD.
    The spectrum is normalized such that integrating ``P_kh`` over ``kh``
    recovers the variance contained in the 2D PSD.

    Parameters
    ----------
    P : ndarray, shape (ny, nx)
        2D power spectral density, e.g. as returned by :func:`psd2d`.
    kx, ky : ndarray
        Wavenumber coordinates along x and y matching the shape of ``P``.
    dkx, dky : float
        Wavenumber bin spacing along x and y.
    verbose : bool, optional
        If True, print the radially-integrated variance, the total PSD
        variance, their ratio, and whether variance is conserved within
        a 5% relative tolerance.
    unit : str, optional
        Unit label used only in the verbose printout.

    Returns
    -------
    P_kh : ndarray
        1D variance-preserving isotropic power spectral density.
    kh : ndarray
        Bin-center horizontal wavenumbers corresponding to ``P_kh``.
    dkh : float
        Wavenumber bin width used for the radial binning.
    """
    KX, KY = np.meshgrid(kx, ky)
    KH = np.sqrt(KX**2 + KY**2)

    dkh = min(dkx, dky)

    kh_edges = np.arange(0, KH.max()+dkh, dkh)
    kh = 0.5 * (kh_edges[1:] + kh_edges[:-1])

    P_kh = np.zeros_like(kh)
    for i in range(len(kh)):
        mask = ((KH >= kh_edges[i]) & (KH < kh_edges[i+1]))

        P_kh[i] = np.sum(P[mask])*dkx*dky/dkh

    var_radial = np.sum(P_kh) * dkh

    var_psd = (
        np.sum(P)
        * dky * dkx
    )

    if verbose:
        print(f"Data variance: {var_radial:.4f} {unit}²\nPSD variance: {var_psd:.4f} {unit}²")
        print(f"\nData/PSD ratio: {var_radial/var_psd:.2f}")
        print("\nParseval OK" if np.isclose(var_radial, var_psd, rtol=0.05) else "\nParseval FAILED")

    return P_kh, kh, dkh