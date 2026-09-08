import numpy as np

def psd3d(x, dt, dy, dx, hanning=True, verbose=False, unit="units"):
    """Compute the 3D power spectral density (PSD) of a field on a regular grid.

    The mean is removed before transforming, an optional 3D Hanning window
    (time x, y) is applied (with variance correction), and the result is
    normalized so that summing the PSD over all frequencies/wavenumbers
    recovers the (windowed) data variance, per Parseval's theorem.

    Parameters
    ----------
    x : ndarray, shape (nt, ny, nx)
        3D input field, with the first axis treated as time.
    dt : float
        Sample spacing along the time axis.
    dy, dx : float
        Grid spacing along the y (rows) and x (columns) axes.
    hanning : bool, optional
        If True (default), apply a 3D Hanning window before the FFT to
        reduce spectral leakage. If False, use a rectangular (all-ones)
        window.
    verbose : bool, optional
        If True, print the data variance, windowed data variance, PSD
        variance, their ratios, and whether Parseval's theorem is
        satisfied within a 5% relative tolerance.
    unit : str, optional
        Unit label used only in the verbose printout.

    Returns
    -------
    P : ndarray, shape (nt, ny, nx)
        3D power spectral density.
    f, ky, kx : ndarray
        Frequency (wavenumber) coordinates along time, y, and x, as
        returned by ``np.fft.fftfreq``.
    df, dky, dkx : float
        Frequency/wavenumber bin spacing along time, y, and x.
    """
    xmean = np.mean(x)
    q = x - xmean

    nt, ny, nx = q.shape

    wt = np.hanning(nt)
    wx = np.hanning(nx)
    wy = np.hanning(ny)

    W3=wt[:,None,None]*wy[None,:,None]*wx[None,None,:]
    if not hanning:
        W3=np.ones_like(W3)

    U = np.mean(W3**2)
    F = np.fft.fftn(W3*q)
    f = np.fft.fftfreq(nt, d=dt)
    kx = np.fft.fftfreq(nx, d=dx)
    ky = np.fft.fftfreq(ny, d=dy)

    df = np.abs(f[1] - f[0])
    dkx = np.abs(kx[1] - kx[0])
    dky = np.abs(ky[1] - ky[0])

    P = ((dt*dy*dx)/(nt*ny*nx*U))*np.abs(F)**2

    var_psd = np.sum(P)*df*dky*dkx
    var_data = np.var(q)
    var_windowed = np.mean((W3*q)**2) / U

    if verbose:
        print(f"Data variance: {var_data:.4f} {unit}²\nWindowed data variance: {var_windowed:.4f} {unit}²\nPSD variance: {var_psd:.4f} {unit}²")
        print(f"\nData/PSD ratio: {var_data/var_psd:.2f}\nWindowed data/PSD ratio: {var_windowed/var_psd:.2f}")
        print("\nParseval OK" if np.isclose(var_windowed, var_psd, rtol=0.05) else "\nParseval FAILED")

    return P, f, ky, kx, df, dky, dkx

def psd_kh_f(P, kx, ky, dkx, dky, df, verbose=False, unit="units"):
    """
    Convert a 3D PSD P(f, ky, kx) into a 2D PSD P(f, kh) by azimuthally integrating the horizontal wavenumbers.

    Parameters
    ----------
    P : ndarray
        3D PSD with shape (nf, ny, nx).
    kx, ky : ndarray
        Horizontal wavenumber coordinates.
    dkx, dky : float
        Horizontal wavenumber spacings.
    df : float
        Frequency spacing.
    verbose : bool
        If True, check variance conservation.
    unit : str
        Unit label used in the variance check.

    Returns
    -------
    P_kh_f : ndarray
        PSD as a function of frequency and isotropic horizontal wavenumber.
    kh : ndarray
        Horizontal isotropic wavenumber bin centers.
    dkh : float
        Horizontal wavenumber bin spacing.
    """

    KX, KY = np.meshgrid(kx, ky)
    KH = np.sqrt(KX**2 + KY**2)

    dkh = min(abs(dkx), abs(dky))
    kh_edges = np.arange(0, KH.max() + dkh, dkh)
    kh = 0.5 * (kh_edges[:-1] + kh_edges[1:])

    P_kh_f = np.zeros((P.shape[0], len(kh)))

    for j in range(len(kh)):
        mask = (KH >= kh_edges[j]) & (KH < kh_edges[j + 1])
        P_kh_f[:, j] = np.sum(P[:, mask], axis=1) * abs(dkx * dky) / dkh

    if verbose:
        var_original = np.sum(P) * df * abs(dky * dkx)
        var_radial = np.sum(P_kh_f) * df * dkh

        print(f"Original PSD variance: {var_original:.6e} {unit}²")
        print(f"Radial PSD variance:   {var_radial:.6e} {unit}²")
        print(f"Original/Radial ratio: {var_original / var_radial:.6f}")

        if np.isclose(var_original, var_radial, rtol=0.05):
            print("Parseval OK")
        else:
            print("Parseval FAILED")

    return P_kh_f, kh, dkh