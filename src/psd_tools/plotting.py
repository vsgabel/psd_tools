import numpy as np

def shift(P, kx, ky, log=True, thresh=1e-8):
    """Center a 2D PSD in wavenumber space and scale it for plotting.

    Applies ``np.fft.fftshift`` to ``P``, ``kx``, and ``ky`` so that the
    zero wavenumber lies at the center of the array, then multiplies the
    PSD by ``|kx|*|ky|`` (a variance-preserving flux-form scaling useful
    for log-log spectral plots), and optionally takes ``log10``.

    Parameters
    ----------
    P : ndarray, shape (ny, nx)
        2D power spectral density, e.g. as returned by :func:`psd_tools.psd_2d.psd2d`.
    kx, ky : ndarray
        Wavenumber coordinates along x and y matching the shape of ``P``.
    log : bool, optional
        If True (default), return ``log10(P_prep + thresh)`` instead of
        ``P_prep`` directly.
    thresh : float, optional
        Small offset added before taking the log to avoid ``log10(0)``.

    Returns
    -------
    P_prep : ndarray, shape (ny, nx)
        Shifted and scaled (optionally log10) PSD.
    ky_plot, kx_plot : ndarray
        Shifted y and x wavenumber coordinates matching ``P_prep``.
    """

    P_shift = np.fft.fftshift(P)
    kx_plot = np.fft.fftshift(kx)
    ky_plot = np.fft.fftshift(ky)

    KX, KY = np.meshgrid(kx_plot, ky_plot)

    P_prep = P_shift*np.abs(KX)*np.abs(KY)
    if log:
        P_prep = np.log10(P_prep+thresh)

    return P_prep, ky_plot, kx_plot

def shift_positive(P, kx, ky, log=True, thresh=1e-8):
    """Restrict a 2D PSD to the positive-wavenumber quadrant and scale it.

    Similar to :func:`shift`, but instead of centering the full spectrum,
    selects only the strictly positive ``kx`` and ``ky`` wavenumbers,
    scales the corresponding PSD values by ``|kx|*|ky|``, and optionally
    takes ``log10``.

    Parameters
    ----------
    P : ndarray, shape (ny, nx)
        2D power spectral density, e.g. as returned by :func:`psd_tools.psd_2d.psd2d`.
    kx, ky : ndarray
        Wavenumber coordinates along x and y matching the shape of ``P``.
    log : bool, optional
        If True (default), return ``log10(P_prep + thresh)`` instead of
        ``P_prep`` directly.
    thresh : float, optional
        Small offset added before taking the log to avoid ``log10(0)``.

    Returns
    -------
    P_prep : ndarray
        Scaled (optionally log10) PSD restricted to positive ``kx``/``ky``.
    ky_pos, kx_pos : ndarray
        Positive-only y and x wavenumber coordinates matching ``P_prep``.
    """
    mask_x = kx > 0
    mask_y = ky > 0

    kx_pos = kx[mask_x]
    ky_pos = ky[mask_y]

    P_pos = P[np.ix_(mask_y, mask_x)]
    KX, KY = np.meshgrid(kx_pos, ky_pos)

    P_prep = P_pos*np.abs(KX)*np.abs(KY)
    if log:
        P_prep = np.log10(P_prep+thresh)

    return P_prep, ky_pos, kx_pos
