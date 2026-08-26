import numpy as np

def shift(P, kx, ky, log=True, radial_weight=False,thresh=1e-8):
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
        If True (default), return ``log10(P_shift + thresh)`` instead of
        ``P_shift`` directly.
    radial_weight : bool, optional
        If True, scale the PSD by ``sqrt(kx**2 + ky**2)``.
    thresh : float, optional
        Small offset added before taking the log to avoid ``log10(0)``.

    Returns
    -------
    P_shift : ndarray, shape (ny, nx)
        Shifted and scaled (optionally log10) PSD.
    ky_plot, kx_plot : ndarray
        Shifted y and x wavenumber coordinates matching ``P_shift``.
    """

    P_shift = np.fft.fftshift(P)
    kx_plot = np.fft.fftshift(kx)
    ky_plot = np.fft.fftshift(ky)

    KX, KY = np.meshgrid(kx_plot, ky_plot)

    if radial_weight:
        P_shift = P_shift * np.sqrt(KX**2 + KY**2)
    if log:
        P_shift = np.log10(P_shift+thresh)

    return P_shift, ky_plot, kx_plot

def shift_positive(P, kx, ky, log=True, radial_weight=False, thresh=1e-8):
    """Restrict a 2D PSD to the positive-wavenumber quadrant and scale it.

    Similar to :func:`shift`, but instead of centering the full spectrum,
    selects only the strictly positive ``kx`` and ``ky`` wavenumbers, optionally
    scales the corresponding PSD values by ``sqrt(kx*2 + ky*2)`` and optionally
    takes ``log10``.

    Parameters
    ----------
    P : ndarray, shape (ny, nx)
        2D power spectral density, e.g. as returned by :func:`psd_tools.psd_2d.psd2d`.
    kx, ky : ndarray
        Wavenumber coordinates along x and y matching the shape of ``P``.
    log : bool, optional
        If True (default), return ``log10(P_pos + thresh)`` instead of
        ``P_pos`` directly.
    radial_weight : bool, optional
            If True, scale the PSD by ``sqrt(kx**2 + ky**2)``.
    thresh : float, optional
        Small offset added before taking the log to avoid ``log10(0)``.

    Returns
    -------
    P_pos : ndarray
        Scaled (optionally log10) PSD restricted to positive ``kx``/``ky``.
    ky_pos, kx_pos : ndarray
        Positive-only y and x wavenumber coordinates matching ``P_pos``.
    """
    mask_x = kx > 0
    mask_y = ky > 0

    kx_pos = kx[mask_x]
    ky_pos = ky[mask_y]

    P_pos = P[np.ix_(mask_y, mask_x)]
    KX, KY = np.meshgrid(kx_pos, ky_pos)

    if radial_weight:
        KH = np.sqrt(KX**2 + KY**2)
        P_pos = P_pos * KH
    if log:
        P_pos = np.log10(P_pos+thresh)

    return P_pos, ky_pos, kx_pos
