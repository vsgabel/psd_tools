import numpy as np

def moving_gaussian_eddy(nt, ny, nx, dt=1.0, dy=1.0, dx=1.0, sigma=4.0,
                          speed=(0.6, 0.3), amplitude=1.0):
    """Synthetic (nt, ny, nx) field: a Gaussian blob translating at constant
    velocity, as an idealized model of an eddy advecting through space.

    ``speed`` is ``(vy, vx)`` in grid units per time step. Choose it (along
    with ``nt``, ``ny``, ``nx``) so the blob stays within the domain.
    """
    y = np.arange(ny) * dy
    x = np.arange(nx) * dx
    Y, X = np.meshgrid(y, x, indexing="ij")

    vy, vx = speed
    y0, x0 = (ny * dy) / 2.0, (nx * dx) / 2.0

    field = np.empty((nt, ny, nx))
    for it in range(nt):
        yc = y0 + vy * it * dt
        xc = x0 + vx * it * dt
        field[it] = amplitude * np.exp(
            -(((Y - yc) ** 2 + (X - xc) ** 2) / (2 * sigma ** 2))
        )
    return field
