import numpy as np
from dune.common import FieldVector
from scipy.interpolate import CubicSpline


def eval_f(params, x_vals, v_vals):
    """
    Parameters: Nufi_params, x array to eval, v array to eval
    return evaluation of f on spline
    return type np.array with shape (Nx_eval, Nv_eval)
    """
    Nx = len(x_vals)
    Nv = len(v_vals)
    f = np.zeros((Nx, Nv))
    a = FieldVector([0, 0])
    for i, x in enumerate(x_vals):
        for j, v in enumerate(v_vals):
            a[0] = wrap_periodic(x, x_vals)
            a[1] = wrap_periodic(v, v_vals)
            # a[0] = (x - x0) % params.Lx + x0
            # a[1] = (v - v0) % params.Lv + v0
            f[i, j] = params.fini(a)
    return f


def check_strict_inc(x):
    diffs = np.diff(x)
    mask = diffs <= 0  # problematic positions
    if not np.any(mask):
        print("OK: x is strictly increasing.")
        return

    bad_idxs = np.where(mask)[0]  # indices where increase fails
    for i in bad_idxs:
        print(f"Failure at index {i}: x[{i}] = {x[i]}, x[{i + 1}] = {x[i + 1]}")


def E_spline(x, y) -> CubicSpline:
    """
    periodic spline generator from points
    x, y: data points to build E_spline
    returns spline to be ealuated later
    """
    Fgrid = np.asarray(y)

    dx = x[1] - x[0]
    L = dx * len(x)
    x0 = x[0]

    # append duplicate endpoint for the spline only
    x_ext = np.concatenate([x, [x0 + L]])
    F_ext = np.concatenate([Fgrid, [Fgrid[0]]])
    check_strict_inc(x_ext)
    spline = CubicSpline(x_ext, F_ext, bc_type="periodic")
    return spline


def compute_density(fs, v_vals):
    dv = abs(v_vals[1] - v_vals[0])
    return np.sum(fs * dv, axis=1)


def vPoisson(params, fs, charge):
    """
    Solve 1D Poisson equation for the electric field given the distribution function.

    Parameters
    ----------
    fs : ndarray, shape (Nv, Nx, Ns)
        Distribution function for all species
    grids : list of Grid
        List of Grid instances, must have dv, kx, kx2
    charge : list or ndarray, length Ns
        Charges for each species

    Returns
    -------
    Efield : ndarray, shape (Nx,)
        Electric field at each spatial point
    """
    rho = np.zeros(params.Nx_eval)

    # Compute total charge density
    rho += charge * compute_density(fs, params.v_sampling_grid)

    kx = np.copy(params.kx)
    K2 = np.copy(params.kx2)

    # Solve Poisson in Fourier space
    b = np.fft.fft(1 - rho)
    phi_fft = -b / K2
    phi_fft[0] = 0  # set mean to zero (zero mode)

    # Compute electric field: E = -dphi/dx
    dphi_dx_h = -1j * phi_fft * kx
    Efield = -np.real(np.fft.ifft(dphi_dx_h))  # 1D field

    return Efield


def wrap_periodic(X, xgrid):
    dx = abs(xgrid[1] - xgrid[0])
    L = dx * len(xgrid)
    x0 = xgrid[0]
    return (X - x0) % L + x0
