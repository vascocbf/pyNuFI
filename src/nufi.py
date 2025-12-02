from .fields import vPoisson, eval_f, wrap_periodic, E_spline
from scipy.interpolate import CubicSpline
import numpy as np


def NuFi(params, data, fs):
    """
    Single NuFi time step update for all species.

    Inputs:
        params - configuration object containing grids, charges, masses, etc.
        fs     - distribution function array (Nx x Nv x Ns)

    Returns:
        fs     - updated distribution function
        params - updated parameters (Efield, Efield_list)
    """
    iT = params.it + 1
    dt = params.dt
    Ns = params.Ns

    for s in range(Ns):
        charge_s = params.Charge[s]
        mass_s = params.Mass[s]
        X, V = sympl_flow_Half(
            n=iT,
            dt=dt,
            X=params.x_sampling_grid,
            V=params.v_sampling_grid,
            Efield_list=data.Efield_list,
            params=params,
            charge=charge_s,
            mass=mass_s,
        )

        # Update distribution function
        fs = eval_f(params, X, V)

    # Compute electric field
    Efield = vPoisson(params, fs, params.Charge[0])
    # Add external field
    # Efield += compute_external_Efield(params, params.grids[0].x, params.time + dt)

    # Update parameters
    data.Efield = Efield
    data.Efield_list[:, iT] = Efield

    return params, data, fs


def sympl_flow_Half(n, dt, X, V, Efield_list, params, charge, mass):
    """
    Symplectic flow for half time step in NuFi method.

    Inputs:
        n      : step number
        dt     : time step
        X, V   : position and velocity arrays
        Efield : electric field array (1D)
        grid   : grid object
        params : simulation parameters
        charge : particle charge
        mass   : particle mass
    Outputs:
        X, V   : updated position and velocity arrays
    """

    Efield_list_normed = Efield_list * (charge / mass)
    if n == 0:
        return X, V

    # Velocity field (positions advance with current velocities)
    def Ux(X_, V_):
        return V_

    # Acceleration field (velocity update from electric field)
    def Uv(X_, E):
        # Interpolate E(x) onto the X_ grid; returns shape like X_
        x_mod = wrap_periodic(X_, params.x_sampling_grid)
        spline = E_spline(X_, E)
        return spline(x_mod)

    # Full steps if n > 2
    if n > 1:
        for i in range(n - 1):
            X = X - dt * Ux(X, V)
            X = wrap_periodic(X, params.x_sampling_grid)
            # Use the corresponding past Efield if needed; here we just use current
            V = V + dt * Uv(X, Efield_list_normed[:, n - i])

    # Final half step
    X = X - dt * Ux(X, V)
    X = wrap_periodic(X, params.x_sampling_grid)
    V = V + 0.5 * dt * Uv(X, Efield_list_normed[:, 0])

    return X, V


def interp1d_periodic(xq, xgrid, Fgrid):
    """
    Periodic cubic interpolation of Fgrid(xgrid) evaluated at xq.
    - xgrid: shape (Nx_sample,), strictly periodic (no duplicated endpoint).
    - Fgrid: shape (Nx_sample,), your E field (not the distribution)
    - xq   : any shape (...), returns same shape as xq
    """
    Fgrid = np.asarray(Fgrid)
    xq = np.asarray(xq)

    dx = xgrid[1] - xgrid[0]
    L = dx * len(xgrid)
    x0 = xgrid[0]

    # wrap queries into [x0, x0+L)
    xq_mod = wrap_periodic(xq, xgrid)

    # append duplicate endpoint for the spline only
    x_ext = np.concatenate([xgrid, [x0 + L]])
    F_ext = np.concatenate([Fgrid, [Fgrid[0]]])

    spline = CubicSpline(x_ext, F_ext, bc_type="periodic")
    Fq = spline(xq_mod)  # vectorized; same shape as xq
    return Fq


def step(params, data, fs):
    """
    Time step for simulation, update parameters, field, and fs
    """

    params, data, fs = NuFi(params, data, fs)
    data.Efield_list[:, params.it] = data.Efield

    return params, data, fs
